//! RESEARCH SPIKE — Cranelift JIT codegen (opt-in `--features jit`). Goal: at load time, compile a kernel specialised
//! to the loaded model (shapes / group size / scales baked in as constants) and see whether it beats the hand-written
//! SIMD kernels. Pure-Rust codegen, no LLVM/C. Strictly experimental; the default build never touches this.
//!
//! Prior (post the decode-is-memory-bound finding): JIT can't beat a bandwidth wall, so the only realistic target is a
//! COMPUTE-bound path — the int4 dequant-dot. This module first validates the Cranelift pipeline (`jit_add_const`),
//! then JITs an int4 dot to measure against `i4_dot` (the AVX2 hand kernel).

use cranelift_codegen::ir::condcodes::IntCC;
use cranelift_codegen::ir::immediates::{Ieee32, Offset32};
use cranelift_codegen::ir::{types, AbiParam, ConstantData, Endianness, InstBuilder, MemFlags};
use cranelift_codegen::settings::{self, Configurable};
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext};
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{Linkage, Module};

/// Build a fresh optimising JIT module for the host ISA (`opt_level=speed`, so the comparison vs the hand kernel is fair).
fn new_module() -> JITModule {
    let mut fb = settings::builder();
    fb.set("opt_level", "speed").unwrap();
    let isa = cranelift_native::builder()
        .expect("host ISA")
        .finish(settings::Flags::new(fb))
        .unwrap();
    JITModule::new(JITBuilder::with_isa(isa, cranelift_module::default_libcall_names()))
}

/// Smoke test: JIT `fn(i64) -> i64` returning `x + c` with `c` baked in as a constant. Confirms codegen + finalize +
/// the transmute-to-fn-pointer round-trip work on this host.
pub fn jit_add_const(c: i64) -> extern "C" fn(i64) -> i64 {
    let mut module = new_module();
    let mut ctx = module.make_context();
    ctx.func.signature.params.push(AbiParam::new(types::I64));
    ctx.func.signature.returns.push(AbiParam::new(types::I64));
    let mut fbctx = FunctionBuilderContext::new();
    {
        let mut b = FunctionBuilder::new(&mut ctx.func, &mut fbctx);
        let blk = b.create_block();
        b.append_block_params_for_function_params(blk);
        b.switch_to_block(blk);
        b.seal_block(blk);
        let x = b.block_params(blk)[0];
        let cv = b.ins().iconst(types::I64, c);
        let sum = b.ins().iadd(x, cv);
        b.ins().return_(&[sum]);
        b.finalize();
    }
    let id = module.declare_function("add_const", Linkage::Export, &ctx.func.signature).unwrap();
    module.define_function(id, &mut ctx).unwrap();
    module.clear_context(&mut ctx);
    module.finalize_definitions().unwrap();
    let code = module.get_finalized_function(id);
    // SAFETY: signature matches the JIT'd function; the JITModule is leaked (kept alive for the process) so `code` stays valid.
    std::mem::forget(module);
    unsafe { std::mem::transmute::<*const u8, extern "C" fn(i64) -> i64>(code) }
}

/// The JIT'd int4 dequant-dot. Signature mirrors `i4_dot(prow, a, scale, g, k)` but with `g`/`k` baked into the code:
/// `(prow: *const u8, a: *const f32, scale: *const f32) -> f32`. Caller guarantees the slices are the right length
/// (`prow >= ceil(k/2)` bytes, `a >= k` f32, `scale >= ceil(k/g)` f32).
pub type I4DotFn = extern "C" fn(*const u8, *const f32, *const f32) -> f32;

/// JIT a `(k, g)`-specialised int4 dequant-dot. The bet: bake every loop bound, every nibble's byte-offset and
/// lo/hi position, and every group boundary as a compile-time constant, fully unrolled — no `kk/2`, no `kk%2`
/// branch, no loop counters at run time. This per-`(k,g)` kernel is shared across ALL rows of a weight matrix, so its
/// compile cost amortises over `n_rows` (e.g. 4864 down-proj rows) — a slow compile is fine, a slow inner loop is not.
///
/// Matches `i4_dot`'s scalar semantics exactly: signed 4-bit nibble (even `kk` = low nibble, odd = high), per-group
/// `Σ a·nib`, then `Σ scale_g · gsum`. FP add order is the same left-to-right fold (fmul+fadd, not fused), so results
/// match the scalar reference to f32 rounding.
pub fn jit_i4_dot(k: usize, g: usize) -> (I4DotFn, std::time::Duration) {
    let t0 = std::time::Instant::now();
    let mut module = new_module();
    let mut ctx = module.make_context();
    // params: prow (ptr), a (ptr), scale (ptr) — all I64 addresses; return F32.
    let ptr = types::I64;
    ctx.func.signature.params.push(AbiParam::new(ptr));
    ctx.func.signature.params.push(AbiParam::new(ptr));
    ctx.func.signature.params.push(AbiParam::new(ptr));
    ctx.func.signature.returns.push(AbiParam::new(types::F32));
    let mut fbctx = FunctionBuilderContext::new();
    {
        let mut b = FunctionBuilder::new(&mut ctx.func, &mut fbctx);
        let blk = b.create_block();
        b.append_block_params_for_function_params(blk);
        b.switch_to_block(blk);
        b.seal_block(blk);
        let prow = b.block_params(blk)[0];
        let a = b.block_params(blk)[1];
        let scale = b.block_params(blk)[2];
        let flags = MemFlags::trusted(); // a/scale are 4-aligned slice bases; byte loads are always aligned
        let mut sum = b.ins().f32const(Ieee32::with_bits(0));
        let mut gi = 0usize;
        let mut kk = 0usize;
        while kk < k {
            let hi = (kk + g).min(k);
            let mut gsum = b.ins().f32const(Ieee32::with_bits(0));
            while kk < hi {
                // byte loaded as signed I8 so the high-nibble sign bit (bit 7) is in place for an arithmetic shift.
                let byte = b.ins().load(types::I8, flags, prow, Offset32::new((kk / 2) as i32));
                // sign-extend the 4-bit nibble to I8: even -> (byte<<4)>>4 ; odd -> byte>>4 (sign bit already at bit 7).
                let snib = if kk % 2 == 0 {
                    let up = b.ins().ishl_imm(byte, 4);
                    b.ins().sshr_imm(up, 4)
                } else {
                    b.ins().sshr_imm(byte, 4)
                };
                let wi = b.ins().sextend(types::I32, snib);
                let wf = b.ins().fcvt_from_sint(types::F32, wi);
                let af = b.ins().load(types::F32, flags, a, Offset32::new((kk * 4) as i32));
                let prod = b.ins().fmul(wf, af);
                gsum = b.ins().fadd(gsum, prod);
                kk += 1;
            }
            let sc = b.ins().load(types::F32, flags, scale, Offset32::new((gi * 4) as i32));
            let scaled = b.ins().fmul(sc, gsum);
            sum = b.ins().fadd(sum, scaled);
            gi += 1;
        }
        b.ins().return_(&[sum]);
        b.finalize();
    }
    let id = module.declare_function("i4_dot_jit", Linkage::Export, &ctx.func.signature).unwrap();
    module.define_function(id, &mut ctx).unwrap();
    module.clear_context(&mut ctx);
    module.finalize_definitions().unwrap();
    let code = module.get_finalized_function(id);
    std::mem::forget(module); // leak the JITModule so `code` stays mapped for the process lifetime
    // SAFETY: signature matches I4DotFn; module is leaked so the code page stays live.
    let f = unsafe { std::mem::transmute::<*const u8, I4DotFn>(code) };
    (f, t0.elapsed())
}

/// Like `jit_i4_dot` but the SMART JIT shape: a real Cranelift loop over the `ng` groups (so code size is O(g), not
/// O(k) — no i-cache blowout, ~constant compile time), with only the `g`-wide inner body unrolled. Bakes `g`/`ng` as
/// constants. Requires `k % g == 0` (group-quant weights are padded to a multiple of `g`).
pub fn jit_i4_dot_looped(k: usize, g: usize) -> (I4DotFn, std::time::Duration) {
    assert!(k % g == 0 && g % 2 == 0, "looped JIT needs k%g==0 and even g");
    let ng = k / g;
    let t0 = std::time::Instant::now();
    let mut module = new_module();
    let mut ctx = module.make_context();
    let ptr = types::I64;
    ctx.func.signature.params.push(AbiParam::new(ptr));
    ctx.func.signature.params.push(AbiParam::new(ptr));
    ctx.func.signature.params.push(AbiParam::new(ptr));
    ctx.func.signature.returns.push(AbiParam::new(types::F32));
    let mut fbctx = FunctionBuilderContext::new();
    {
        let mut b = FunctionBuilder::new(&mut ctx.func, &mut fbctx);
        let entry = b.create_block();
        let header = b.create_block(); // loop header, params: (gi: i64, sum: f32)
        let body = b.create_block();
        let exit = b.create_block(); // param: (sum: f32)
        b.append_block_params_for_function_params(entry);
        b.append_block_param(header, types::I64);
        b.append_block_param(header, types::F32);
        b.append_block_param(exit, types::F32);
        let flags = MemFlags::trusted();

        // entry: sum=0, gi=0, jump header
        b.switch_to_block(entry);
        b.seal_block(entry);
        let prow = b.block_params(entry)[0];
        let a = b.block_params(entry)[1];
        let scale = b.block_params(entry)[2];
        let zero_f = b.ins().f32const(Ieee32::with_bits(0));
        let zero_i = b.ins().iconst(types::I64, 0);
        b.ins().jump(header, &[zero_i, zero_f]);

        // header: if gi >= ng -> exit(sum) else body
        b.switch_to_block(header);
        let gi = b.block_params(header)[0];
        let sum = b.block_params(header)[1];
        let ngv = b.ins().iconst(types::I64, ng as i64);
        let done = b.ins().icmp(IntCC::UnsignedGreaterThanOrEqual, gi, ngv);
        b.ins().brif(done, exit, &[sum], body, &[]);

        // body: compute this group's base pointers from gi, unroll g nibbles, accumulate, loop back
        b.switch_to_block(body);
        b.seal_block(body);
        let off_p = b.ins().imul_imm(gi, (g / 2) as i64); // bytes: gi * g/2
        let pb = b.ins().iadd(prow, off_p);
        let off_a = b.ins().imul_imm(gi, (g * 4) as i64); // f32 bytes: gi * g * 4
        let pa = b.ins().iadd(a, off_a);
        let off_s = b.ins().imul_imm(gi, 4);
        let ps = b.ins().iadd(scale, off_s);
        let mut gsum = b.ins().f32const(Ieee32::with_bits(0));
        for j in 0..g {
            let byte = b.ins().load(types::I8, flags, pb, Offset32::new((j / 2) as i32));
            let snib = if j % 2 == 0 {
                let up = b.ins().ishl_imm(byte, 4);
                b.ins().sshr_imm(up, 4)
            } else {
                b.ins().sshr_imm(byte, 4)
            };
            let wi = b.ins().sextend(types::I32, snib);
            let wf = b.ins().fcvt_from_sint(types::F32, wi);
            let af = b.ins().load(types::F32, flags, pa, Offset32::new((j * 4) as i32));
            let prod = b.ins().fmul(wf, af);
            gsum = b.ins().fadd(gsum, prod);
        }
        let sc = b.ins().load(types::F32, flags, ps, Offset32::new(0));
        let scaled = b.ins().fmul(sc, gsum);
        let newsum = b.ins().fadd(sum, scaled);
        let gi1 = b.ins().iadd_imm(gi, 1);
        b.ins().jump(header, &[gi1, newsum]);

        // header's predecessors (entry, body) are now both emitted; seal it.
        b.seal_block(header);
        b.switch_to_block(exit);
        b.seal_block(exit);
        let res = b.block_params(exit)[0];
        b.ins().return_(&[res]);
        b.finalize();
    }
    let id = module.declare_function("i4_dot_jit_looped", Linkage::Export, &ctx.func.signature).unwrap();
    module.define_function(id, &mut ctx).unwrap();
    module.clear_context(&mut ctx);
    module.finalize_definitions().unwrap();
    let code = module.get_finalized_function(id);
    std::mem::forget(module);
    let f = unsafe { std::mem::transmute::<*const u8, I4DotFn>(code) };
    (f, t0.elapsed())
}

/// The JIT's BEST SHOT: a 128-bit-SIMD int4 dot emitted as Cranelift *vector* IR (the scalar/looped JITs above can't
/// beat 256-bit AVX2 because Cranelift has no autovectorizer — anything in scalar IR stays scalar). This hand-writes
/// the vector kernel in Cranelift IR: widen bytes to i16 lanes, extract+sign-extend both nibbles per byte via 16-bit
/// shifts (SSE has no byte-lane shift), `shuffle` them into kk-order (the `unpacklo/unpackhi` the hand AVX2 kernel
/// does), widen to f32x4, FMA against contiguous activations, hsum per group. Requires `g % 32 == 0`, `k % g == 0`.
///
/// Ceiling: Cranelift's SIMD is 128-bit (F32X4 / I16X8) — *half* the AVX2 hand kernel's 256-bit width — so even a
/// perfect lowering tops out near half AVX2 throughput. This measures whether codegen can close even that much.
pub fn jit_i4_dot_vec(k: usize, g: usize) -> (I4DotFn, std::time::Duration) {
    assert!(g % 32 == 0 && k % g == 0, "vector JIT needs g%32==0 and k%g==0");
    let ng = k / g;
    let blocks = g / 32; // 16-byte / 32-nibble vector blocks per group
    // unpacklo/unpackhi of two I16X8 (as byte-shuffle masks; src `a`=bytes 0..15, src `b`=bytes 16..31).
    let mask_lo: Vec<u8> = vec![0, 1, 16, 17, 2, 3, 18, 19, 4, 5, 20, 21, 6, 7, 22, 23];
    let mask_hi: Vec<u8> = vec![8, 9, 24, 25, 10, 11, 26, 27, 12, 13, 28, 29, 14, 15, 30, 31];
    let t0 = std::time::Instant::now();
    let mut module = new_module();
    let mut ctx = module.make_context();
    let ptr = types::I64;
    for _ in 0..3 {
        ctx.func.signature.params.push(AbiParam::new(ptr));
    }
    ctx.func.signature.returns.push(AbiParam::new(types::F32));
    let mut fbctx = FunctionBuilderContext::new();
    {
        let mut b = FunctionBuilder::new(&mut ctx.func, &mut fbctx);
        let m_lo = b.func.dfg.immediates.push(ConstantData::from(mask_lo));
        let m_hi = b.func.dfg.immediates.push(ConstantData::from(mask_hi));
        let (i8x16, i16x8, f32x4) = (types::I8X16, types::I16X8, types::F32X4);
        let entry = b.create_block();
        let header = b.create_block();
        let body = b.create_block();
        let exit = b.create_block();
        b.append_block_params_for_function_params(entry);
        b.append_block_param(header, types::I64);
        b.append_block_param(header, types::F32);
        b.append_block_param(exit, types::F32);
        let vf = MemFlags::new(); // vector + f32 loads are only 4-aligned -> unaligned moves
        let bc = MemFlags::new().with_endianness(Endianness::Little); // lane-count-changing bitcast needs a byte order

        b.switch_to_block(entry);
        b.seal_block(entry);
        let prow = b.block_params(entry)[0];
        let a = b.block_params(entry)[1];
        let scale = b.block_params(entry)[2];
        let zf = b.ins().f32const(Ieee32::with_bits(0));
        let zi = b.ins().iconst(types::I64, 0);
        b.ins().jump(header, &[zi, zf]);

        b.switch_to_block(header);
        let gi = b.block_params(header)[0];
        let sum = b.block_params(header)[1];
        let ngv = b.ins().iconst(types::I64, ng as i64);
        let done = b.ins().icmp(IntCC::UnsignedGreaterThanOrEqual, gi, ngv);
        b.ins().brif(done, exit, &[sum], body, &[]);

        b.switch_to_block(body);
        b.seal_block(body);
        let off_p = b.ins().imul_imm(gi, (g / 2) as i64);
        let pb = b.ins().iadd(prow, off_p);
        let off_a = b.ins().imul_imm(gi, (g * 4) as i64);
        let pa = b.ins().iadd(a, off_a);
        let off_s = b.ins().imul_imm(gi, 4);
        let ps = b.ins().iadd(scale, off_s);
        let z = b.ins().f32const(Ieee32::with_bits(0));
        let mut accv = b.ins().splat(f32x4, z); // f32x4 zero accumulator
        for blk in 0..blocks {
            let bytes = b.ins().load(i8x16, vf, pb, Offset32::new((blk * 16) as i32));
            for chunk in 0..2 {
                // widen the chunk's 8 bytes to i16 lanes (raw 0..255)
                let w = if chunk == 0 { b.ins().uwiden_low(bytes) } else { b.ins().uwiden_high(bytes) };
                // signed low nibble: ((w<<12)>>12); signed high nibble: ((w<<8)>>12)  — 16-bit-lane shifts (psllw/psraw)
                let slo = { let u = b.ins().ishl_imm(w, 12); b.ins().sshr_imm(u, 12) };
                let shi = { let u = b.ins().ishl_imm(w, 8); b.ins().sshr_imm(u, 12) };
                // interleave to kk-order: klo = n0..n7, khi = n8..n15 (shuffle operates on bytes -> bitcast in/out)
                let slo8 = b.ins().bitcast(i8x16, bc, slo);
                let shi8 = b.ins().bitcast(i8x16, bc, shi);
                let klo8 = b.ins().shuffle(slo8, shi8, m_lo);
                let khi8 = b.ins().shuffle(slo8, shi8, m_hi);
                let klo = b.ins().bitcast(i16x8, bc, klo8);
                let khi = b.ins().bitcast(i16x8, bc, khi8);
                // widen each i16x8 -> 2x i32x4 -> f32x4
                let klo_lo = b.ins().swiden_low(klo);
                let klo_hi = b.ins().swiden_high(klo);
                let khi_lo = b.ins().swiden_low(khi);
                let khi_hi = b.ins().swiden_high(khi);
                let f0 = b.ins().fcvt_from_sint(f32x4, klo_lo);
                let f1 = b.ins().fcvt_from_sint(f32x4, klo_hi);
                let f2 = b.ins().fcvt_from_sint(f32x4, khi_lo);
                let f3 = b.ins().fcvt_from_sint(f32x4, khi_hi);
                // activations: this chunk covers 16 contiguous f32 at a-offset blk*128 + chunk*64
                let ao = (blk * 128 + chunk * 64) as i32;
                let a0 = b.ins().load(f32x4, vf, pa, Offset32::new(ao));
                let a1 = b.ins().load(f32x4, vf, pa, Offset32::new(ao + 16));
                let a2 = b.ins().load(f32x4, vf, pa, Offset32::new(ao + 32));
                let a3 = b.ins().load(f32x4, vf, pa, Offset32::new(ao + 48));
                accv = b.ins().fma(f0, a0, accv);
                accv = b.ins().fma(f1, a1, accv);
                accv = b.ins().fma(f2, a2, accv);
                accv = b.ins().fma(f3, a3, accv);
            }
        }
        // horizontal sum of the f32x4 accumulator
        let l0 = b.ins().extractlane(accv, 0);
        let l1 = b.ins().extractlane(accv, 1);
        let l2 = b.ins().extractlane(accv, 2);
        let l3 = b.ins().extractlane(accv, 3);
        let s01 = b.ins().fadd(l0, l1);
        let s23 = b.ins().fadd(l2, l3);
        let gsum = b.ins().fadd(s01, s23);
        let sc = b.ins().load(types::F32, MemFlags::trusted(), ps, Offset32::new(0));
        let scaled = b.ins().fmul(sc, gsum);
        let newsum = b.ins().fadd(sum, scaled);
        let gi1 = b.ins().iadd_imm(gi, 1);
        b.ins().jump(header, &[gi1, newsum]);

        b.seal_block(header);
        b.switch_to_block(exit);
        b.seal_block(exit);
        let res = b.block_params(exit)[0];
        b.ins().return_(&[res]);
        b.finalize();
    }
    let id = module.declare_function("i4_dot_jit_vec", Linkage::Export, &ctx.func.signature).unwrap();
    module.define_function(id, &mut ctx).unwrap();
    module.clear_context(&mut ctx);
    module.finalize_definitions().unwrap();
    let code = module.get_finalized_function(id);
    std::mem::forget(module);
    let f = unsafe { std::mem::transmute::<*const u8, I4DotFn>(code) };
    (f, t0.elapsed())
}

/// Scalar reference int4 dequant-dot — a copy of `i4_dot`'s scalar fold, used to validate the JIT independent of the
/// AVX2 dispatch (and as the "scalar" bar in the bench).
fn i4_dot_scalar_ref(prow: &[u8], a: &[f32], scale: &[f32], g: usize, k: usize) -> f32 {
    let mut sum = 0.0f32;
    let (mut gi, mut kk) = (0usize, 0usize);
    while kk < k {
        let hi = (kk + g).min(k);
        let mut gsum = 0.0f32;
        while kk < hi {
            let byte = prow[kk / 2];
            let nib = if kk % 2 == 0 { byte & 0x0F } else { byte >> 4 };
            gsum += a[kk] * ((((nib << 4) as i8) >> 4) as f32);
            kk += 1;
        }
        sum += scale[gi] * gsum;
        gi += 1;
    }
    sum
}

/// Research-spike bench: JIT an int4 dot specialised to `(k, g)`, validate it against the scalar reference and the
/// production `i4_dot` (AVX2-dispatched), then time all three. Reports compile time, per-dot ns, and the speed ratio.
/// Run with `cargo run --release --features jit -- --jit-bench [k] [g] [iters]`.
pub fn bench_i4_dot(k: usize, g: usize, iters: usize) {
    // Deterministic pseudo-random inputs (LCG; no rand dep). prow nibbles span the full signed 4-bit range.
    let nbytes = k.div_ceil(2);
    let ng = k.div_ceil(g);
    let mut s: u64 = 0x9E3779B97F4A7C15;
    let mut next = || {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (s >> 33) as u32
    };
    let prow: Vec<u8> = (0..nbytes).map(|_| next() as u8).collect();
    let a: Vec<f32> = (0..k).map(|_| (next() as f32 / u32::MAX as f32) * 2.0 - 1.0).collect();
    let scale: Vec<f32> = (0..ng).map(|_| 0.005 + (next() as f32 / u32::MAX as f32) * 0.02).collect();

    let (jit, compile) = jit_i4_dot(k, g);
    let looped = if k % g == 0 && g % 2 == 0 { Some(jit_i4_dot_looped(k, g)) } else { None };
    let vec = if k % g == 0 && g % 32 == 0 { Some(jit_i4_dot_vec(k, g)) } else { None };

    // Correctness: JIT vs scalar reference vs production i4_dot (AVX2 where available).
    let r_jit = jit(prow.as_ptr(), a.as_ptr(), scale.as_ptr());
    let r_scalar = i4_dot_scalar_ref(&prow, &a, &scale, g, k);
    let r_prod = crate::bundle::i4_dot(&prow, &a, &scale, g, k);
    let d_js = (r_jit - r_scalar).abs();
    let d_jp = (r_jit - r_prod).abs();
    let tol = 1e-2 * r_scalar.abs().max(1.0);
    println!("\n=== JIT int4 dot bench  (k={k}, g={g}, ng={ng}, iters={iters}) ===");
    println!("compile unrolled (codegen+finalize): {:.2} ms   [amortised over n_rows of the matrix]", compile.as_secs_f64() * 1e3);
    if let Some((lf, lc)) = looped.as_ref() {
        let rl = lf(prow.as_ptr(), a.as_ptr(), scale.as_ptr());
        println!("compile looped   (codegen+finalize): {:.2} ms   (loop over groups, g-wide inner unroll)", lc.as_secs_f64() * 1e3);
        println!("result   jit={r_jit:.5}  looped={rl:.5}  scalar={r_scalar:.5}  prod(i4_dot)={r_prod:.5}");
    } else {
        println!("result   jit={r_jit:.5}  scalar={r_scalar:.5}  prod(i4_dot)={r_prod:.5}");
    }
    if let Some((vfn, vc)) = vec.as_ref() {
        let rv = vfn(prow.as_ptr(), a.as_ptr(), scale.as_ptr());
        let dv = (rv - r_scalar).abs();
        println!("compile vector   (codegen+finalize): {:.2} ms   (128-bit Cranelift vector IR)", vc.as_secs_f64() * 1e3);
        println!("result   vector={rv:.5}  |vec-scalar|={dv:.2e}  {}", if dv < tol { "OK" } else { "*** MISMATCH ***" });
    }
    println!("|jit-scalar|={d_js:.2e}  |jit-prod|={d_jp:.2e}  tol={tol:.2e}  {}",
        if d_js < tol && d_jp < tol { "OK" } else { "*** MISMATCH ***" });

    // Timing. The owned buffers (prow/a/scale) live to the end of this fn; every closure borrows them via raw
    // pointers (Copy), so nothing is moved/freed under the timed loops. Accumulate into `acc` (printed) so the
    // optimiser can't drop the calls.
    let warm = 1000.min(iters);
    let mut acc = 0.0f32;
    let (pp, ap, sp) = (prow.as_ptr(), a.as_ptr(), scale.as_ptr());
    let mut bench = |label: &str, mut f: Box<dyn FnMut() -> f32>| {
        for _ in 0..warm { acc += f(); }
        let t = std::time::Instant::now();
        for _ in 0..iters { acc += f(); }
        let ns = t.elapsed().as_nanos() as f64 / iters as f64;
        println!("  {label:<22} {ns:8.1} ns/dot");
        ns
    };
    let ns_scalar = bench("scalar (ref fold)", Box::new(move || {
        let pr = unsafe { std::slice::from_raw_parts(pp, nbytes) };
        let av = unsafe { std::slice::from_raw_parts(ap, k) };
        let sv = unsafe { std::slice::from_raw_parts(sp, ng) };
        i4_dot_scalar_ref(pr, av, sv, g, k)
    }));
    let ns_prod = bench("prod i4_dot (AVX2)", Box::new(move || {
        let pr = unsafe { std::slice::from_raw_parts(pp, nbytes) };
        let av = unsafe { std::slice::from_raw_parts(ap, k) };
        let sv = unsafe { std::slice::from_raw_parts(sp, ng) };
        crate::bundle::i4_dot(pr, av, sv, g, k)
    }));
    let ns_jit = bench("jit unrolled (k,g)", Box::new(move || jit(pp, ap, sp)));
    let ns_loop = looped.as_ref().map(|(lf, _)| {
        let lf = *lf;
        bench("jit looped (g inner)", Box::new(move || lf(pp, ap, sp)))
    });
    let ns_vec = vec.as_ref().map(|(vfn, _)| {
        let vfn = *vfn;
        bench("jit vector (128-bit)", Box::new(move || vfn(pp, ap, sp)))
    });

    println!("speed   jit-unrolled {:.2}× vs scalar,  {:.2}× vs prod-AVX2", ns_scalar / ns_jit, ns_prod / ns_jit);
    if let Some(nl) = ns_loop {
        println!("        jit-looped   {:.2}× vs scalar,  {:.2}× vs prod-AVX2", ns_scalar / nl, ns_prod / nl);
    }
    if let Some(nv) = ns_vec {
        println!("        jit-vector   {:.2}× vs scalar,  {:.2}× vs prod-AVX2", ns_scalar / nv, ns_prod / nv);
    }
    println!("(acc={acc:.3})  [printed to defeat dead-code elimination]");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn jit_pipeline_smoke() {
        let f = jit_add_const(5);
        assert_eq!(f(10), 15);
        assert_eq!(f(-3), 2);
    }

    #[test]
    fn jit_i4_dot_matches_scalar() {
        // odd k and a partial last group exercise the lo/hi nibble + group-boundary baking.
        let (k, g) = (37usize, 16usize);
        let nbytes = k.div_ceil(2);
        let ng = k.div_ceil(g);
        let mut s: u64 = 12345;
        let mut next = || { s = s.wrapping_mul(6364136223846793005).wrapping_add(1); (s >> 33) as u32 };
        let prow: Vec<u8> = (0..nbytes).map(|_| next() as u8).collect();
        let a: Vec<f32> = (0..k).map(|_| (next() as f32 / u32::MAX as f32) * 2.0 - 1.0).collect();
        let scale: Vec<f32> = (0..ng).map(|_| 0.01 + (next() as f32 / u32::MAX as f32) * 0.03).collect();
        let (f, _) = jit_i4_dot(k, g);
        let got = f(prow.as_ptr(), a.as_ptr(), scale.as_ptr());
        let want = i4_dot_scalar_ref(&prow, &a, &scale, g, k);
        assert!((got - want).abs() < 1e-3, "jit={got} scalar={want}");
    }

    #[test]
    fn jit_i4_dot_vec_matches_scalar() {
        // g=32 (one vector block, two chunks) and g=64 (two blocks) over multiple groups.
        for (k, g) in [(64usize, 32usize), (96, 32), (128, 64)] {
            let nbytes = k.div_ceil(2);
            let ng = k.div_ceil(g);
            let mut s: u64 = 99;
            let mut next = || { s = s.wrapping_mul(6364136223846793005).wrapping_add(1); (s >> 33) as u32 };
            let prow: Vec<u8> = (0..nbytes).map(|_| next() as u8).collect();
            let a: Vec<f32> = (0..k).map(|_| (next() as f32 / u32::MAX as f32) * 2.0 - 1.0).collect();
            let scale: Vec<f32> = (0..ng).map(|_| 0.01 + (next() as f32 / u32::MAX as f32) * 0.03).collect();
            let (f, _) = jit_i4_dot_vec(k, g);
            let got = f(prow.as_ptr(), a.as_ptr(), scale.as_ptr());
            let want = i4_dot_scalar_ref(&prow, &a, &scale, g, k);
            assert!((got - want).abs() < 1e-3, "vec jit k={k} g={g}: got={got} scalar={want}");
        }
    }
}
