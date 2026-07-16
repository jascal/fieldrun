//! GPU-resident RoPE-family forward (Llama-3.2 / Qwen2.5) via wgpu, opt-in. The RoPE counterpart of `gpu_gpt2.rs`:
//! RMSNorm + rotary position embedding + grouped-query attention + SwiGLU, all as WGSL compute shaders over weights
//! uploaded once, the residual kept on-device, only the logits read back. Validated top-1 vs the CPU RoPE kernel
//! (`--gpu-check`).
//!
//! Quantized weights: matmul weights kept **int8 in VRAM** and dequantised in the kernel (`matmul_i8`, per-output-col
//! scale) — so an int8 bundle (e.g. Qwen2.5-3B ≈ 3 GB) fits an 8 GB card where f32 (4×) would not. Norms/biases are
//! f16→f32; the rowi8 embed/unembed is dequantised to f32 once at upload. Attention accumulator holds head_dim ≤ 128.

use std::collections::HashMap;

use pollster::block_on;
use wgpu::util::DeviceExt;

use crate::bundle::Bundle;
use crate::model::Model;

// Shared-memory tiled GEMM (16×16): each A/B element is read from global memory once per tile and reused by all 16
// threads in its row/col (~16× less traffic than the naive one-thread-per-output kernel). a:(m,k), w:(k,n), c:(m,n).
const MATMUL: &str = r#"
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> w: array<f32>;
@group(0) @binding(2) var<storage, read_write> c: array<f32>;
@group(0) @binding(3) var<uniform> dims: vec4<u32>;
const TILE = 16u;
var<workgroup> As: array<f32, 256>;
var<workgroup> Bs: array<f32, 256>;
@compute @workgroup_size(16,16,1)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let m=dims.x; let k=dims.y; let n=dims.z; let lx=lid.x; let ly=lid.y;
  let row=wid.x*TILE+lx; let col=wid.y*TILE+ly;
  var acc=0.0; let nt=(k+TILE-1u)/TILE;
  for (var t=0u;t<nt;t++){
    let kx=t*TILE+ly; var av=0.0; if (row<m && kx<k){ av=a[row*k+kx]; } As[lx*TILE+ly]=av;
    let kr=t*TILE+lx; var bv=0.0; if (kr<k && col<n){ bv=w[kr*n+col]; } Bs[lx*TILE+ly]=bv;
    workgroupBarrier();
    for (var kk=0u;kk<TILE;kk++){ acc=acc+As[lx*TILE+kk]*Bs[kk*TILE+ly]; }
    workgroupBarrier();
  }
  if (row<m && col<n){ c[row*n+col]=acc; }
}"#;

// int8 tiled GEMM: weight kept int8 (4 codes/u32, output-column-major `wt[col*k+i]`), dequantised into the shared
// tile, then the per-output-column scale applied once at the end. c[row,col] = scale[col] * Σ_i a[row,i]*int8(wt[..]).
const MATMUL_I8: &str = r#"
@group(0) @binding(0) var<storage, read> a: array<f32>;        // (m, k) f32 activations
@group(0) @binding(1) var<storage, read> wq: array<u32>;       // i8 codes, 4 per u32, logical index col*k+i
@group(0) @binding(2) var<storage, read> scale: array<f32>;    // (n) per-output-column scale
@group(0) @binding(3) var<storage, read_write> c: array<f32>;  // (m, n)
@group(0) @binding(4) var<uniform> dims: vec4<u32>;            // m,k,n,_
const TILE = 16u;
var<workgroup> As: array<f32, 256>;
var<workgroup> Bs: array<f32, 256>;
@compute @workgroup_size(16,16,1)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let m=dims.x; let k=dims.y; let n=dims.z; let lx=lid.x; let ly=lid.y;
  let row=wid.x*TILE+lx; let col=wid.y*TILE+ly;
  var acc=0.0; let nt=(k+TILE-1u)/TILE;
  for (var t=0u;t<nt;t++){
    let kx=t*TILE+ly; var av=0.0; if (row<m && kx<k){ av=a[row*k+kx]; } As[lx*TILE+ly]=av;
    let kr=t*TILE+lx; var bv=0.0;
    if (kr<k && col<n){ let e=col*k+kr; let word=wq[e>>2u]; let bb=(word>>((e&3u)*8u))&0xFFu; bv=f32(i32(bb)-select(0,256,bb>127u)); }
    Bs[lx*TILE+ly]=bv;
    workgroupBarrier();
    for (var kk=0u;kk<TILE;kk++){ acc=acc+As[lx*TILE+kk]*Bs[kk*TILE+ly]; }
    workgroupBarrier();
  }
  if (row<m && col<n){ c[row*n+col]=acc*scale[col]; }
}"#;

const ADDBIAS: &str = r#"
@group(0) @binding(0) var<storage, read_write> c: array<f32>;
@group(0) @binding(1) var<storage, read> bias: array<f32>;
@group(0) @binding(2) var<uniform> dims: vec4<u32>; // m,n,_,_
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) { let i=gid.x; let n=dims.y; if (i>=dims.x*n) { return; } c[i]=c[i]+bias[i%n]; }"#;

const ADD: &str = r#"
@group(0) @binding(0) var<storage, read_write> dst: array<f32>;
@group(0) @binding(1) var<storage, read> src: array<f32>;
@group(0) @binding(2) var<uniform> dims: vec4<u32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) { let i=gid.x; if (i>=dims.x) { return; } dst[i]=dst[i]+src[i]; }"#;

const RMSNORM: &str = r#"
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> w: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform> dims: vec4<u32>; // m,d,eps_bits,_
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let row=gid.x; let d=dims.y; if (row>=dims.x) { return; }
  let eps=bitcast<f32>(dims.z);
  var ms=0.0; for (var c=0u;c<d;c++){ let v=x[row*d+c]; ms=ms+v*v; } ms=ms/f32(d);
  let inv=1.0/sqrt(ms+eps);
  for (var c=0u;c<d;c++){ out[row*d+c]=x[row*d+c]*inv*w[c]; }
}"#;

const ROPE: &str = r#"
@group(0) @binding(0) var<storage, read_write> buf: array<f32>; // (seq, nh*hd)
@group(0) @binding(1) var<storage, read> inv: array<f32>;       // hd/2 rotary freqs
@group(0) @binding(2) var<uniform> dims: vec4<u32>; // seq,nh,hd,_
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx=gid.x; let seq=dims.x; let nh=dims.y; let hd=dims.z; if (idx>=seq*nh) { return; }
  let pos=idx/nh; let head=idx%nh; let half=hd/2u; let o=pos*nh*hd + head*hd;
  for (var j=0u;j<half;j++){
    let ang=f32(pos)*inv[j]; let cs=cos(ang); let sn=sin(ang);
    let a=buf[o+j]; let b=buf[o+j+half];
    buf[o+j]=a*cs-b*sn; buf[o+j+half]=b*cs+a*sn;
  }
}"#;

// attention accumulator sized for head_dim ≤ 128 (Qwen2.5 0.5B=64, 1.5B/3B/7B=128).
const GQA: &str = r#"
@group(0) @binding(0) var<storage, read> q: array<f32>;   // (seq, H*hd)
@group(0) @binding(1) var<storage, read> k: array<f32>;   // (seq, nkv*hd)
@group(0) @binding(2) var<storage, read> v: array<f32>;   // (seq, nkv*hd)
@group(0) @binding(3) var<storage, read_write> out: array<f32>; // (seq, H*hd)
@group(0) @binding(4) var<uniform> dims: vec4<u32>; // seq,hd,H,nkv
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx=gid.x; let seq=dims.x; let hd=dims.y; let nh=dims.z; let nkv=dims.w;
  if (idx>=seq*nh) { return; }
  let i=idx/nh; let h=idx%nh; let rep=nh/nkv; let kv=h/rep;
  let qo=i*nh*hd + h*hd; let scale=1.0/sqrt(f32(hd));
  var mx=-1e30;
  for (var j=0u;j<=i;j++){ let ko=j*nkv*hd+kv*hd; var s=0.0; for (var c=0u;c<hd;c++){ s=s+q[qo+c]*k[ko+c]; } mx=max(mx,s*scale); }
  var acc: array<f32,128>; for (var c=0u;c<hd;c++){ acc[c]=0.0; } var den=0.0;
  for (var j=0u;j<=i;j++){ let ko=j*nkv*hd+kv*hd; var s=0.0; for (var c=0u;c<hd;c++){ s=s+q[qo+c]*k[ko+c]; }
    let w=exp(s*scale-mx); den=den+w; for (var c=0u;c<hd;c++){ acc[c]=acc[c]+w*v[ko+c]; } }
  for (var c=0u;c<hd;c++){ out[i*nh*hd + h*hd + c]=acc[c]/den; }
}"#;

const SWIGLU: &str = r#"
@group(0) @binding(0) var<storage, read> gate: array<f32>;
@group(0) @binding(1) var<storage, read> up: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform> dims: vec4<u32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i=gid.x; if (i>=dims.x) { return; } let g=gate[i]; out[i]=(g/(1.0+exp(-g)))*up[i];
}"#;

const ROWDOT: &str = r#"
@group(0) @binding(0) var<storage, read> xf: array<f32>;
@group(0) @binding(1) var<storage, read> wte: array<f32>;
@group(0) @binding(2) var<storage, read_write> logits: array<f32>;
@group(0) @binding(3) var<uniform> dims: vec4<u32>; // vocab,d,last_off,_
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let vrow=gid.x; let d=dims.y; let off=dims.z; if (vrow>=dims.x) { return; }
  var s=0.0; for (var c=0u;c<d;c++){ s=s+xf[off+c]*wte[vrow*d+c]; } logits[vrow]=s;
}"#;

/// An uploaded weight: either f32 (norms, biases, dequantised embed) or int8 kept in VRAM with its per-column scale.
enum Wt {
    F32(wgpu::Buffer),
    I8 { q: wgpu::Buffer, scale: wgpu::Buffer },
}

pub struct GpuRope {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pl: HashMap<&'static str, wgpu::ComputePipeline>,
    w: HashMap<String, Wt>,
    embed: wgpu::Buffer,
    inv: wgpu::Buffer,
    n_layer: usize,
    h: usize,
    nkv: usize,
    hd: usize,
    d: usize,
    ffn: usize,
    vocab: usize,
    eps: f32,
    pub name: String,
}

impl GpuRope {
    pub fn new(b: &Bundle) -> Option<GpuRope> {
        let instance = wgpu::Instance::default();
        let adapter = block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance, ..Default::default()
        }))?;
        let name = format!("{}, {:?}", adapter.get_info().name, adapter.get_info().backend);
        let mut lim = wgpu::Limits::default();
        lim.max_buffer_size = 1 << 31;
        lim.max_storage_buffer_binding_size = 1 << 31;
        let (device, queue) = block_on(adapter.request_device(&wgpu::DeviceDescriptor { required_limits: lim, ..Default::default() }, None)).ok()?;
        let mut pl = HashMap::new();
        for (k, src) in [("matmul", MATMUL), ("matmul_i8", MATMUL_I8), ("addbias", ADDBIAS), ("add", ADD), ("rmsnorm", RMSNORM),
                         ("rope", ROPE), ("gqa", GQA), ("swiglu", SWIGLU), ("rowdot", ROWDOT)] {
            let m = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some(k), source: wgpu::ShaderSource::Wgsl(src.into()) });
            pl.insert(k, device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(k), layout: None, module: &m, entry_point: "main", compilation_options: Default::default(), cache: None,
            }));
        }
        let c = &b.config; // [n_layer, H, nkv, hd, d, ffn, vocab, tied]
        let (n_layer, h, nkv, hd, d, ffn, vocab) =
            (c[0] as usize, c[1] as usize, c[2] as usize, c[3] as usize, c[4] as usize, c[5] as usize, c[6] as usize);
        assert!(hd <= 128, "gpu_rope: head_dim {hd} > 128 (GQA accumulator cap)");
        let (theta, eps) = (b.config_f[0] as f32, b.config_f[1] as f32);
        let invf: Vec<f32> = (0..hd / 2).map(|j| 1.0 / theta.powf(2.0 * j as f32 / hd as f32)).collect();
        let store = |data: &[f32]| device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None, contents: cast(data),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        });
        let store_i8 = |q: &[i8]| {
            let mut bytes: Vec<u8> = q.iter().map(|&x| x as u8).collect();
            while bytes.len() % 4 != 0 { bytes.push(0); } // pad to whole u32 words
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None, contents: &bytes,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            })
        };
        // embed/unembed: dequantise the rowi8 (or f16/f32) table to f32 once for the unembed rowdot.
        let embed_ids: Vec<i64> = (0..vocab as i64).collect();
        let embed = store(b.rows_f32("embed", &embed_ids).as_slice().unwrap());
        let inv = store(&invf);
        let mut w = HashMap::new();
        let mut names = vec!["norm".to_string()];
        for l in 0..n_layer {
            for s in ["in_ln", "post_ln", "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
                      "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj", "self_attn.q_proj.bias", "self_attn.k_proj.bias", "self_attn.v_proj.bias"] {
                let nm = format!("l{l}.{s}");
                if b.has(&nm) { names.push(nm); }
            }
        }
        for nm in names {
            let wt = if let Some((q, scale, _n, _k)) = b.i8_for_gpu(&nm) {
                Wt::I8 { q: store_i8(&q), scale: store(&scale) }
            } else {
                Wt::F32(store(&b.f32_array(&nm).1)) // f16/f32 (norms, biases) → f32
            };
            w.insert(nm, wt);
        }
        Some(GpuRope { device, queue, pl, w, embed, inv, n_layer, h, nkv, hd, d, ffn, vocab, eps, name })
    }

    fn storage(&self, len: usize) -> wgpu::Buffer {
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None, size: (len * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false,
        })
    }

    fn record(&self, enc: &mut wgpu::CommandEncoder, ku: &mut Vec<wgpu::Buffer>, kb: &mut Vec<wgpu::BindGroup>,
              name: &str, bufs: &[&wgpu::Buffer], dims: [u32; 4], groups: (u32, u32)) {
        let pipe = &self.pl[name];
        let u = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: None, contents: cast_u32(&dims), usage: wgpu::BufferUsages::UNIFORM });
        let mut entries: Vec<wgpu::BindGroupEntry> = bufs.iter().enumerate()
            .map(|(i, b)| wgpu::BindGroupEntry { binding: i as u32, resource: b.as_entire_binding() }).collect();
        entries.push(wgpu::BindGroupEntry { binding: bufs.len() as u32, resource: u.as_entire_binding() });
        let bind = self.device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &pipe.get_bind_group_layout(0), entries: &entries });
        drop(entries);
        ku.push(u);
        kb.push(bind);
        let mut pass = enc.begin_compute_pass(&Default::default());
        pass.set_pipeline(pipe);
        pass.set_bind_group(0, kb.last().unwrap(), &[]);
        pass.dispatch_workgroups(groups.0, groups.1, 1);
    }

    fn read(&self, buf: &wgpu::Buffer, len: usize) -> Vec<f32> {
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor { label: None, size: (len * 4) as u64, usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
        let mut enc = self.device.create_command_encoder(&Default::default());
        enc.copy_buffer_to_buffer(buf, 0, &staging, 0, (len * 4) as u64);
        self.queue.submit(Some(enc.finish()));
        let slice = staging.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        self.device.poll(wgpu::Maintain::Wait);
        let out: Vec<f32> = slice.get_mapped_range().chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
        staging.unmap();
        out
    }

    /// Full-vocab next-token logits at the last position. Same forward as `predict`; only the argmax is dropped.
    pub fn logits(&self, ids: &[i64], b: &Bundle) -> Vec<f32> {
        let (seq, d, hd, h, nkv, ffn) = (ids.len(), self.d, self.hd, self.h, self.nkv, self.ffn);
        let qd = h * hd;
        let kvd = nkv * hd;
        let x0 = b.rows_f32("embed", ids); // (seq, d), plain RoPE has no embed scale
        let x = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None, contents: cast(x0.as_slice().unwrap()),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        });
        let (a, q, k, v) = (self.storage(seq * d), self.storage(seq * qd), self.storage(seq * kvd), self.storage(seq * kvd));
        let (attn, proj, a2) = (self.storage(seq * qd), self.storage(seq * d), self.storage(seq * d));
        let (gate, up, hbuf, down, xf) = (self.storage(seq * ffn), self.storage(seq * ffn), self.storage(seq * ffn), self.storage(seq * d), self.storage(seq * d));
        let logits = self.storage(self.vocab);
        let mut enc = self.device.create_command_encoder(&Default::default());
        let (mut ku, mut kb) = (Vec::new(), Vec::new());
        let mm = |m: usize, n: usize| (m.div_ceil(16) as u32, n.div_ceil(16) as u32); // 16×16 tiled matmul workgroups
        let g1 = |n: usize| (n.div_ceil(64) as u32, 1u32);
        let epsb = self.eps.to_bits();
        let (su, du) = (seq as u32, d as u32);
        for l in 0..self.n_layer {
            let p = format!("l{l}.");
            let wf = |s: &str| match &self.w[&format!("{p}{s}")] { Wt::F32(b) => b, _ => panic!("gpu_rope: {s} expected f32") };
            let bias = |s: &str| self.w.contains_key(&format!("{p}{s}.bias"));
            let mut r = |name: &str, bufs: &[&wgpu::Buffer], dims: [u32; 4], groups: (u32, u32)|
                self.record(&mut enc, &mut ku, &mut kb, name, bufs, dims, groups);
            // matmul against a weight that may be f32 or int8 (dequant-in-kernel).
            let mw = |r: &mut dyn FnMut(&str, &[&wgpu::Buffer], [u32; 4], (u32, u32)),
                          s: &str, a: &wgpu::Buffer, out: &wgpu::Buffer, dims: [u32; 4], groups: (u32, u32)| {
                match &self.w[&format!("{p}{s}")] {
                    Wt::F32(wb) => r("matmul", &[a, wb, out], dims, groups),
                    Wt::I8 { q, scale } => r("matmul_i8", &[a, q, scale, out], dims, groups),
                }
            };
            r("rmsnorm", &[&x, wf("in_ln"), &a], [su, du, epsb, 0], g1(seq));
            mw(&mut r, "self_attn.q_proj", &a, &q, [su, du, qd as u32, 0], mm(seq, qd));
            mw(&mut r, "self_attn.k_proj", &a, &k, [su, du, kvd as u32, 0], mm(seq, kvd));
            mw(&mut r, "self_attn.v_proj", &a, &v, [su, du, kvd as u32, 0], mm(seq, kvd));
            if bias("self_attn.q_proj") { r("addbias", &[&q, wf("self_attn.q_proj.bias")], [su, qd as u32, 0, 0], g1(seq * qd)); }
            if bias("self_attn.k_proj") { r("addbias", &[&k, wf("self_attn.k_proj.bias")], [su, kvd as u32, 0, 0], g1(seq * kvd)); }
            if bias("self_attn.v_proj") { r("addbias", &[&v, wf("self_attn.v_proj.bias")], [su, kvd as u32, 0, 0], g1(seq * kvd)); }
            r("rope", &[&q, &self.inv], [su, h as u32, hd as u32, 0], ((seq * h).div_ceil(64) as u32, 1));
            r("rope", &[&k, &self.inv], [su, nkv as u32, hd as u32, 0], ((seq * nkv).div_ceil(64) as u32, 1));
            r("gqa", &[&q, &k, &v, &attn], [su, hd as u32, h as u32, nkv as u32], ((seq * h).div_ceil(64) as u32, 1));
            mw(&mut r, "self_attn.o_proj", &attn, &proj, [su, qd as u32, du, 0], mm(seq, d));
            r("add", &[&x, &proj], [(seq * d) as u32, 0, 0, 0], g1(seq * d));
            r("rmsnorm", &[&x, wf("post_ln"), &a2], [su, du, epsb, 0], g1(seq));
            mw(&mut r, "mlp.gate_proj", &a2, &gate, [su, du, ffn as u32, 0], mm(seq, ffn));
            mw(&mut r, "mlp.up_proj", &a2, &up, [su, du, ffn as u32, 0], mm(seq, ffn));
            r("swiglu", &[&gate, &up, &hbuf], [(seq * ffn) as u32, 0, 0, 0], g1(seq * ffn));
            mw(&mut r, "mlp.down_proj", &hbuf, &down, [su, ffn as u32, du, 0], mm(seq, d));
            r("add", &[&x, &down], [(seq * d) as u32, 0, 0, 0], g1(seq * d));
        }
        let normw = match &self.w["norm"] { Wt::F32(b) => b, _ => panic!("gpu_rope: norm expected f32") };
        self.record(&mut enc, &mut ku, &mut kb, "rmsnorm", &[&x, normw, &xf], [su, du, epsb, 0], g1(seq));
        self.record(&mut enc, &mut ku, &mut kb, "rowdot", &[&xf, &self.embed, &logits],
                    [self.vocab as u32, du, ((seq - 1) * d) as u32, 0], (self.vocab.div_ceil(64) as u32, 1));
        self.queue.submit(Some(enc.finish()));
        self.read(&logits, self.vocab)
    }

    pub fn predict(&self, ids: &[i64], b: &Bundle) -> i64 {
        let lg = self.logits(ids, b);
        lg.iter().enumerate().max_by(|x, y| x.1.partial_cmp(y.1).unwrap()).unwrap().0 as i64
    }

    /// Per-block residual writes at the last position, final-RMSNorm-folded — same semantics as
    /// `Rope::residual_normed_writes` (block order `["embed","L0.attn","L0.mlp",…]`, `d̃_b = inv_rms · gain ⊙ write_b`).
    /// One GPU submit: last-row o_proj / down_proj writes + pre-final-norm residual are copied into a collector in the
    /// same encoder as the layer loop (no per-layer sync). Block 0 ("embed") is already host-side from the embed gather.
    pub fn residual_normed_writes(&self, ids: &[i64], b: &Bundle) -> Option<(Vec<String>, Vec<Vec<f32>>)> {
        if ids.is_empty() { return None; }
        let (seq, d, hd, h, nkv, ffn) = (ids.len(), self.d, self.hd, self.h, self.nkv, self.ffn);
        let qd = h * hd;
        let kvd = nkv * hd;
        let x0 = b.rows_f32("embed", ids); // (seq, d) — also the raw embed write (block 0), host-side
        let x = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None, contents: cast(x0.as_slice().unwrap()),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        });
        let (a, q, k, v) = (self.storage(seq * d), self.storage(seq * qd), self.storage(seq * kvd), self.storage(seq * kvd));
        let (attn, proj, a2) = (self.storage(seq * qd), self.storage(seq * d), self.storage(seq * d));
        let (gate, up, hbuf, down) = (self.storage(seq * ffn), self.storage(seq * ffn), self.storage(seq * ffn), self.storage(seq * d));
        // collector: 2*n_layer last-row writes (attn, mlp per layer) + 1 slot for pre-final-norm xpre
        let n_slots = 2 * self.n_layer + 1;
        let collector = self.storage(n_slots * d);
        let last_off = ((seq - 1) * d * 4) as u64;
        let d_bytes = (d * 4) as u64;
        let mut enc = self.device.create_command_encoder(&Default::default());
        let (mut ku, mut kb) = (Vec::new(), Vec::new());
        let mm = |m: usize, n: usize| (m.div_ceil(16) as u32, n.div_ceil(16) as u32);
        let g1 = |n: usize| (n.div_ceil(64) as u32, 1u32);
        let epsb = self.eps.to_bits();
        let (su, du) = (seq as u32, d as u32);
        let mut slot = 0usize;
        for l in 0..self.n_layer {
            // Scope the compute-pass closure so it drops before the collector copies need `&mut enc`.
            // After the layer finishes, `proj`/`down` still hold this layer's o_proj/down_proj last-row writes.
            {
                let p = format!("l{l}.");
                let wf = |s: &str| match &self.w[&format!("{p}{s}")] { Wt::F32(b) => b, _ => panic!("gpu_rope: {s} expected f32") };
                let bias = |s: &str| self.w.contains_key(&format!("{p}{s}.bias"));
                let mut r = |name: &str, bufs: &[&wgpu::Buffer], dims: [u32; 4], groups: (u32, u32)|
                    self.record(&mut enc, &mut ku, &mut kb, name, bufs, dims, groups);
                let mw = |r: &mut dyn FnMut(&str, &[&wgpu::Buffer], [u32; 4], (u32, u32)),
                              s: &str, a: &wgpu::Buffer, out: &wgpu::Buffer, dims: [u32; 4], groups: (u32, u32)| {
                    match &self.w[&format!("{p}{s}")] {
                        Wt::F32(wb) => r("matmul", &[a, wb, out], dims, groups),
                        Wt::I8 { q, scale } => r("matmul_i8", &[a, q, scale, out], dims, groups),
                    }
                };
                r("rmsnorm", &[&x, wf("in_ln"), &a], [su, du, epsb, 0], g1(seq));
                mw(&mut r, "self_attn.q_proj", &a, &q, [su, du, qd as u32, 0], mm(seq, qd));
                mw(&mut r, "self_attn.k_proj", &a, &k, [su, du, kvd as u32, 0], mm(seq, kvd));
                mw(&mut r, "self_attn.v_proj", &a, &v, [su, du, kvd as u32, 0], mm(seq, kvd));
                if bias("self_attn.q_proj") { r("addbias", &[&q, wf("self_attn.q_proj.bias")], [su, qd as u32, 0, 0], g1(seq * qd)); }
                if bias("self_attn.k_proj") { r("addbias", &[&k, wf("self_attn.k_proj.bias")], [su, kvd as u32, 0, 0], g1(seq * kvd)); }
                if bias("self_attn.v_proj") { r("addbias", &[&v, wf("self_attn.v_proj.bias")], [su, kvd as u32, 0, 0], g1(seq * kvd)); }
                r("rope", &[&q, &self.inv], [su, h as u32, hd as u32, 0], ((seq * h).div_ceil(64) as u32, 1));
                r("rope", &[&k, &self.inv], [su, nkv as u32, hd as u32, 0], ((seq * nkv).div_ceil(64) as u32, 1));
                r("gqa", &[&q, &k, &v, &attn], [su, hd as u32, h as u32, nkv as u32], ((seq * h).div_ceil(64) as u32, 1));
                mw(&mut r, "self_attn.o_proj", &attn, &proj, [su, qd as u32, du, 0], mm(seq, d));
                r("add", &[&x, &proj], [(seq * d) as u32, 0, 0, 0], g1(seq * d));
                r("rmsnorm", &[&x, wf("post_ln"), &a2], [su, du, epsb, 0], g1(seq));
                mw(&mut r, "mlp.gate_proj", &a2, &gate, [su, du, ffn as u32, 0], mm(seq, ffn));
                mw(&mut r, "mlp.up_proj", &a2, &up, [su, du, ffn as u32, 0], mm(seq, ffn));
                r("swiglu", &[&gate, &up, &hbuf], [(seq * ffn) as u32, 0, 0, 0], g1(seq * ffn));
                mw(&mut r, "mlp.down_proj", &hbuf, &down, [su, ffn as u32, du, 0], mm(seq, d));
                r("add", &[&x, &down], [(seq * d) as u32, 0, 0, 0], g1(seq * d));
            }
            // same encoder as the layer loop — no extra submit/sync; just more commands
            enc.copy_buffer_to_buffer(&proj, last_off, &collector, (slot * d * 4) as u64, d_bytes);
            slot += 1;
            enc.copy_buffer_to_buffer(&down, last_off, &collector, (slot * d * 4) as u64, d_bytes);
            slot += 1;
        }
        // exact pre-final-norm residual the GPU accumulated (do not reconstruct by summing host-side writes)
        enc.copy_buffer_to_buffer(&x, last_off, &collector, (slot * d * 4) as u64, d_bytes);
        self.queue.submit(Some(enc.finish()));
        let raw = self.read(&collector, n_slots * d);

        let mut labels: Vec<String> = vec!["embed".into()];
        let mut writes: Vec<Vec<f32>> = vec![x0.row(seq - 1).to_vec()];
        for l in 0..self.n_layer {
            let aw = &raw[(2 * l) * d..(2 * l + 1) * d];
            writes.push(aw.to_vec());
            labels.push(format!("L{l}.attn"));
            let mw = &raw[(2 * l + 1) * d..(2 * l + 2) * d];
            writes.push(mw.to_vec());
            labels.push(format!("L{l}.mlp"));
        }
        let xpre = &raw[(2 * self.n_layer) * d..(2 * self.n_layer + 1) * d];
        let inv_rms = 1.0 / (xpre.iter().map(|v| v * v).sum::<f32>() / d as f32 + self.eps).sqrt();
        let gain = b.arr1("norm");
        let dvec: Vec<Vec<f32>> = writes
            .iter()
            .map(|w| w.iter().zip(gain.iter()).map(|(wd, gd)| inv_rms * wd * gd).collect())
            .collect();
        Some((labels, dvec))
    }
}

/// `Model` wrapper around `GpuRope` for `--source-dump` under `--device gpu`. Owns the bundle (host embed gather +
/// unembed rows) and the uploaded GPU state. Only the methods `dump_one` / `run_source_dump` call are implemented.
pub struct GpuRopeModel {
    pub gpu: GpuRope,
    b: Bundle,
}

impl GpuRopeModel {
    pub fn new(b: Bundle) -> Option<Self> {
        let gpu = GpuRope::new(&b)?;
        Some(Self { gpu, b })
    }

    fn unembed_name(&self) -> &'static str {
        if self.b.config[7] != 0 { "embed" } else { "lm_head" }
    }
}

impl Model for GpuRopeModel {
    fn predict(&self, ids: &[i64]) -> i64 {
        self.gpu.predict(ids, &self.b)
    }

    fn logits(&self, ids: &[i64]) -> Option<Vec<f32>> {
        Some(self.gpu.logits(ids, &self.b))
    }

    fn residual_normed_writes(&self, ids: &[i64]) -> Option<(Vec<String>, Vec<Vec<f32>>)> {
        self.gpu.residual_normed_writes(ids, &self.b)
    }

    fn unembed_row(&self, id: usize) -> Option<Vec<f32>> {
        let r = self.b.weight_row(self.unembed_name(), id);
        if r.is_empty() { None } else { Some(r) }
    }
}

fn cast(s: &[f32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(s.as_ptr() as *const u8, std::mem::size_of_val(s)) }
}
fn cast_u32(s: &[u32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(s.as_ptr() as *const u8, std::mem::size_of_val(s)) }
}
