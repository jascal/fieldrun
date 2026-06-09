//! GPU-resident GPT-2 forward (wgpu, opt-in). Weights are uploaded to the GPU once at construction; the residual
//! stream stays in a GPU buffer across the layer, with matmul / LayerNorm / GELU / residual-add as WGSL compute
//! shaders. Attention is computed on the CPU for v1 (small, and the masked-softmax shader is the one tricky piece —
//! the only per-layer round-trip), and only the last-position hidden is read back for the (CPU) unembed.
//!
//! Validated by top-1 agreement against the CPU forward (`fieldrun --device gpu --gpu-check`). Cross-platform via wgpu.

use std::collections::HashMap;

use pollster::block_on;
use wgpu::util::DeviceExt;

use crate::bundle::Bundle;

const MATMUL: &str = r#"
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> w: array<f32>;
@group(0) @binding(2) var<storage, read_write> c: array<f32>;
@group(0) @binding(3) var<uniform> dims: vec4<u32>; // m,k,n,_
@compute @workgroup_size(8,8,1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let m=dims.x; let k=dims.y; let n=dims.z; let row=gid.x; let col=gid.y;
  if (row>=m || col>=n) { return; }
  var acc=0.0; for (var i=0u;i<k;i++){ acc=acc+a[row*k+i]*w[i*n+col]; }
  c[row*n+col]=acc;
}"#;

const ADDBIAS: &str = r#"
@group(0) @binding(0) var<storage, read_write> c: array<f32>;
@group(0) @binding(1) var<storage, read> bias: array<f32>;
@group(0) @binding(2) var<uniform> dims: vec4<u32>; // m,n,_,_
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i=gid.x; let n=dims.y; if (i>=dims.x*n) { return; }
  c[i]=c[i]+bias[i%n];
}"#;

const ADD: &str = r#"
@group(0) @binding(0) var<storage, read_write> dst: array<f32>;
@group(0) @binding(1) var<storage, read> src: array<f32>;
@group(0) @binding(2) var<uniform> dims: vec4<u32>; // n,_,_,_
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i=gid.x; if (i>=dims.x) { return; } dst[i]=dst[i]+src[i];
}"#;

const GELU: &str = r#"
@group(0) @binding(0) var<storage, read_write> x: array<f32>;
@group(0) @binding(1) var<uniform> dims: vec4<u32>; // n,_,_,_
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i=gid.x; if (i>=dims.x) { return; }
  let v=x[i]; let c=0.7978845608; // sqrt(2/pi)
  x[i]=0.5*v*(1.0+tanh(c*(v+0.044715*v*v*v)));
}"#;

// Causal multi-head attention over a (seq, 3d) qkv buffer → (seq, d). One thread per (query i, head h); two passes
// (max, then weighted-sum) with a function-local accumulator (hd ≤ 64 for GPT-2). Keeps attention on the GPU so the
// residual never leaves the device.
const ATTENTION: &str = r#"
@group(0) @binding(0) var<storage, read> qkv: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<uniform> dims: vec4<u32>; // seq,d,nh,hd
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx=gid.x; let seq=dims.x; let d=dims.y; let nh=dims.z; let hd=dims.w;
  if (idx>=seq*nh) { return; }
  let i=idx/nh; let h=idx%nh; let qb=i*3u*d + h*hd; let scale=1.0/sqrt(f32(hd));
  var mx=-1e30;
  for (var j=0u;j<=i;j++){ let kb=j*3u*d+d+h*hd; var s=0.0; for (var c=0u;c<hd;c++){ s=s+qkv[qb+c]*qkv[kb+c]; } mx=max(mx,s*scale); }
  var acc: array<f32,64>; for (var c=0u;c<hd;c++){ acc[c]=0.0; }
  var den=0.0;
  for (var j=0u;j<=i;j++){
    let kb=j*3u*d+d+h*hd; let vb=j*3u*d+2u*d+h*hd;
    var s=0.0; for (var c=0u;c<hd;c++){ s=s+qkv[qb+c]*qkv[kb+c]; }
    let w=exp(s*scale-mx); den=den+w;
    for (var c=0u;c<hd;c++){ acc[c]=acc[c]+w*qkv[vb+c]; }
  }
  for (var c=0u;c<hd;c++){ out[i*d+h*hd+c]=acc[c]/den; }
}"#;

const LAYERNORM: &str = r#"
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> g: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
@group(0) @binding(4) var<uniform> dims: vec4<u32>; // m,d,_,_
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let row=gid.x; let d=dims.y; if (row>=dims.x) { return; }
  var mu=0.0; for (var i=0u;i<d;i++){ mu=mu+x[row*d+i]; } mu=mu/f32(d);
  var va=0.0; for (var i=0u;i<d;i++){ let t=x[row*d+i]-mu; va=va+t*t; } va=va/f32(d);
  let inv=1.0/sqrt(va+1e-5);
  for (var i=0u;i<d;i++){ out[row*d+i]=(x[row*d+i]-mu)*inv*g[i]+b[i]; }
}"#;

// Tied unembed on the GPU: logits[v] = dot(xf[last_row], wte[v]). One thread per vocab row, reading back only the
// logits (vocab,) instead of the CPU rowdot over the whole 50k-row table per forward.
const ROWDOT: &str = r#"
@group(0) @binding(0) var<storage, read> xf: array<f32>;
@group(0) @binding(1) var<storage, read> wte: array<f32>;
@group(0) @binding(2) var<storage, read_write> logits: array<f32>;
@group(0) @binding(3) var<uniform> dims: vec4<u32>; // vocab,d,last_off,_
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let v=gid.x; let d=dims.y; let off=dims.z; if (v>=dims.x) { return; }
  var s=0.0; for (var c=0u;c<d;c++){ s=s+xf[off+c]*wte[v*d+c]; }
  logits[v]=s;
}"#;

struct W {
    buf: wgpu::Buffer,
    rows: usize,
    cols: usize,
}

pub struct GpuGpt2 {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pl: HashMap<&'static str, wgpu::ComputePipeline>,
    w: HashMap<String, W>, // resident weights (2D: rows,cols) and 1D biases/norms (rows=1)
    wte: wgpu::Buffer,     // resident token embedding (also the tied unembed), uploaded once
    n_layer: usize,
    n_head: usize,
    d: usize,
    vocab: usize,
    pub name: String,
}

impl GpuGpt2 {
    pub fn new(b: &Bundle) -> Option<GpuGpt2> {
        let instance = wgpu::Instance::default();
        let adapter = block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..Default::default()
        }))?;
        let name = format!("{}, {:?}", adapter.get_info().name, adapter.get_info().backend);
        // bump limits so the big GPT-2 buffers (wte 50257*768) fit a single binding
        let mut lim = wgpu::Limits::default();
        lim.max_buffer_size = 1 << 30;
        lim.max_storage_buffer_binding_size = 1 << 30;
        let (device, queue) = block_on(adapter.request_device(
            &wgpu::DeviceDescriptor { required_limits: lim, ..Default::default() },
            None,
        )).ok()?;
        let mut pl = HashMap::new();
        for (k, src) in [("matmul", MATMUL), ("addbias", ADDBIAS), ("add", ADD), ("gelu", GELU), ("layernorm", LAYERNORM), ("attention", ATTENTION), ("rowdot", ROWDOT)] {
            let m = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some(k), source: wgpu::ShaderSource::Wgsl(src.into()) });
            pl.insert(k, device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(k), layout: None, module: &m, entry_point: "main", compilation_options: Default::default(), cache: None,
            }));
        }
        let (n_layer, n_head, d, vocab) = (b.config[0] as usize, b.config[1] as usize, b.config[2] as usize, b.config[4] as usize);
        let wte = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("wte"), contents: cast(&b.f32_array("wte").1),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        });
        let mut g = GpuGpt2 { device, queue, pl, w: HashMap::new(), wte, n_layer, n_head, d, vocab, name };
        // upload every weight the forward needs (as f32)
        let mut names: Vec<String> = vec!["ln_f.weight".into(), "ln_f.bias".into()];
        for l in 0..n_layer {
            for s in ["ln_1.weight", "ln_1.bias", "attn.c_attn.weight", "attn.c_attn.bias", "attn.c_proj.weight",
                      "attn.c_proj.bias", "ln_2.weight", "ln_2.bias", "mlp.c_fc.weight", "mlp.c_fc.bias",
                      "mlp.c_proj.weight", "mlp.c_proj.bias"] {
                names.push(format!("h{l}.{s}"));
            }
        }
        for nm in names {
            let (shape, data) = b.f32_array(&nm);
            let (rows, cols) = if shape.len() == 2 { (shape[0], shape[1]) } else { (1, shape[0]) };
            g.w.insert(nm, W { buf: g.storage_init(&data), rows, cols });
        }
        Some(g)
    }

    fn storage_init(&self, data: &[f32]) -> wgpu::Buffer {
        self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None, contents: cast(data),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        })
    }

    fn storage(&self, len: usize) -> wgpu::Buffer {
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None, size: (len * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    fn uniform(&self, dims: [u32; 4]) -> wgpu::Buffer {
        self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None, contents: cast_u32(&dims), usage: wgpu::BufferUsages::UNIFORM,
        })
    }

    /// Record one shader dispatch into a shared encoder (its own compute pass — wgpu inserts the barriers between
    /// passes for dependent buffers). The uniform + bind group are kept alive in `ku`/`kb` until submit.
    fn record(&self, enc: &mut wgpu::CommandEncoder, ku: &mut Vec<wgpu::Buffer>, kb: &mut Vec<wgpu::BindGroup>,
              name: &str, bufs: &[&wgpu::Buffer], dims: [u32; 4], groups: (u32, u32)) {
        let pipe = &self.pl[name];
        let u = self.uniform(dims);
        let mut entries: Vec<wgpu::BindGroupEntry> = bufs.iter().enumerate()
            .map(|(i, b)| wgpu::BindGroupEntry { binding: i as u32, resource: b.as_entire_binding() }).collect();
        entries.push(wgpu::BindGroupEntry { binding: bufs.len() as u32, resource: u.as_entire_binding() });
        let bind = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None, layout: &pipe.get_bind_group_layout(0), entries: &entries,
        });
        drop(entries);
        ku.push(u);
        kb.push(bind);
        let mut pass = enc.begin_compute_pass(&Default::default());
        pass.set_pipeline(pipe);
        pass.set_bind_group(0, kb.last().unwrap(), &[]);
        pass.dispatch_workgroups(groups.0, groups.1, 1);
    }

    fn read(&self, buf: &wgpu::Buffer, len: usize) -> Vec<f32> {
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None, size: (len * 4) as u64, usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false,
        });
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

    /// GPU forward, batched into ONE command encoder/submit with reused working buffers (no per-op alloc/submit).
    /// `x0` is the already-embedded input (seq*d). Runs all layers + the tied unembed on the GPU and reads back the
    /// logits (vocab,) — the only thing that leaves the device. Argmax happens on the CPU.
    fn last_hidden(&self, x0: &[f32], seq: usize) -> Vec<f32> {
        let (d, hd, nh) = (self.d, self.d / self.n_head, self.n_head);
        // working buffers, allocated once and reused across layers
        let x = self.storage_init(x0);
        let (a, qkv, attn, proj) = (self.storage(seq * d), self.storage(seq * 3 * d), self.storage(seq * d), self.storage(seq * d));
        let (a2, hb, mp, xf) = (self.storage(seq * d), self.storage(seq * 4 * d), self.storage(seq * d), self.storage(seq * d));
        let mut enc = self.device.create_command_encoder(&Default::default());
        let (mut ku, mut kb) = (Vec::new(), Vec::new());
        let mm = |m: usize, n: usize| (m.div_ceil(8) as u32, n.div_ceil(8) as u32);
        let g1 = |n: usize| (n.div_ceil(64) as u32, 1u32);
        let (su, du, td, fd) = (seq as u32, d as u32, (3 * d) as u32, (4 * d) as u32);
        for l in 0..self.n_layer {
            let p = format!("h{l}.");
            let wb = |s: &str| &self.w[&format!("{p}{s}")].buf;
            let mut r = |name, bufs: &[&wgpu::Buffer], dims, groups| self.record(&mut enc, &mut ku, &mut kb, name, bufs, dims, groups);
            r("layernorm", &[&x, wb("ln_1.weight"), wb("ln_1.bias"), &a], [su, du, 0, 0], g1(seq));
            r("matmul", &[&a, wb("attn.c_attn.weight"), &qkv], [su, du, td, 0], mm(seq, 3 * d));
            r("addbias", &[&qkv, wb("attn.c_attn.bias")], [su, td, 0, 0], g1(seq * 3 * d));
            r("attention", &[&qkv, &attn], [su, du, nh as u32, hd as u32], ((seq * nh).div_ceil(64) as u32, 1));
            r("matmul", &[&attn, wb("attn.c_proj.weight"), &proj], [su, du, du, 0], mm(seq, d));
            r("addbias", &[&proj, wb("attn.c_proj.bias")], [su, du, 0, 0], g1(seq * d));
            r("add", &[&x, &proj], [(seq * d) as u32, 0, 0, 0], g1(seq * d));
            r("layernorm", &[&x, wb("ln_2.weight"), wb("ln_2.bias"), &a2], [su, du, 0, 0], g1(seq));
            r("matmul", &[&a2, wb("mlp.c_fc.weight"), &hb], [su, du, fd, 0], mm(seq, 4 * d));
            r("addbias", &[&hb, wb("mlp.c_fc.bias")], [su, fd, 0, 0], g1(seq * 4 * d));
            r("gelu", &[&hb], [(seq * 4 * d) as u32, 0, 0, 0], g1(seq * 4 * d));
            r("matmul", &[&hb, wb("mlp.c_proj.weight"), &mp], [su, fd, du, 0], mm(seq, d));
            r("addbias", &[&mp, wb("mlp.c_proj.bias")], [su, du, 0, 0], g1(seq * d));
            r("add", &[&x, &mp], [(seq * d) as u32, 0, 0, 0], g1(seq * d));
        }
        self.record(&mut enc, &mut ku, &mut kb, "layernorm",
                    &[&x, &self.w["ln_f.weight"].buf, &self.w["ln_f.bias"].buf, &xf], [su, du, 0, 0], g1(seq));
        // tied unembed on the GPU (last row only) → logits, the only thing read back
        let logits = self.storage(self.vocab);
        self.record(&mut enc, &mut ku, &mut kb, "rowdot", &[&xf, &self.wte, &logits],
                    [self.vocab as u32, du, ((seq - 1) * d) as u32, 0], (self.vocab.div_ceil(64) as u32, 1));
        self.queue.submit(Some(enc.finish()));
        self.read(&logits, self.vocab)
    }

    /// Predict the next token: embed the looked-up rows → GPU forward + GPU unembed → argmax on CPU.
    pub fn predict(&self, ids: &[i64], b: &Bundle) -> i64 {
        let pos: Vec<i64> = (0..ids.len() as i64).collect();
        let x0 = &b.rows_f32("wte", ids) + &b.rows_f32("wpe", &pos); // (seq, d), no full-embedding copy
        let logits = self.last_hidden(x0.as_slice().unwrap(), ids.len());
        logits.iter().enumerate().max_by(|a, c| a.1.partial_cmp(c.1).unwrap()).unwrap().0 as i64
    }
}

fn cast(s: &[f32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(s.as_ptr() as *const u8, std::mem::size_of_val(s)) }
}
fn cast_u32(s: &[u32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(s.as_ptr() as *const u8, std::mem::size_of_val(s)) }
}
