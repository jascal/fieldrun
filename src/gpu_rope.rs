//! GPU-resident RoPE-family forward (Llama-3.2 / Qwen2.5) via wgpu, opt-in. The RoPE counterpart of `gpu_gpt2.rs`:
//! RMSNorm + rotary position embedding + grouped-query attention + SwiGLU, all as WGSL compute shaders over weights
//! uploaded once, the residual kept on-device, only the logits read back. Validated top-1 vs the CPU RoPE kernel
//! (`--gpu-check`). hd ≤ 64 (the attention accumulator) — fits Qwen2.5-0.5B; bigger head_dim bumps the array.

use std::collections::HashMap;

use pollster::block_on;
use wgpu::util::DeviceExt;

use crate::bundle::Bundle;

const MATMUL: &str = r#"
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> w: array<f32>;
@group(0) @binding(2) var<storage, read_write> c: array<f32>;
@group(0) @binding(3) var<uniform> dims: vec4<u32>;
@compute @workgroup_size(8,8,1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let m=dims.x; let k=dims.y; let n=dims.z; let row=gid.x; let col=gid.y;
  if (row>=m || col>=n) { return; }
  var acc=0.0; for (var i=0u;i<k;i++){ acc=acc+a[row*k+i]*w[i*n+col]; } c[row*n+col]=acc;
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
  var acc: array<f32,64>; for (var c=0u;c<hd;c++){ acc[c]=0.0; } var den=0.0;
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

struct W { buf: wgpu::Buffer }

pub struct GpuRope {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pl: HashMap<&'static str, wgpu::ComputePipeline>,
    w: HashMap<String, W>,
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
        for (k, src) in [("matmul", MATMUL), ("addbias", ADDBIAS), ("add", ADD), ("rmsnorm", RMSNORM),
                         ("rope", ROPE), ("gqa", GQA), ("swiglu", SWIGLU), ("rowdot", ROWDOT)] {
            let m = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some(k), source: wgpu::ShaderSource::Wgsl(src.into()) });
            pl.insert(k, device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(k), layout: None, module: &m, entry_point: "main", compilation_options: Default::default(), cache: None,
            }));
        }
        let c = &b.config; // [n_layer, H, nkv, hd, d, ffn, vocab, tied]
        let (n_layer, h, nkv, hd, d, ffn, vocab) =
            (c[0] as usize, c[1] as usize, c[2] as usize, c[3] as usize, c[4] as usize, c[5] as usize, c[6] as usize);
        let (theta, eps) = (b.config_f[0] as f32, b.config_f[1] as f32);
        let invf: Vec<f32> = (0..hd / 2).map(|j| 1.0 / theta.powf(2.0 * j as f32 / hd as f32)).collect();
        let store = |data: &[f32]| device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None, contents: cast(data),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        });
        let embed = store(&b.f32_array("embed").1);
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
        for nm in names { w.insert(nm.clone(), W { buf: store(&b.f32_array(&nm).1) }); }
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

    pub fn predict(&self, ids: &[i64], b: &Bundle) -> i64 {
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
        let mm = |m: usize, n: usize| (m.div_ceil(8) as u32, n.div_ceil(8) as u32);
        let g1 = |n: usize| (n.div_ceil(64) as u32, 1u32);
        let epsb = self.eps.to_bits();
        let (su, du) = (seq as u32, d as u32);
        for l in 0..self.n_layer {
            let p = format!("l{l}.");
            let wb = |s: &str| &self.w[&format!("{p}{s}")].buf;
            let bias = |s: &str| self.w.contains_key(&format!("{p}{s}.bias"));
            let mut r = |name, bufs: &[&wgpu::Buffer], dims, groups| self.record(&mut enc, &mut ku, &mut kb, name, bufs, dims, groups);
            r("rmsnorm", &[&x, wb("in_ln"), &a], [su, du, epsb, 0], g1(seq));
            r("matmul", &[&a, wb("self_attn.q_proj"), &q], [su, du, qd as u32, 0], mm(seq, qd));
            r("matmul", &[&a, wb("self_attn.k_proj"), &k], [su, du, kvd as u32, 0], mm(seq, kvd));
            r("matmul", &[&a, wb("self_attn.v_proj"), &v], [su, du, kvd as u32, 0], mm(seq, kvd));
            if bias("self_attn.q_proj") { r("addbias", &[&q, wb("self_attn.q_proj.bias")], [su, qd as u32, 0, 0], g1(seq * qd)); }
            if bias("self_attn.k_proj") { r("addbias", &[&k, wb("self_attn.k_proj.bias")], [su, kvd as u32, 0, 0], g1(seq * kvd)); }
            if bias("self_attn.v_proj") { r("addbias", &[&v, wb("self_attn.v_proj.bias")], [su, kvd as u32, 0, 0], g1(seq * kvd)); }
            r("rope", &[&q, &self.inv], [su, h as u32, hd as u32, 0], ((seq * h).div_ceil(64) as u32, 1));
            r("rope", &[&k, &self.inv], [su, nkv as u32, hd as u32, 0], ((seq * nkv).div_ceil(64) as u32, 1));
            r("gqa", &[&q, &k, &v, &attn], [su, hd as u32, h as u32, nkv as u32], ((seq * h).div_ceil(64) as u32, 1));
            r("matmul", &[&attn, wb("self_attn.o_proj"), &proj], [su, qd as u32, du, 0], mm(seq, d));
            r("add", &[&x, &proj], [(seq * d) as u32, 0, 0, 0], g1(seq * d));
            r("rmsnorm", &[&x, wb("post_ln"), &a2], [su, du, epsb, 0], g1(seq));
            r("matmul", &[&a2, wb("mlp.gate_proj"), &gate], [su, du, ffn as u32, 0], mm(seq, ffn));
            r("matmul", &[&a2, wb("mlp.up_proj"), &up], [su, du, ffn as u32, 0], mm(seq, ffn));
            r("swiglu", &[&gate, &up, &hbuf], [(seq * ffn) as u32, 0, 0, 0], g1(seq * ffn));
            r("matmul", &[&hbuf, wb("mlp.down_proj"), &down], [su, ffn as u32, du, 0], mm(seq, d));
            r("add", &[&x, &down], [(seq * d) as u32, 0, 0, 0], g1(seq * d));
        }
        self.record(&mut enc, &mut ku, &mut kb, "rmsnorm", &[&x, &self.w["norm"].buf, &xf], [su, du, epsb, 0], g1(seq));
        self.record(&mut enc, &mut ku, &mut kb, "rowdot", &[&xf, &self.embed, &logits],
                    [self.vocab as u32, du, ((seq - 1) * d) as u32, 0], (self.vocab.div_ceil(64) as u32, 1));
        self.queue.submit(Some(enc.finish()));
        let lg = self.read(&logits, self.vocab);
        lg.iter().enumerate().max_by(|x, y| x.1.partial_cmp(y.1).unwrap()).unwrap().0 as i64
    }
}

fn cast(s: &[f32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(s.as_ptr() as *const u8, std::mem::size_of_val(s)) }
}
fn cast_u32(s: &[u32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(s.as_ptr() as *const u8, std::mem::size_of_val(s)) }
}
