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
        for (k, src) in [("matmul", MATMUL), ("addbias", ADDBIAS), ("add", ADD), ("gelu", GELU), ("layernorm", LAYERNORM)] {
            let m = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some(k), source: wgpu::ShaderSource::Wgsl(src.into()) });
            pl.insert(k, device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(k), layout: None, module: &m, entry_point: "main", compilation_options: Default::default(), cache: None,
            }));
        }
        let (n_layer, n_head, d, vocab) = (b.config[0] as usize, b.config[1] as usize, b.config[2] as usize, b.config[4] as usize);
        let mut g = GpuGpt2 { device, queue, pl, w: HashMap::new(), n_layer, n_head, d, vocab, name };
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

    fn run(&self, name: &str, bufs: &[&wgpu::Buffer], dims: [u32; 4], groups: (u32, u32)) {
        let pipe = &self.pl[name];
        let dbuf = self.uniform(dims);
        let mut entries: Vec<wgpu::BindGroupEntry> = bufs.iter().enumerate()
            .map(|(i, b)| wgpu::BindGroupEntry { binding: i as u32, resource: b.as_entire_binding() }).collect();
        entries.push(wgpu::BindGroupEntry { binding: bufs.len() as u32, resource: dbuf.as_entire_binding() });
        let bind = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None, layout: &pipe.get_bind_group_layout(0), entries: &entries,
        });
        let mut enc = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(pipe);
            pass.set_bind_group(0, &bind, &[]);
            pass.dispatch_workgroups(groups.0, groups.1, 1);
        }
        self.queue.submit(Some(enc.finish()));
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

    fn matmul(&self, a: &wgpu::Buffer, m: usize, k: usize, wname: &str) -> wgpu::Buffer {
        let wt = &self.w[wname];
        let (k2, n) = (wt.rows, wt.cols);
        assert_eq!(k, k2, "matmul k mismatch for {wname}");
        let c = self.storage(m * n);
        self.run("matmul", &[a, &wt.buf, &c], [m as u32, k as u32, n as u32, 0], (m.div_ceil(8) as u32, n.div_ceil(8) as u32));
        c
    }

    fn add_bias(&self, c: &wgpu::Buffer, m: usize, n: usize, bname: &str) {
        self.run("addbias", &[c, &self.w[bname].buf], [m as u32, n as u32, 0, 0], ((m * n).div_ceil(64) as u32, 1));
    }

    fn layernorm(&self, x: &wgpu::Buffer, m: usize, gname: &str, bname: &str) -> wgpu::Buffer {
        let out = self.storage(m * self.d);
        self.run("layernorm", &[x, &self.w[gname].buf, &self.w[bname].buf, &out], [m as u32, self.d as u32, 0, 0], (m.div_ceil(64) as u32, 1));
        out
    }

    /// GPU forward; returns the last-position hidden (d,) read back for the CPU unembed.
    fn last_hidden(&self, ids: &[i64], wte: &[f32], wpe: &[f32]) -> Vec<f32> {
        let (seq, d, hd) = (ids.len(), self.d, self.d / self.n_head);
        let mut x0 = vec![0f32; seq * d]; // embed on CPU
        for (t, &id) in ids.iter().enumerate() {
            for c in 0..d {
                x0[t * d + c] = wte[id as usize * d + c] + wpe[t * d + c];
            }
        }
        let x = self.storage_init(&x0);
        for l in 0..self.n_layer {
            let p = format!("h{l}.");
            let a = self.layernorm(&x, seq, &format!("{p}ln_1.weight"), &format!("{p}ln_1.bias"));
            let qkv = self.matmul(&a, seq, d, &format!("{p}attn.c_attn.weight"));
            self.add_bias(&qkv, seq, 3 * d, &format!("{p}attn.c_attn.bias"));
            // attention on CPU (v1): read qkv, masked softmax per head, upload attn_out
            let qkv_c = self.read(&qkv, seq * 3 * d);
            let mut attn = vec![0f32; seq * d];
            for head in 0..self.n_head {
                for i in 0..seq {
                    let mut sc = vec![0f32; i + 1];
                    let mut mx = f32::NEG_INFINITY;
                    for (j, scj) in sc.iter_mut().enumerate() {
                        let mut s = 0.0;
                        for c in 0..hd {
                            s += qkv_c[i * 3 * d + head * hd + c] * qkv_c[j * 3 * d + d + head * hd + c];
                        }
                        *scj = s / (hd as f32).sqrt();
                        mx = mx.max(*scj);
                    }
                    let mut den = 0.0;
                    for s in sc.iter_mut() { *s = (*s - mx).exp(); den += *s; }
                    for c in 0..hd {
                        let mut o = 0.0;
                        for (j, &s) in sc.iter().enumerate() {
                            o += s / den * qkv_c[j * 3 * d + 2 * d + head * hd + c];
                        }
                        attn[i * d + head * hd + c] = o;
                    }
                }
            }
            let attn_b = self.storage_init(&attn);
            let proj = self.matmul(&attn_b, seq, d, &format!("{p}attn.c_proj.weight"));
            self.add_bias(&proj, seq, d, &format!("{p}attn.c_proj.bias"));
            self.run("add", &[&x, &proj], [(seq * d) as u32, 0, 0, 0], ((seq * d).div_ceil(64) as u32, 1));
            let a2 = self.layernorm(&x, seq, &format!("{p}ln_2.weight"), &format!("{p}ln_2.bias"));
            let h = self.matmul(&a2, seq, d, &format!("{p}mlp.c_fc.weight"));
            self.add_bias(&h, seq, 4 * d, &format!("{p}mlp.c_fc.bias"));
            self.run("gelu", &[&h], [(seq * 4 * d) as u32, 0, 0, 0], ((seq * 4 * d).div_ceil(64) as u32, 1));
            let mp = self.matmul(&h, seq, 4 * d, &format!("{p}mlp.c_proj.weight"));
            self.add_bias(&mp, seq, d, &format!("{p}mlp.c_proj.bias"));
            self.run("add", &[&x, &mp], [(seq * d) as u32, 0, 0, 0], ((seq * d).div_ceil(64) as u32, 1));
        }
        let xf = self.layernorm(&x, seq, "ln_f.weight", "ln_f.bias");
        let all = self.read(&xf, seq * d);
        all[(seq - 1) * d..seq * d].to_vec()
    }

    /// Predict the next token: GPU forward for the hidden state, CPU unembed (tied wte) for argmax.
    pub fn predict(&self, ids: &[i64], b: &Bundle) -> i64 {
        let (_, wte) = b.f32_array("wte");
        let (_, wpe) = b.f32_array("wpe");
        let last = self.last_hidden(ids, &wte, &wpe);
        let logits = b.rowdot_f32("wte", &last);
        logits.iter().enumerate().max_by(|a, c| a.1.partial_cmp(c.1).unwrap()).unwrap().0 as i64
    }
}

fn cast(s: &[f32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(s.as_ptr() as *const u8, std::mem::size_of_val(s)) }
}
fn cast_u32(s: &[u32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(s.as_ptr() as *const u8, std::mem::size_of_val(s)) }
}
