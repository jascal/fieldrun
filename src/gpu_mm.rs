//! GPU matmul via wgpu (opt-in, `--features gpu`). A WGSL compute kernel for C = A·B over f32 storage buffers, behind
//! a tiny context that owns the device/queue/pipeline. This is the GPU counterpart of `Bundle::mm`'s CPU path; it is
//! validated against the CPU matmul (faithfulness gate) before being wired into the forward pass. Cross-platform via
//! wgpu: Metal (Apple), DX12 (Windows incl. ARM), Vulkan (Linux).

use wgpu::util::DeviceExt;

const SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> c: array<f32>;
@group(0) @binding(3) var<uniform> dims: vec4<u32>; // m, k, n, _

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let m = dims.x; let k = dims.y; let n = dims.z;
    let row = gid.x; let col = gid.y;
    if (row >= m || col >= n) { return; }
    var acc = 0.0;
    for (var i = 0u; i < k; i = i + 1u) {
        acc = acc + a[row * k + i] * b[i * n + col];
    }
    c[row * n + col] = acc;
}
"#;

pub struct GpuCtx {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    pub name: String,
}

impl GpuCtx {
    /// Initialise the GPU context (adapter → device/queue → compiled matmul pipeline). None if no adapter.
    pub fn new() -> Option<GpuCtx> {
        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..Default::default()
        }))?;
        let name = format!("{}, {:?}", adapter.get_info().name, adapter.get_info().backend);
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default(), None)).ok()?;
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("matmul"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("matmul"),
            layout: None,
            module: &module,
            entry_point: "main",
            compilation_options: Default::default(),
            cache: None,
        });
        Some(GpuCtx { device, queue, pipeline, name })
    }

    /// C = A (m,k) · B (k,n), both row-major f32 → (m*n) row-major. Runs on the GPU, reads the result back.
    pub fn matmul(&self, a: &[f32], m: usize, k: usize, b: &[f32], n: usize) -> Vec<f32> {
        let (d, q) = (&self.device, &self.queue);
        let abuf = d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None, contents: bytemuck_cast(a), usage: wgpu::BufferUsages::STORAGE,
        });
        let bbuf = d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None, contents: bytemuck_cast(b), usage: wgpu::BufferUsages::STORAGE,
        });
        let csize = (m * n * 4) as u64;
        let cbuf = d.create_buffer(&wgpu::BufferDescriptor {
            label: None, size: csize, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false,
        });
        let dims = [m as u32, k as u32, n as u32, 0u32];
        let dbuf = d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None, contents: bytemuck_cast_u32(&dims), usage: wgpu::BufferUsages::UNIFORM,
        });
        let bind = d.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: abuf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: bbuf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: cbuf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: dbuf.as_entire_binding() },
            ],
        });
        let mut enc = d.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind, &[]);
            pass.dispatch_workgroups(m.div_ceil(8) as u32, n.div_ceil(8) as u32, 1);
        }
        let staging = d.create_buffer(&wgpu::BufferDescriptor {
            label: None, size: csize, usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false,
        });
        enc.copy_buffer_to_buffer(&cbuf, 0, &staging, 0, csize);
        q.submit(Some(enc.finish()));
        let slice = staging.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        d.poll(wgpu::Maintain::Wait);
        let data = slice.get_mapped_range();
        let out: Vec<f32> = data.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
        drop(data);
        staging.unmap();
        out
    }
}

fn bytemuck_cast(s: &[f32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(s.as_ptr() as *const u8, std::mem::size_of_val(s)) }
}

fn bytemuck_cast_u32(s: &[u32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(s.as_ptr() as *const u8, std::mem::size_of_val(s)) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gpu_matmul_matches_cpu() {
        let Some(ctx) = GpuCtx::new() else {
            eprintln!("no GPU adapter — skipping");
            return;
        };
        // small known matmul: A (2,3) · B (3,2)
        let a = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = [7.0f32, 8.0, 9.0, 10.0, 11.0, 12.0];
        let got = ctx.matmul(&a, 2, 3, &b, 2);
        // CPU reference
        let mut want = vec![0f32; 4];
        for i in 0..2 {
            for j in 0..2 {
                for k in 0..3 {
                    want[i * 2 + j] += a[i * 3 + k] * b[k * 2 + j];
                }
            }
        }
        for (g, w) in got.iter().zip(&want) {
            assert!((g - w).abs() < 1e-3, "got {got:?} want {want:?}");
        }
    }
}
