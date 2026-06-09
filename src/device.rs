//! Device selection — pick CPU or a GPU (wgpu) backend by preference + a memory budget, with CPU fallback.
//!
//! The CPU path is the default and the faithful reference (the "minimum to run on a CPU" thesis); the GPU is an opt-in
//! accelerator for usable generation speed, built with `--features gpu`. On unified-memory machines (Apple silicon) the
//! GPU shares the one RAM pool, so the budget is against total memory, not a separate VRAM copy. The budget caps us to
//! consumer hardware (default 24 GB): GPU is chosen only when an adapter exists AND the model fits the budget, else CPU.

pub struct Choice {
    pub use_gpu: bool,
    pub detail: String, // human-readable reason, shown at startup
}

fn mb(b: u64) -> u64 {
    b / 1_000_000
}

/// Select the compute device. `pref` ∈ {cpu, gpu, auto}; `model_bytes` = resident weight size; `budget_bytes` = the
/// consumer VRAM/unified-memory cap.
pub fn select(pref: &str, model_bytes: u64, budget_bytes: u64) -> Choice {
    let cpu = |why: String| Choice { use_gpu: false, detail: why };
    match pref {
        "cpu" => cpu("CPU (requested)".into()),
        "gpu" | "auto" => {
            #[cfg(feature = "gpu")]
            {
                match gpu_probe() {
                    Some(name) if model_bytes <= budget_bytes => Choice {
                        use_gpu: true,
                        detail: format!("GPU [{name}] — model {} MB fits {} MB budget", mb(model_bytes), mb(budget_bytes)),
                    },
                    Some(name) => cpu(format!(
                        "CPU fallback — model {} MB exceeds {} MB budget on GPU [{name}]",
                        mb(model_bytes), mb(budget_bytes)
                    )),
                    None => cpu("CPU — no GPU adapter found".into()),
                }
            }
            #[cfg(not(feature = "gpu"))]
            {
                let _ = (model_bytes, budget_bytes);
                cpu("CPU (built without the gpu feature — rebuild with `--features gpu` for a GPU)".into())
            }
        }
        other => cpu(format!("CPU (unknown --device {other:?}; use cpu|gpu|auto)")),
    }
}

#[cfg(feature = "gpu")]
fn gpu_probe() -> Option<String> {
    let instance = wgpu::Instance::default();
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        ..Default::default()
    }))?;
    let info = adapter.get_info();
    Some(format!("{}, {:?}", info.name, info.backend))
}
