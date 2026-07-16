//! Device selection + the startup "does this model fit?" line.
//!
//! Generation runs on the **CPU** kernels — that's the faithful reference, and the GPU (opt-in `--features gpu`) is
//! currently used only by `--gpu-check`, not by chat/serve/generate. So the constraint that actually matters is
//! **system RAM**: the CPU loads the weight bundle into RAM. We detect total RAM and report `model MB / RAM MB`, with
//! a warning when the model is too big to fit comfortably (→ it will swap). `--max-vram <GB>` overrides the budget.

pub struct Choice {
    pub use_gpu: bool,
    pub detail: String, // human-readable reason, shown at startup
}

fn mb(b: u64) -> u64 {
    b / 1_000_000
}

/// Total system RAM in bytes — the real "does it fit?" budget (the CPU loads the weights into RAM). None if unknown.
pub fn total_ram_bytes() -> Option<u64> {
    #[cfg(target_os = "linux")]
    if let Ok(s) = std::fs::read_to_string("/proc/meminfo") {
        for line in s.lines() {
            if let Some(rest) = line.strip_prefix("MemTotal:") {
                if let Some(kb) = rest.split_whitespace().next().and_then(|n| n.parse::<u64>().ok()) {
                    return Some(kb * 1024);
                }
            }
        }
    }
    #[cfg(target_os = "macos")]
    if let Ok(out) = std::process::Command::new("sysctl").args(["-n", "hw.memsize"]).output() {
        if let Ok(b) = String::from_utf8_lossy(&out.stdout).trim().parse::<u64>() {
            return Some(b);
        }
    }
    None
}

/// The startup device/fit line. Generation runs on the CPU regardless; `ram_bytes` is the fit budget (detected RAM or
/// the `--max-vram` override, 0 = unknown). Warns when the model won't fit comfortably (it'll swap).
pub fn select(pref: &str, model_bytes: u64, ram_bytes: u64) -> Choice {
    let fit = if ram_bytes > 0 {
        format!("model {} MB / {} MB RAM", mb(model_bytes), mb(ram_bytes))
    } else {
        format!("model {} MB", mb(model_bytes))
    };
    // leave ~2 GB headroom for the OS + activations/KV cache; over that, the weights won't fit in RAM → swapping.
    let warn = if ram_bytes > 0 && model_bytes + 2_000_000_000 > ram_bytes {
        "  ⚠ exceeds free RAM — expect heavy swapping; use a smaller model or --dtype int8"
    } else {
        ""
    };
    let cpu = |detail: String| Choice { use_gpu: false, detail };
    match pref {
        "cpu" => cpu(format!("CPU (requested) · {fit}{warn}")),
        // Explicit --device gpu only: opt-in/experimental (GPU residual dumps via --source-dump). "auto" stays CPU.
        "gpu" => {
            #[cfg(feature = "gpu")]
            {
                match gpu_probe() {
                    Some(n) => Choice {
                        use_gpu: true,
                        detail: format!("GPU [{n}] · {fit}{warn} · GPU path active for --source-dump (rope)"),
                    },
                    None => cpu(format!("CPU · {fit}{warn} · GPU requested but no adapter found")),
                }
            }
            #[cfg(not(feature = "gpu"))]
            {
                cpu(format!("CPU · {fit}{warn} · GPU requested but not built with --features gpu"))
            }
        }
        "auto" => {
            #[cfg(feature = "gpu")]
            {
                let g = gpu_probe()
                    .map(|n| format!(" · GPU [{n}] present (opt-in via --device gpu for --source-dump)"))
                    .unwrap_or_default();
                cpu(format!("CPU · {fit}{warn}{g}"))
            }
            #[cfg(not(feature = "gpu"))]
            {
                cpu(format!("CPU · {fit}{warn}"))
            }
        }
        other => cpu(format!("CPU (unknown --device {other:?}; use cpu|gpu) · {fit}")),
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
