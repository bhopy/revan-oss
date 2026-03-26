// monitor.rs — VRAM monitoring via nvml-wrapper
//
// Polls NVIDIA GPU every N seconds, warns if VRAM exceeds threshold.
// Critical for 8GB VRAM limit (RTX 2060 Super).

use tracing::{debug, error, info, warn};

/// VRAM monitor for NVIDIA GPUs.
pub struct VramMonitor {
    /// NVML instance (kept alive for the lifetime of the monitor)
    nvml: Option<nvml_wrapper::Nvml>,
    /// GPU device index (usually 0)
    device_index: u32,
    /// Warning threshold (0.0 - 1.0, e.g. 0.95 = warn at 95% usage)
    warn_threshold: f64,
}

/// Snapshot of current VRAM state.
#[derive(Debug, Clone)]
pub struct VramStatus {
    pub used_mib: u32,
    pub total_mib: u32,
    pub free_mib: u32,
    pub usage_percent: f64,
}

impl VramMonitor {
    /// Create a new VRAM monitor. Returns Ok even if NVML fails (monitor becomes a no-op).
    pub fn new(warn_threshold_percent: u32) -> Self {
        let nvml = match nvml_wrapper::Nvml::init() {
            Ok(n) => {
                info!("NVML initialized successfully");
                Some(n)
            }
            Err(e) => {
                warn!(error = %e, "NVML init failed — VRAM monitoring disabled");
                None
            }
        };

        Self {
            nvml,
            device_index: 0,
            warn_threshold: warn_threshold_percent as f64 / 100.0,
        }
    }

    /// Get current VRAM status. Returns None if NVML is unavailable.
    pub fn status(&self) -> Option<VramStatus> {
        let nvml = self.nvml.as_ref()?;

        let device = match nvml.device_by_index(self.device_index) {
            Ok(d) => d,
            Err(e) => {
                error!(error = %e, "failed to get GPU device");
                return None;
            }
        };

        let mem_info = match device.memory_info() {
            Ok(m) => m,
            Err(e) => {
                error!(error = %e, "failed to get memory info");
                return None;
            }
        };

        let total_mib = (mem_info.total / (1024 * 1024)) as u32;
        let used_mib = (mem_info.used / (1024 * 1024)) as u32;
        let free_mib = total_mib.saturating_sub(used_mib);
        let usage_percent = if mem_info.total > 0 {
            mem_info.used as f64 / mem_info.total as f64
        } else {
            0.0
        };

        Some(VramStatus {
            used_mib,
            total_mib,
            free_mib,
            usage_percent,
        })
    }

    /// Check if VRAM usage is below the warning threshold.
    pub fn is_safe(&self) -> bool {
        match self.status() {
            Some(s) => s.usage_percent < self.warn_threshold,
            None => true, // if we can't check, assume safe
        }
    }

    /// Spawn a background task that polls VRAM every `interval_secs` seconds.
    pub fn spawn_watchdog(&self, interval_secs: u64) {
        if self.nvml.is_none() {
            info!("VRAM watchdog not started (NVML unavailable)");
            return;
        }

        // We can't move self into the task, so we clone the config
        let device_index = self.device_index;
        let warn_threshold = self.warn_threshold;

        tokio::spawn(async move {
            // Re-init NVML inside the task (Nvml isn't Send)
            let nvml = match nvml_wrapper::Nvml::init() {
                Ok(n) => n,
                Err(e) => {
                    error!(error = %e, "VRAM watchdog: NVML init failed");
                    return;
                }
            };

            let mut interval = tokio::time::interval(
                std::time::Duration::from_secs(interval_secs),
            );

            loop {
                interval.tick().await;

                let device = match nvml.device_by_index(device_index) {
                    Ok(d) => d,
                    Err(_) => continue,
                };

                let mem = match device.memory_info() {
                    Ok(m) => m,
                    Err(_) => continue,
                };

                let used_mib = mem.used / (1024 * 1024);
                let total_mib = mem.total / (1024 * 1024);
                let usage = mem.used as f64 / mem.total as f64;

                if usage >= warn_threshold {
                    warn!(
                        used_mib,
                        total_mib,
                        usage_pct = format!("{:.1}%", usage * 100.0).as_str(),
                        "VRAM usage above threshold!"
                    );
                } else {
                    debug!(
                        used_mib,
                        total_mib,
                        usage_pct = format!("{:.1}%", usage * 100.0).as_str(),
                        "VRAM check ok"
                    );
                }
            }
        });

        info!(
            interval_secs,
            threshold_pct = format!("{:.0}%", self.warn_threshold * 100.0).as_str(),
            "VRAM watchdog started"
        );
    }
}
