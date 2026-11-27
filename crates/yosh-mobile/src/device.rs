//! Device tier and memory budget from the platform's RAM / CPU probes.

use super::*;

/// `(budget_mb, total_mb)` — the memory the reader may use for its page cache +
/// GPU textures, and the device's raw RAM (which the tier heuristic reads).
///
/// Decoded pages and GPU textures are *native* allocations, not bounded by the
/// (small) Java heap, so a healthy slice of device RAM is fine. But **MemTotal
/// alone is a lie about what's available**: the old `MemTotal/8` made every ≥3 GB
/// phone — a cheap one included — saturate every `Budget::derive` clamp at the
/// desktop maxima, which is the main reason yosh ran badly on slow devices. Take
/// the smaller of a total-RAM slice and a third of what's actually free right now,
/// then clamp into a range that keeps a small device conservative and a big one
/// from claiming more than the reader can usefully hold.
///
/// **Probed once per process.** `MemTotal` is a constant, and while `MemAvailable`
/// genuinely moves, re-reading it per resume bought nothing: the budget it feeds is
/// a coarse tier of caps, and a resume is exactly when the number is least
/// meaningful (we have just torn our own textures down, so free memory is
/// transiently inflated). First-launch `MemAvailable` is as good a sample as any,
/// and it keeps the resume path off `/proc`. The real answer to a squeeze is
/// [`Shell::memory_warning`], which sheds caches on demand rather than guessing.
pub(crate) fn device_mem(platform: &dyn Platform) -> (u64, u64) {
    static DEVICE_MEM: std::sync::OnceLock<(u64, u64)> = std::sync::OnceLock::new();
    *DEVICE_MEM.get_or_init(|| {
        let (total, avail) = platform.mem_mb();
        let budget = (total / 8).min(avail / 3).clamp(192, 1024);
        (budget, total)
    })
}

/// Classify the device from RAM + peak CPU clock. Deliberately conservative in the
/// middle: only a clearly-flagship device (≥ 8 GB **and** ≥ 2.5 GHz) gets `High`,
/// which is the tier that applies no ceilings at all. Never tier on GPU name —
/// the same GPU string ships across wildly different thermal envelopes.
///
/// (Pixel 9 Pro XL → High; Snapdragon 695 / 6 GB → Mid; Helio G85 / 4 GB → Low.)
pub(crate) fn device_tier(total_mb: u64, max_khz: Option<u64>) -> DeviceTier {
    let gb = total_mb as f64 / 1024.0;
    match max_khz {
        Some(khz) if khz < 2_100_000 => DeviceTier::Low,
        Some(khz) => {
            if gb < 4.0 {
                DeviceTier::Low
            } else if gb >= 8.0 && khz >= 2_500_000 {
                DeviceTier::High
            } else {
                DeviceTier::Mid
            }
        }
        // Clock unreadable: RAM-only fallback.
        None if gb < 4.0 => DeviceTier::Low,
        None if gb >= 8.0 => DeviceTier::High,
        None => DeviceTier::Mid,
    }
}
