// This is the lifetime high-water mark of the current process, not a per-query sample.
pub(super) fn process_memory_peak_mb() -> Option<f64> {
    #[cfg(target_os = "linux")]
    {
        linux_peak_mb(&std::fs::read_to_string("/proc/self/status").ok()?)
    }
    #[cfg(target_os = "macos")]
    {
        macos::peak_mb()
    }
    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    {
        None
    }
}

#[cfg(any(target_os = "linux", test))]
fn linux_peak_mb(status: &str) -> Option<f64> {
    let (_, value) = status
        .lines()
        .filter_map(|line| line.split_once(':'))
        .find(|(field, _)| *field == "VmHWM")?;
    let mut words = value.split_whitespace();
    let kib = words.next()?.parse::<u64>().ok()?;
    if kib == 0 || words.next()? != "kB" || words.next().is_some() {
        return None;
    }
    Some(kib as f64 / 1024.0)
}

#[cfg(target_os = "macos")]
mod macos {
    use std::os::raw::{c_int, c_long};

    // Darwin SDK: sys/_types/_timeval.h and sys/resource.h. ru_maxrss is bytes.
    #[derive(Default)]
    #[repr(C)]
    struct Timeval {
        seconds: c_long,
        microseconds: c_int,
    }

    #[derive(Default)]
    #[repr(C)]
    struct Rusage {
        user_time: Timeval,
        system_time: Timeval,
        max_rss: c_long,
        remaining_counters: [c_long; 13],
    }

    pub(super) fn peak_mb() -> Option<f64> {
        extern "C" {
            fn getrusage(who: c_int, usage: *mut Rusage) -> c_int;
        }
        let mut usage = Rusage::default();
        // RUSAGE_SELF = 0; the initialized, repr(C) buffer matches Darwin's rusage ABI.
        let status = unsafe { getrusage(0, &mut usage) };
        (status == 0 && usage.max_rss > 0).then_some(usage.max_rss as f64 / 1_048_576.0)
    }

    #[test]
    #[cfg(target_pointer_width = "64")]
    fn rusage_matches_supported_darwin_abi() {
        assert_eq!(std::mem::size_of::<Timeval>(), 16);
        assert_eq!(std::mem::size_of::<Rusage>(), 144);
        assert_eq!(std::mem::align_of::<Rusage>(), 8);
        assert_eq!(std::mem::offset_of!(Rusage, max_rss), 32);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linux_peak_uses_only_valid_high_water_marks_across_100_samples() {
        for seed in 1..=100 {
            let kib = seed * 1024 + 17;
            for status in [
                format!("Name:\tcoffer\nVmRSS:\t1 kB\nVmHWM:\t{kib} kB\n"),
                format!("VmHWM: {kib} kB\nVmRSS: 999999999 kB\n"),
            ] {
                assert_eq!(linux_peak_mb(&status), Some(kib as f64 / 1024.0));
            }
            for invalid in [
                "-1 kB",
                "0 kB",
                "NaN kB",
                "inf kB",
                "1.5 kB",
                "1 MB",
                "1",
                "1 kB extra",
                "18446744073709551616 kB",
            ] {
                assert_eq!(
                    linux_peak_mb(&format!("VmRSS: {kib} kB\nVmHWM: {invalid}\n")),
                    None
                );
            }
            assert_eq!(linux_peak_mb(&format!("VmRSS: {kib} kB\n")), None);
        }
    }

    #[test]
    #[cfg(any(target_os = "macos", target_os = "linux"))]
    fn peak_survives_release_of_touched_memory() {
        const CHILD: &str = "COFFERDB_TEST_PEAK_RSS_CHILD";
        if std::env::var_os(CHILD).is_some() {
            let before = process_memory_peak_mb().expect("supported host reports peak RSS");
            let pages = std::hint::black_box(vec![0x5a_u8; 96 * 1024 * 1024]);
            let during = process_memory_peak_mb().unwrap();
            assert!(during >= before + 64.0, "before={before} during={during}");
            drop(pages);
            let after = process_memory_peak_mb().unwrap();
            // Linux VmHWM uses approximate RSS counters; verify retention, not exact monotonicity.
            assert!(after >= before + 64.0, "before={before} after={after}");
            #[cfg(target_os = "macos")]
            assert!(after >= during, "during={during} after={after}");
            return;
        }
        // Isolate the measurement from memory peaks of concurrently executing tests.
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "benchmark::runner::memory::tests::peak_survives_release_of_touched_memory",
                "--nocapture",
            ])
            .env(CHILD, "1")
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(String::from_utf8_lossy(&output.stdout).contains("1 passed"));
    }
}
