// Query sample events. API timing stays separate from event publication.
#[repr(C)]
struct MachTimebase { numer: u32, denom: u32 }
extern "C" {
    fn mach_absolute_time() -> u64;
    fn mach_timebase_info(info: *mut MachTimebase) -> i32;
}
fn sample_clock() -> (u64, u32, u32) {
    let mut info = MachTimebase { numer: 0, denom: 0 };
    assert_eq!(unsafe { mach_timebase_info(&mut info) }, 0);
    assert!(info.numer > 0 && info.denom > 0);
    let ticks = unsafe { mach_absolute_time() };
    let ns = (ticks as u128 * info.numer as u128 / info.denom as u128).try_into().unwrap();
    (ns, info.numer, info.denom)
}
fn sample_json_string(value: &str) -> String {
    let mut out = String::from("\"");
    for c in value.chars() {
        match c {
            '"' => out.push_str("\\\""), '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"), '\r' => out.push_str("\\r"), '\t' => out.push_str("\\t"),
            c if c < '\u{20}' => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"'); out
}
fn sample_output(row: &str) -> std::io::Result<()> {
    use std::io::Write;
    let mut out = std::io::stdout().lock();
    writeln!(out, "{row}")?; out.flush()
}
struct SampleContext {
    api_start: Option<std::time::Instant>, api_elapsed_ms: Option<f64>,
    elapsed_scope: &'static str, reads_active: bool,
}
impl SampleContext {
    fn begin_api(&mut self) {
        begin_read_probe(); self.reads_active = true;
        self.api_start = Some(std::time::Instant::now());
    }
    fn finish_api(&mut self) -> (f64, String) {
        let elapsed = self.api_start.unwrap().elapsed().as_secs_f64() * 1000.0;
        self.api_elapsed_ms = Some(elapsed); self.elapsed_scope = "completed_api_scope";
        self.reads_active = false;
        (elapsed, end_read_probe())
    }
    fn finish_error(&mut self) {
        if self.api_elapsed_ms.is_none() {
            self.api_elapsed_ms = self.api_start.map(|start| start.elapsed().as_secs_f64() * 1000.0);
            self.elapsed_scope = "api_until_error_observed_in_adapter";
        }
        if self.reads_active {
            self.reads_active = false;
            // Preserve the original query failure if diagnostics also fail.
            let _ = std::panic::catch_unwind(end_read_probe);
        }
    }
}
fn with_sample<F>(trial_id: &str, sample_id: &str, query_id: &str, body: F) -> Result<(), Box<dyn std::error::Error>>
where F: FnOnce(&mut SampleContext) -> Result<(), Box<dyn std::error::Error>> {
    assert!(trial_id.len() == 8 && trial_id.starts_with("trial-") && trial_id[6..].chars().all(|c| c.is_ascii_digit()));
    let (start, numer, denom) = sample_clock();
    sample_output(&format!("{{\"event\":\"sample_start\",\"trial_id\":{},\"sample_id\":{},\"query_id\":{},\"attempt\":1,\"monotonic_ns\":{start},\"clock\":\"mach_absolute_time_ns\",\"numer\":{numer},\"denom\":{denom}}}",
        sample_json_string(trial_id), sample_json_string(sample_id), sample_json_string(query_id)))?;
    let mut context = SampleContext { api_start: None, api_elapsed_ms: None, elapsed_scope: "not_started", reads_active: false };
    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| body(&mut context)));
    let (status, reason) = match &outcome {
        Ok(Ok(())) => ("success", String::new()),
        Ok(Err(error)) => ("error", error.to_string()),
        Err(value) => ("panic", value.downcast_ref::<String>().cloned().or_else(|| value.downcast_ref::<&str>().map(|s| s.to_string())).unwrap_or_else(|| "non-string panic".into())),
    };
    if status != "success" { context.finish_error(); }
    let (end, _, _) = sample_clock();
    let elapsed = context.api_elapsed_ms.map_or("null".into(), |v| v.to_string());
    let terminal = format!("{{\"event\":\"sample_terminal\",\"trial_id\":{},\"sample_id\":{},\"query_id\":{},\"attempt\":1,\"status\":\"{status}\",\"reason\":{},\"monotonic_ns\":{end},\"clock\":\"mach_absolute_time_ns\",\"api_elapsed_ms\":{elapsed},\"api_elapsed_scope\":\"{}\",\"attempt_until_terminal_creation_ms\":{}}}",
        sample_json_string(trial_id), sample_json_string(sample_id), sample_json_string(query_id),
        sample_json_string(&reason.chars().take(512).collect::<String>()), context.elapsed_scope, end.saturating_sub(start) as f64 / 1e6);
    let published = sample_output(&terminal);
    match outcome {
        Ok(Ok(())) => { published?; Ok(()) },
        Ok(Err(error)) => { let _ = published; Err(error) },
        Err(payload) => { let _ = published; std::panic::resume_unwind(payload) },
    }
}
