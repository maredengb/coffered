include!("read_probe_bindings.rs");
include!("sample_events.rs");
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mode = std::env::args().nth(1).unwrap();
    if mode == "clock" {
        let (ns, numer, denom) = sample_clock();
        println!("{{\"monotonic_ns\":{ns},\"numer\":{numer},\"denom\":{denom}}}");
        return Ok(());
    }
    with_sample("trial-00", "control:sample", "control-query", |sample| {
        sample.begin_api(); std::thread::sleep(std::time::Duration::from_millis(5));
        if mode == "error" { return Err("synthetic control error".into()); }
        if mode == "panic" { panic!("synthetic control panic"); }
        let (elapsed, reads) = sample.finish_api(); assert_eq!(reads, "null");
        if mode == "wrong_result" {
            sample_output(&format!("{{\"event\":\"sample_result\",\"trial_id\":\"trial-00\",\"sample_id\":\"control:sample\",\"query_id\":\"control-query\",\"attempt\":1,\"mode\":\"text\",\"elapsed_ms\":{elapsed},\"result_retained_before_validation\":true,\"returned_count\":1,\"result\":{{\"results\":[{{\"rank\":9,\"chunk_external_id\":\"wrong-id\",\"final_score\":0}}]}}}}"))?;
            panic!("synthetic result validation failed after raw retention");
        }
        Ok(())
    })
}
