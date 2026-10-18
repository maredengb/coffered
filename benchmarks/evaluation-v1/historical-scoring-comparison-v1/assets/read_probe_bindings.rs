#[cfg(read_probe)]
#[repr(C)]
#[derive(Default)]
struct ReadProbe { calls:u64, requested:u64, returned:u64, errors:u64, saturated:u64 }
#[cfg(read_probe)]
extern "C" { fn supplement_read_begin(); fn supplement_read_end() -> ReadProbe; }
fn begin_read_probe() {
    #[cfg(read_probe)] unsafe { supplement_read_begin(); }
}
fn end_read_probe() -> String {
    #[cfg(read_probe)] {
        let row=unsafe { supplement_read_end() };
        assert_eq!(row.saturated,0);
        assert_eq!(row.errors,0,"read error requires separate failure analysis");
        return format!("{{\"scope\":\"querying_thread_userspace_read_pread_readv\",\"calls\":{},\"known_requested_bytes\":{},\"returned_bytes\":{},\"errors\":{},\"saturated\":false,\"physical_read_bytes\":null}}",row.calls,row.requested,row.returned,row.errors);
    }
    #[cfg(not(read_probe))] { "null".into() }
}
