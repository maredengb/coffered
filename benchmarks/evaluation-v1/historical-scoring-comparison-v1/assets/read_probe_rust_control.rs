use std::{fs::{self,File},io::{Read,Seek,SeekFrom},os::unix::fs::FileExt,path::PathBuf};
#[repr(C)] struct Probe{calls:u64,requested:u64,returned:u64,errors:u64,saturated:u64}
extern "C" {fn supplement_read_begin();fn supplement_read_end()->Probe;}
fn main(){let p=PathBuf::from(std::env::args().nth(1).unwrap());assert!(!p.exists());fs::write(&p,b"abcdef").unwrap();let mut f=File::open(&p).unwrap();let mut b=[0;2];
unsafe{supplement_read_begin()};f.read_exact(&mut b).unwrap();assert_eq!(&b,b"ab");f.read_exact_at(&mut b,2).unwrap();assert_eq!(&b,b"cd");f.seek(SeekFrom::Start(4)).unwrap();f.read_exact(&mut b).unwrap();assert_eq!(&b,b"ef");let q=unsafe{supplement_read_end()};assert_eq!((q.calls,q.requested,q.returned,q.errors,q.saturated),(3,6,6,0,0));println!("PASS Rust std::fs File read_exact/read_exact_at:3 calls/6 bytes; no engine query");}
