use std::io::{BufWriter, Read, Write};
use std::path::Path;
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::sync::{Arc, Mutex};

use crate::{Error, ErrorCode, Result};

struct ChildGuard {
    child: Arc<Mutex<Child>>,
}

struct InputGuard(Option<BufWriter<ChildStdin>>);

impl Drop for InputGuard {
    fn drop(&mut self) {
        if let Some(output) = self.0.take() {
            let (stdin, _) = output.into_parts();
            drop(stdin);
        }
    }
}

impl Drop for ChildGuard {
    fn drop(&mut self) {
        let mut child = self.child.lock().unwrap_or_else(|error| error.into_inner());
        let _ = child.kill();
        let _ = child.wait();
    }
}

struct KillOnFailure(Option<Arc<Mutex<Child>>>);

impl Drop for KillOnFailure {
    fn drop(&mut self) {
        if let Some(child) = self.0.take() {
            let _ = child
                .lock()
                .unwrap_or_else(|error| error.into_inner())
                .kill();
        }
    }
}

pub(super) fn execute(
    path: &Path,
    write: impl FnOnce(&mut BufWriter<ChildStdin>) -> Result<()>,
) -> Result<()> {
    execute_command(&mut super::sqlite3_command(path), |output| {
        load_extension(output)?;
        write(output)
    })
}

fn execute_command(
    command: &mut Command,
    write: impl FnOnce(&mut BufWriter<ChildStdin>) -> Result<()>,
) -> Result<()> {
    let (child, mut input, _) = spawn(command, false)?;
    let output = input.0.as_mut().expect("stdin was installed at spawn");
    let written = write(output).and_then(|()| output.flush().map_err(Error::from));
    drop(input);
    written?;
    wait(&child)
}

fn spawn(
    command: &mut Command,
    capture: bool,
) -> Result<(ChildGuard, InputGuard, Option<ChildStdout>)> {
    let child = ChildGuard {
        child: Arc::new(Mutex::new(
            command
                .stdin(Stdio::piped())
                .stdout(if capture {
                    Stdio::piped()
                } else {
                    Stdio::null()
                })
                .stderr(Stdio::null())
                .spawn()?,
        )),
    };
    let mut process = child
        .child
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    let stdin = process.stdin.take().ok_or_else(|| {
        Error::new(
            ErrorCode::IoError,
            "failed to open sqlite3 stdin for benchmark baseline",
        )
    })?;
    let stdout = process.stdout.take();
    drop(process);
    Ok((
        child,
        InputGuard(Some(BufWriter::with_capacity(8192, stdin))),
        stdout,
    ))
}

fn wait(child: &ChildGuard) -> Result<()> {
    if !child
        .child
        .lock()
        .unwrap_or_else(|error| error.into_inner())
        .wait()?
        .success()
    {
        return Err(Error::new(
            ErrorCode::IoError,
            "sqlite3 benchmark baseline command failed",
        )
        .with_detail("reason", "sqlite_subprocess_failed"));
    }
    Ok(())
}

pub(super) fn query<T: Send, U>(
    path: &Path,
    write: impl FnOnce(&mut BufWriter<ChildStdin>) -> Result<T> + Send,
    read: impl FnOnce(&mut ChildStdout) -> Result<U>,
) -> Result<(T, U)> {
    query_command(
        &mut super::sqlite3_command(path),
        |output| {
            load_extension(output)?;
            write(output)
        },
        read,
    )
}

fn query_command<T: Send, U>(
    command: &mut Command,
    write: impl FnOnce(&mut BufWriter<ChildStdin>) -> Result<T> + Send,
    read: impl FnOnce(&mut ChildStdout) -> Result<U>,
) -> Result<(T, U)> {
    let (child, mut input, stdout) = spawn(command, true)?;
    let mut stdout =
        stdout.ok_or_else(|| Error::new(ErrorCode::IoError, "failed to open sqlite3 stdout"))?;
    std::thread::scope(move |scope| {
        // This guard must drop inside the scope, before a panicking reader joins the writer.
        let child = child;
        let process = Arc::clone(&child.child);
        let writer = std::thread::Builder::new()
            .name("sqlite-input".into())
            .spawn_scoped(scope, move || {
                let mut failure = KillOnFailure(Some(process));
                let output = input.0.as_mut().expect("stdin was installed at spawn");
                let result = write(output).and_then(|value| {
                    output.flush()?;
                    Ok(value)
                });
                drop(input);
                if result.is_ok() {
                    failure.0 = None;
                }
                result
            })?;
        let result = read(&mut stdout);
        drop(stdout);
        if result.is_err() {
            let _ = child
                .child
                .lock()
                .unwrap_or_else(|error| error.into_inner())
                .kill();
        }
        let written = match writer.join() {
            Ok(result) => result,
            Err(panic) => std::panic::resume_unwind(panic),
        };
        let result = result?;
        let written = written?;
        wait(&child)?;
        Ok((written, result))
    })
}

pub(super) fn script(path: &Path, sql: &str) -> Result<String> {
    query(
        path,
        |output| Ok(output.write_all(sql.as_bytes())?),
        |input| {
            let context = crate::json::JsonOperationContext::standalone();
            let mut work = context.reservation();
            let mut bytes = Vec::new();
            let mut buffer = [0; 8192];
            loop {
                let count = match input.read(&mut buffer) {
                    Err(error) if error.kind() == std::io::ErrorKind::Interrupted => continue,
                    result => result?,
                };
                if count == 0 {
                    break;
                }
                work.reserve_vec(
                    &mut bytes,
                    count,
                    crate::json::JsonInputKind::BenchmarkArtifact,
                )?;
                bytes.extend_from_slice(&buffer[..count]);
            }
            String::from_utf8(bytes).map_err(|_| {
                std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "sqlite3 output was not UTF-8",
                )
                .into()
            })
        },
    )
    .map(|(_, output)| output)
}

pub(super) fn visit_scores(
    input: impl Read,
    context: &crate::json::JsonOperationContext,
    mut visit: impl FnMut(String, f64) -> Result<()>,
) -> Result<()> {
    use crate::json::{JsonLines, JsonSchema, JsonValue};
    for row in JsonLines::new(input, JsonSchema::BenchmarkRecord, context, u64::MAX)? {
        let (line, mut row) = row?;
        let invalid = || {
            Error::new(ErrorCode::InvalidJsonlSchema, "invalid sqlite3 result row")
                .with_detail("line", line)
        };
        let score = row
            .value
            .get("score")
            .and_then(JsonValue::as_f64)
            .filter(|score| score.is_finite())
            .ok_or_else(invalid)?;
        let mut id = row
            .value
            .as_object_mut()
            .and_then(|row| row.remove("chunk_id"))
            .ok_or_else(invalid)?;
        let JsonValue::String(id) = &mut id else {
            return Err(invalid());
        };
        visit(std::mem::take(id), score)?;
    }
    Ok(())
}

pub(super) fn load_extension(output: &mut impl Write) -> Result<()> {
    if let Some(extension) = std::env::var_os("COFFERDB_SQLITE_VEC_EXTENSION") {
        write_load_extension(output, &extension)?;
    }
    Ok(())
}

fn write_load_extension(output: &mut impl Write, extension: &std::ffi::OsStr) -> Result<()> {
    let extension = extension
        .to_str()
        .filter(|value| !value.contains('\0'))
        .ok_or_else(|| {
            Error::new(
                ErrorCode::CliUsage,
                "sqlite-vec extension path must be UTF-8 without NUL bytes",
            )
        })?;
    output.write_all(b"SELECT load_extension(")?;
    write_string(output, extension)?;
    output.write_all(b");\n")?;
    Ok(())
}

pub(super) fn write_string(output: &mut impl Write, value: &str) -> std::io::Result<()> {
    if value.contains('\0') {
        output.write_all(b"CAST(X'")?;
        let mut buffer = [0; 4096];
        const HEX: &[u8; 16] = b"0123456789abcdef";
        for chunk in value.as_bytes().chunks(buffer.len() / 2) {
            for (value, pair) in chunk.iter().zip(buffer.chunks_exact_mut(2)) {
                pair[0] = HEX[(value >> 4) as usize];
                pair[1] = HEX[(value & 15) as usize];
            }
            output.write_all(&buffer[..chunk.len() * 2])?;
        }
        return output.write_all(b"' AS TEXT)");
    }
    output.write_all(b"'")?;
    for (index, part) in value.split('\'').enumerate() {
        if index != 0 {
            output.write_all(b"''")?;
        }
        for chunk in part.as_bytes().chunks(64 * 1024) {
            output.write_all(chunk)?;
        }
    }
    output.write_all(b"'")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(unix)]
    fn wait_for_pid(path: &Path) -> String {
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        loop {
            let pid = std::fs::read_to_string(path).unwrap();
            if pid.parse::<u32>().is_ok() {
                return pid;
            }
            assert!(std::time::Instant::now() < deadline, "child did not start");
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
    }

    #[cfg(unix)]
    fn child_is_running(pid: &str) -> bool {
        assert!(pid.parse::<u32>().expect("child PID must be numeric") > 0);
        // Use the shell builtin: minimal Unix images may not ship an external kill.
        let status = Command::new("sh")
            .args(["-c", "kill -0 \"$1\"", "cofferdb-child-probe", pid])
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .expect("child process probe must start");
        match status.code() {
            Some(0) => true,
            Some(1) => false,
            _ => panic!("unexpected child process probe status: {status}"),
        }
    }

    #[cfg(unix)]
    #[test]
    fn query_writes_and_reads_concurrently_and_rejects_nonzero_exit() {
        let mut command = Command::new("sh");
        command.args([
            "-c",
            "head -c 131072 /dev/zero; head -c 131072 /dev/zero >&2; exec cat",
        ]);
        let (written, bytes) = query_command(
            &mut command,
            |output| {
                for index in 0..256 {
                    output.write_all(&[index as u8; 4096])?;
                }
                Ok(256)
            },
            |input| {
                let mut bytes = Vec::new();
                input.read_to_end(&mut bytes)?;
                Ok(bytes)
            },
        )
        .unwrap();
        assert_eq!(written, 256);
        assert_eq!(bytes.len(), 131072 + 256 * 4096);
        assert!(bytes[..131072].iter().all(|byte| *byte == 0));
        for (index, chunk) in bytes[131072..].chunks_exact(4096).enumerate() {
            assert!(chunk.iter().all(|byte| *byte == index as u8));
        }
        let error = query_command(
            Command::new("sh").args(["-c", "printf 'partial'; exit 7"]),
            |_| Ok(()),
            |input| {
                let mut bytes = Vec::new();
                input.read_to_end(&mut bytes)?;
                Ok(bytes)
            },
        )
        .unwrap_err();
        assert_eq!(error.code(), ErrorCode::IoError);
        assert!(error
            .detail()
            .iter()
            .any(|(key, value)| key == "reason" && value == "sqlite_subprocess_failed"));
    }

    #[cfg(unix)]
    #[test]
    fn query_reaps_child_when_either_side_errors_or_panics() {
        for site in 0..4 {
            let (file, path) =
                super::super::create_temporary_file("sqlite-query-pid", "txt").unwrap();
            drop(file);
            let mut command = Command::new("sh");
            command
                .args([
                    "-c",
                    "printf '%s' \"$$\" > \"$PID_FILE\"; exec cat /dev/zero",
                ])
                .env("PID_FILE", path.path());
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                query_command(
                    &mut command,
                    |output| {
                        let pid = wait_for_pid(path.path());
                        if site < 2 {
                            assert!(child_is_running(&pid));
                        }
                        if site == 0 {
                            return Err(Error::new(ErrorCode::IoError, "producer failed"));
                        }
                        assert_ne!(site, 1, "producer panicked");
                        output.write_all(&vec![0; 1024 * 1024])?;
                        Ok(())
                    },
                    |input| {
                        let pid = wait_for_pid(path.path());
                        if site >= 2 {
                            assert!(child_is_running(&pid));
                        }
                        if site == 2 {
                            return Err(Error::new(ErrorCode::IoError, "consumer failed"));
                        }
                        assert_ne!(site, 3, "consumer panicked");
                        std::io::copy(input, &mut std::io::sink())?;
                        Ok(())
                    },
                )
            }));
            if site % 2 == 1 {
                assert!(result.is_err());
            } else {
                let error = result.unwrap().unwrap_err();
                assert!(error.to_string().contains(if site == 0 {
                    "producer failed"
                } else {
                    "consumer failed"
                }));
            }
            let pid = wait_for_pid(path.path());
            assert!(!child_is_running(&pid));
        }
    }

    #[test]
    fn score_rows_stream_under_shared_quota_and_preserve_candidate_ordering() {
        use crate::index::vector::VectorCandidates;
        use crate::json::{JsonInputKind, JsonOperationContext, JsonProcessingPolicyV1, JsonValue};
        for seed in 0..100 {
            let context = JsonOperationContext::standalone();
            let mut busy = context.reservation();
            busy.grow(
                JsonProcessingPolicyV1::WORK_BUDGET_BYTES - 16 * 1024,
                JsonInputKind::BenchmarkArtifact,
            )
            .unwrap();
            let mut bytes = Vec::new();
            let mut expected = Vec::new();
            for ordinal in 0..100 {
                let id = format!("c-{:03}'\n\t\0\u{4e2d}", (ordinal * 37 + seed) % 100);
                let score = ((ordinal + seed) % 11) as f64 / 10.0;
                let row = crate::json::object_from_pairs([
                    ("chunk_id", JsonValue::String(id.clone())),
                    ("score", JsonValue::Number(score)),
                ]);
                writeln!(bytes, "{}", row.to_compact_string()).unwrap();
                expected.push((id, score));
            }
            let limit = seed % 17;
            let mut candidates = VectorCandidates::new(limit, &context);
            visit_scores(bytes.as_slice(), &context, |chunk_id, score| {
                candidates.push(&chunk_id, score, score)
            })
            .unwrap();
            expected.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
            expected.truncate(limit);
            assert_eq!(
                candidates
                    .finish()
                    .iter()
                    .map(|hit| (hit.chunk_id.clone(), hit.normalized_score))
                    .collect::<Vec<_>>(),
                expected
            );
            busy.grow(16 * 1024, JsonInputKind::BenchmarkArtifact)
                .unwrap();
            assert!(visit_scores(bytes.as_slice(), &context, |_, _| Ok(())).is_err());
            drop(busy);
            visit_scores(bytes.as_slice(), &context, |_, _| Ok(())).unwrap();
            context
                .reservation()
                .grow(
                    JsonProcessingPolicyV1::WORK_BUDGET_BYTES,
                    JsonInputKind::BenchmarkArtifact,
                )
                .unwrap();
        }
        for bytes in [
            &b"{\"chunk_id\":\"x\",\"score\":1e400}\n"[..],
            b"{\"chunk_id\":\"x\",\"score\":null}\n",
            b"{\"chunk_id\":3,\"score\":1}\n",
            b"{\"chunk_id\":\"\xff\",\"score\":1}\n",
            b"{\"chunk_id\":\"x\",\"score\":1,\"score\":2}\n",
        ] {
            assert!(
                visit_scores(bytes, &JsonOperationContext::standalone(), |_, _| panic!(
                    "invalid row visited"
                ))
                .is_err()
            );
        }
    }

    #[test]
    fn extension_paths_reject_nul_and_invalid_utf8_before_writing() {
        let mut bytes = Vec::new();
        assert!(write_load_extension(&mut bytes, std::ffi::OsStr::new("prefix\0suffix")).is_err());
        assert!(bytes.is_empty());
        #[cfg(unix)]
        {
            use std::os::unix::ffi::OsStrExt;
            assert!(
                write_load_extension(&mut bytes, std::ffi::OsStr::from_bytes(b"\xff")).is_err()
            );
            assert!(bytes.is_empty());
        }
        write_load_extension(&mut bytes, std::ffi::OsStr::new("/valid ' \u{4e2d}.so")).unwrap();
        assert_eq!(
            String::from_utf8(bytes).unwrap(),
            "SELECT load_extension('/valid '' \u{4e2d}.so');\n"
        );
    }

    #[test]
    fn sql_literals_stream_bounded_fragments_and_roundtrip_utf8_and_nul() {
        struct Fragmented {
            bytes: Vec<u8>,
            calls: usize,
            limit: usize,
        }
        impl Write for Fragmented {
            fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
                assert!(bytes.len() <= 64 * 1024);
                self.calls += 1;
                if self.calls % 7 == 0 {
                    return Err(std::io::ErrorKind::Interrupted.into());
                }
                if self.bytes.len() == self.limit {
                    return Err(std::io::Error::other("injected literal error"));
                }
                let count = bytes.len().min(257).min(self.limit - self.bytes.len());
                self.bytes.extend_from_slice(&bytes[..count]);
                Ok(count)
            }
            fn flush(&mut self) -> std::io::Result<()> {
                Ok(())
            }
        }
        let sqlite = super::super::sqlite3_fts5_available();
        for seed in 0..100 {
            let value = format!(
                "{}'\n\t\u{4e2d}{}'",
                "x".repeat(if seed % 10 == 0 { 65_539 } else { seed * 3 }),
                if seed % 2 == 0 { "\0" } else { "" }
            );
            let mut output = Fragmented {
                bytes: Vec::new(),
                calls: 0,
                limit: usize::MAX,
            };
            write_string(&mut output, &value).unwrap();
            let sql = String::from_utf8(output.bytes).unwrap();
            if !value.contains('\0') {
                assert_eq!(sql, super::super::sql_quote(&value));
            }
            if sqlite {
                let actual = super::super::sqlite3_script(
                    Path::new(":memory:"),
                    &format!("SELECT hex({sql});"),
                )
                .unwrap();
                let expected = value
                    .as_bytes()
                    .iter()
                    .map(|byte| format!("{byte:02X}"))
                    .collect::<String>();
                assert_eq!(actual.trim(), expected);
            }
            for limit in [0, 1, 2, sql.len() - 1] {
                let mut output = Fragmented {
                    bytes: Vec::new(),
                    calls: 0,
                    limit,
                };
                assert!(write_string(&mut output, &value).is_err());
                assert_eq!(output.bytes, sql.as_bytes()[..limit]);
            }
        }
    }

    #[cfg(unix)]
    #[test]
    fn execute_drains_no_output_and_reports_write_and_exit_failures() {
        let (file, output) =
            super::super::create_temporary_file("sqlite-stream-output", "txt").unwrap();
        drop(file);
        let mut command = Command::new("sh");
        command
            .args([
                "-c",
                "head -c 131072 /dev/zero; head -c 131072 /dev/zero >&2; cat > \"$OUTPUT\"",
            ])
            .env("OUTPUT", output.path());
        execute_command(&mut command, |writer| {
            for _ in 0..256 {
                writer.write_all(&[37; 4096])?;
            }
            Ok(())
        })
        .unwrap();
        assert_eq!(std::fs::read(output.path()).unwrap(), vec![37; 1024 * 1024]);
        let error =
            execute_command(Command::new("sh").args(["-c", "exit 7"]), |_| Ok(())).unwrap_err();
        assert_eq!(error.code(), ErrorCode::IoError);
        assert!(error
            .detail()
            .iter()
            .any(|(key, value)| key == "reason" && value == "sqlite_subprocess_failed"));
        let error = execute_command(Command::new("sh").args(["-c", "exit 0"]), |writer| {
            writer.write_all(&vec![0; 1024 * 1024])?;
            Ok(())
        })
        .unwrap_err();
        assert_eq!(error.code(), ErrorCode::IoError);
    }

    #[cfg(unix)]
    #[test]
    fn execute_discards_buffered_sql_and_reaps_child_after_error_or_unwind() {
        for panic in [false, true] {
            let (file, output) =
                super::super::create_temporary_file("sqlite-discard-output", "txt").unwrap();
            let (pid_file, pid_path) =
                super::super::create_temporary_file("sqlite-child-pid", "txt").unwrap();
            drop((file, pid_file));
            let mut command = Command::new("sh");
            command
                .args([
                    "-c",
                    "printf '%s' \"$$\" > \"$PID_FILE\"; exec cat > \"$OUTPUT\"",
                ])
                .env("PID_FILE", pid_path.path())
                .env("OUTPUT", output.path());
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                execute_command(&mut command, |writer| {
                    let pid = wait_for_pid(pid_path.path());
                    assert!(child_is_running(&pid));
                    writer.write_all(b"uncommitted SQL")?;
                    if panic {
                        panic!("injected producer panic");
                    }
                    Err(Error::new(ErrorCode::IoError, "injected producer error"))
                })
            }));
            if panic {
                assert!(result.is_err());
            } else {
                assert!(result.unwrap().is_err());
            }
            let pid = std::fs::read_to_string(pid_path.path()).unwrap();
            assert!(!child_is_running(&pid));
            assert!(std::fs::read(output.path()).unwrap().is_empty());
        }
    }
}
