use std::fs;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use crate::json::{
    JsonInputKind, JsonOperationContext, JsonOutputSchema, JsonReservation, JsonValue,
};
use crate::temporary::create_temporary_file_in;
use crate::{Error, ErrorCode, Result};

pub(super) struct BenchmarkOutput {
    path: PathBuf,
    sources: Vec<PathBuf>,
}

fn destination_path(path: &Path) -> Result<PathBuf> {
    let name = path.file_name().ok_or_else(|| {
        Error::new(
            ErrorCode::IoError,
            "benchmark output must name a regular file",
        )
    })?;
    let parent = path
        .parent()
        .filter(|path| !path.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    Ok(fs::canonicalize(parent)?.join(name))
}

impl BenchmarkOutput {
    pub(super) fn new(path: &Path, database: &Path, mut inputs: Vec<PathBuf>) -> Result<Self> {
        inputs.push(database.to_path_buf());
        inputs.push(crate::lock::lock_path(database));
        let output = Self {
            path: destination_path(path)?,
            sources: inputs,
        };
        output.validate()?;
        Ok(output)
    }

    fn validate(&self) -> Result<()> {
        let output_metadata = match fs::metadata(&self.path) {
            Ok(metadata) => {
                if !metadata.is_file() {
                    return Err(Error::new(
                        ErrorCode::IoError,
                        "benchmark output is not a regular file",
                    ));
                }
                Some(metadata)
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                if fs::symlink_metadata(&self.path).is_ok() {
                    return Err(Error::new(
                        ErrorCode::PermissionDenied,
                        "benchmark output cannot be a dangling symlink",
                    ));
                }
                None
            }
            Err(error) => return Err(error.into()),
        };
        let output_path = if output_metadata.is_some() {
            fs::canonicalize(&self.path)?
        } else {
            self.path.clone()
        };
        for source in &self.sources {
            let source_path = match fs::canonicalize(source) {
                Ok(path) => path,
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                    destination_path(source)?
                }
                Err(error) => return Err(error.into()),
            };
            let same_file = output_path == source_path;
            #[cfg(unix)]
            let same_file = {
                use std::os::unix::fs::MetadataExt;
                same_file
                    || match (&output_metadata, fs::metadata(source)) {
                        (Some(output), Ok(source)) => {
                            output.dev() == source.dev() && output.ino() == source.ino()
                        }
                        _ => false,
                    }
            };
            if same_file {
                return Err(Error::new(
                    ErrorCode::PermissionDenied,
                    "benchmark output cannot replace the database, writer lock, or dataset inputs",
                ));
            }
        }
        Ok(())
    }

    pub(super) fn write(
        self,
        value: &JsonValue,
        context: &JsonOperationContext,
        retained: &mut JsonReservation,
    ) -> Result<()> {
        let mut buffer_work = context.reservation();
        buffer_work.grow(8192, JsonInputKind::BenchmarkArtifact)?;
        self.write_with(|file| {
            let mut output = BufWriter::with_capacity(8192, file);
            let result = value
                .write_pretty_retained_in(
                    &mut output,
                    JsonOutputSchema::Value(JsonInputKind::BenchmarkArtifact),
                    context,
                    retained,
                )
                .and_then(|()| output.flush().map_err(Error::from));
            // Do not retry buffered output after a failed serialization or flush.
            let _ = output.into_parts();
            result
        })
    }

    fn write_with(self, write: impl FnOnce(&mut fs::File) -> Result<()>) -> Result<()> {
        self.validate()?;
        let parent = self
            .path
            .parent()
            .expect("output path has a canonical parent");
        let (file, staged) = create_temporary_file_in(parent, "benchmark-result", "json")?;
        {
            // Close the handle before the cleanup guard removes it, including on unwind.
            let mut file = file;
            write(&mut file)?;
            file.sync_all()?;
        }
        self.validate()?;
        staged.persist(&self.path)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct TestDirectory(PathBuf);

    impl TestDirectory {
        fn new() -> Self {
            let (file, guard) =
                create_temporary_file_in(&std::env::temp_dir(), "benchmark-output-test", "tmp")
                    .unwrap();
            drop(file);
            let path = guard.path().with_extension("dir");
            fs::create_dir(&path).unwrap();
            Self(path)
        }
        fn files(&self) -> usize {
            fs::read_dir(&self.0).unwrap().count()
        }
    }

    impl Drop for TestDirectory {
        fn drop(&mut self) {
            fs::remove_dir_all(&self.0).unwrap();
        }
    }

    #[cfg(unix)]
    #[test]
    fn failed_writes_and_changed_destinations_preserve_sources_and_remove_staging() {
        let dir = TestDirectory::new();
        let source = dir.0.join("source.coffer");
        let destination = dir.0.join("result.json");
        fs::write(&source, b"source").unwrap();
        for stage in 0..5 {
            fs::write(&destination, b"old result").unwrap();
            let output = BenchmarkOutput::new(&destination, &source, Vec::new()).unwrap();
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                output.write_with(|file| {
                    file.write_all(b"partial output")?;
                    assert_eq!(fs::read(&destination).unwrap(), b"old result");
                    #[cfg(unix)]
                    {
                        use std::os::unix::fs::PermissionsExt;
                        assert_eq!(file.metadata()?.permissions().mode() & 0o777, 0o600);
                    }
                    match stage {
                        0 => Err(Error::new(ErrorCode::IoError, "injected write error")),
                        1 => panic!("injected write panic"),
                        2 => {
                            fs::remove_file(&destination)?;
                            fs::hard_link(&source, &destination)?;
                            Ok(())
                        }
                        3 => {
                            fs::remove_file(&destination)?;
                            fs::create_dir(&destination)?;
                            Ok(())
                        }
                        _ => {
                            file.write_all(b" complete")?;
                            Ok(())
                        }
                    }
                })
            }));
            if stage == 1 {
                assert!(result.is_err());
            } else {
                assert_eq!(result.unwrap().is_ok(), stage == 4);
            }
            assert_eq!(dir.files(), 2);
            assert_eq!(fs::read(&source).unwrap(), b"source");
            match stage {
                2 => {
                    assert_eq!(fs::read(&destination).unwrap(), b"source");
                    fs::remove_file(&destination).unwrap();
                }
                3 => fs::remove_dir(&destination).unwrap(),
                4 => assert_eq!(fs::read(&destination).unwrap(), b"partial output complete"),
                _ => assert_eq!(fs::read(&destination).unwrap(), b"old result"),
            }
        }
    }

    #[test]
    fn streamed_report_reuses_budget_and_does_not_limit_the_whole_report_to_eight_mib() {
        use crate::json::JsonProcessingPolicyV1;
        let dir = TestDirectory::new();
        let output = dir.0.join("result.json");
        let database = dir.0.join("database.coffer");
        let context = JsonOperationContext::standalone();
        let value = JsonValue::Array(
            (0..100)
                .map(|_| JsonValue::String("a".repeat(90_000)))
                .collect(),
        );
        let schema = JsonOutputSchema::Value(JsonInputKind::BenchmarkArtifact);
        let mut retained = context.reservation();
        value
            .reconcile_output_retained_in(schema, &mut retained)
            .unwrap();
        let mut busy = context.reservation();
        busy.grow(
            JsonProcessingPolicyV1::WORK_BUDGET_BYTES - retained.reserved_bytes() - 16 * 1024,
            JsonInputKind::BenchmarkArtifact,
        )
        .unwrap();
        assert!(value
            .pretty_retained_in(schema, &context, &mut retained)
            .is_err());
        BenchmarkOutput::new(&output, &database, Vec::new())
            .unwrap()
            .write(&value, &context, &mut retained)
            .unwrap();
        assert_eq!(
            fs::read_to_string(&output).unwrap(),
            format!("{}\n", value.to_pretty_string())
        );
        assert!(fs::metadata(&output).unwrap().len() > 8 * 1024 * 1024);
        assert_eq!(dir.files(), 1);
        busy.grow(16 * 1024, JsonInputKind::BenchmarkArtifact)
            .unwrap();
        let old = fs::read(&output).unwrap();
        assert!(BenchmarkOutput::new(&output, &database, Vec::new())
            .unwrap()
            .write(&value, &context, &mut retained)
            .is_err());
        assert_eq!(fs::read(&output).unwrap(), old);
        assert_eq!(dir.files(), 1);
        drop((value, retained, busy));
        context
            .reservation()
            .grow(
                JsonProcessingPolicyV1::WORK_BUDGET_BYTES,
                JsonInputKind::BenchmarkArtifact,
            )
            .unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn aliases_directories_and_dangling_symlinks_are_rejected() {
        use std::os::unix::fs::symlink;
        let dir = TestDirectory::new();
        let source = dir.0.join("database.coffer");
        assert_eq!(
            BenchmarkOutput::new(&source, &source, Vec::new())
                .err()
                .unwrap()
                .code(),
            ErrorCode::PermissionDenied
        );
        fs::write(&source, b"original").unwrap();
        let hard = dir.0.join("hard.json");
        let link = dir.0.join("link.json");
        let dangling = dir.0.join("dangling.json");
        fs::hard_link(&source, &hard).unwrap();
        symlink(&source, &link).unwrap();
        symlink(dir.0.join("missing"), &dangling).unwrap();
        for path in [&source, &hard, &link, &dangling] {
            assert_eq!(
                BenchmarkOutput::new(path, &source, Vec::new())
                    .err()
                    .unwrap()
                    .code(),
                ErrorCode::PermissionDenied
            );
        }
        assert_eq!(
            BenchmarkOutput::new(&dir.0, &source, Vec::new())
                .err()
                .unwrap()
                .code(),
            ErrorCode::IoError
        );
        assert_eq!(fs::read(&source).unwrap(), b"original");
    }
}
