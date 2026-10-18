use std::fs;
use std::path::PathBuf;
use std::process::{Command, Output};
use std::sync::atomic::{AtomicU64, Ordering};

use cofferdb::{parse_json, JsonValue};

const CURRENT: &str = r#"
use cofferdb::{DatabaseConfig, ResourceProfile};
use cofferdb::ResourceProfile::{Expanded, Balanced, Constrained};
fn main() {
    for profile in [Expanded, Balanced, Constrained] {
        let config = DatabaseConfig { resource_profile: profile, ..Default::default() };
        assert_eq!(config.resource_profile, profile);
    }
    let _ = [ResourceProfile::Expanded, ResourceProfile::Balanced, ResourceProfile::Constrained];
}
"#;

struct Consumer(PathBuf);

impl Consumer {
    fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let path = loop {
            let path = std::env::temp_dir().join(format!(
                "cofferdb-profile-consumer-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            match fs::create_dir(&path) {
                Ok(()) => break path,
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(error) => panic!("cannot create consumer directory: {error}"),
            }
        };
        let consumer = Self(path);
        fs::create_dir(consumer.0.join("src")).unwrap();
        let dependency = JsonValue::String(env!("CARGO_MANIFEST_DIR").into()).to_compact_string();
        fs::write(
            consumer.0.join("Cargo.toml"),
            format!(
                "[package]\nname = \"profile_consumer\"\nversion = \"0.0.0\"\nedition = \"2021\"\n\
                 [workspace]\n[dependencies]\ncofferdb = {{ path = {dependency} }}\n"
            ),
        )
        .unwrap();
        fs::write(consumer.0.join("src/main.rs"), CURRENT).unwrap();
        let output = consumer
            .cargo()
            .args(["generate-lockfile", "--offline"])
            .output()
            .unwrap();
        assert_success(&output, "generate the isolated consumer lockfile");
        consumer
    }

    fn cargo(&self) -> Command {
        let mut command = Command::new(std::env::var_os("CARGO").unwrap_or_else(|| "cargo".into()));
        command.current_dir(&self.0);
        // A separate target directory avoids nesting a check under the parent Cargo lock.
        command.env("CARGO_TARGET_DIR", self.0.join("target"));
        command
    }

    fn check(&self, source: &str) -> Output {
        fs::write(self.0.join("src/main.rs"), source).unwrap();
        self.cargo()
            .args(["check", "--offline", "--locked", "--message-format=json"])
            .output()
            .unwrap()
    }
}

impl Drop for Consumer {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn assert_success(output: &Output, label: &str) {
    assert!(
        output.status.success(),
        "{label}:\n{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}

fn assert_absent(form: &str, expected_code: &str) {
    let consumer = Consumer::new();
    assert_success(
        &consumer.check(CURRENT),
        "current names before negative controls",
    );
    for name in ["QualityFirst", "ResourceFirst"] {
        let source = form.replace("LEGACY", name);
        let output = consumer.check(&source);
        assert!(
            !output.status.success(),
            "legacy consumer compiled: {source}"
        );
        let diagnostics = std::str::from_utf8(&output.stdout).unwrap();
        let matched = diagnostics
            .lines()
            .filter_map(|line| parse_json(line).ok())
            .any(|value| {
                if value.get("reason").and_then(JsonValue::as_str) != Some("compiler-message") {
                    return false;
                }
                let Some(message) = value.get("message") else {
                    return false;
                };
                message.get("level").and_then(JsonValue::as_str) == Some("error")
                    && message
                        .get("code")
                        .and_then(|code| code.get("code"))
                        .and_then(JsonValue::as_str)
                        == Some(expected_code)
                    && message
                        .get("message")
                        .and_then(JsonValue::as_str)
                        .is_some_and(|text| text.contains(name))
                    && message
                        .get("spans")
                        .and_then(JsonValue::as_array)
                        .is_some_and(|spans| {
                            spans.iter().any(|span| {
                                span.get("file_name").and_then(JsonValue::as_str)
                                    == Some("src/main.rs")
                            })
                        })
            });
        assert!(
            matched,
            "expected {expected_code} for absent {name}:\n{diagnostics}\n{}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
    assert_success(
        &consumer.check(CURRENT),
        "current names after negative controls",
    );
}

#[test]
fn legacy_profile_variants_or_associated_constants_fail_in_independent_consumers() {
    assert_absent(
        "fn main() { let _ = cofferdb::ResourceProfile::LEGACY; }",
        "E0599",
    );
}

#[test]
fn legacy_profile_variant_imports_fail_in_independent_consumers() {
    assert_absent(
        "use cofferdb::ResourceProfile::LEGACY; fn main() { let _ = LEGACY; }",
        "E0432",
    );
}

#[test]
fn legacy_profile_root_reexports_fail_in_independent_consumers() {
    assert_absent(
        "use cofferdb::LEGACY; fn main() { let _ = LEGACY; }",
        "E0432",
    );
}
