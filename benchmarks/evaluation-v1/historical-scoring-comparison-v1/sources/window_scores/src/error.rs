use std::fmt::{Display, Formatter};

use crate::json::{object_from_pairs, JsonValue};

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ErrorCode {
    IoError,
    FileFormatError,
    UnsupportedVersion,
    UnsupportedFeatureFlag,
    CollectionNotFound,
    CollectionAlreadyExists,
    InvalidEmbeddingDimension,
    EmbeddingMissing,
    InvalidMetadataFilter,
    InvalidJsonlSchema,
    MissingRequiredField,
    InvalidVectorLength,
    EmbeddingModelMismatch,
    UnsupportedMetric,
    DuplicateExternalId,
    CollectionDeletedNameReserved,
    LockContention,
    ChecksumMismatch,
    BenchmarkArtifactModeConflict,
    CorruptTail,
    IndexCorruption,
    RecoveryError,
    ResourceExhausted,
    PermissionDenied,
    InvalidSearchParams,
    InvalidSearchWeights,
    InvalidMetadataBoost,
    DocumentNotFound,
    ChunkNotFound,
    CliUsage,
}

impl ErrorCode {
    pub fn as_str(self) -> &'static str {
        match self {
            ErrorCode::IoError => "io_error",
            ErrorCode::FileFormatError => "file_format_error",
            ErrorCode::UnsupportedVersion => "unsupported_version",
            ErrorCode::UnsupportedFeatureFlag => "unsupported_feature_flag",
            ErrorCode::CollectionNotFound => "collection_not_found",
            ErrorCode::CollectionAlreadyExists => "collection_already_exists",
            ErrorCode::InvalidEmbeddingDimension => "invalid_embedding_dimension",
            ErrorCode::EmbeddingMissing => "embedding_missing",
            ErrorCode::InvalidMetadataFilter => "invalid_metadata_filter",
            ErrorCode::InvalidJsonlSchema => "invalid_jsonl_schema",
            ErrorCode::MissingRequiredField => "missing_required_field",
            ErrorCode::InvalidVectorLength => "invalid_vector_length",
            ErrorCode::EmbeddingModelMismatch => "embedding_model_mismatch",
            ErrorCode::UnsupportedMetric => "unsupported_metric",
            ErrorCode::DuplicateExternalId => "duplicate_external_id",
            ErrorCode::CollectionDeletedNameReserved => "collection_deleted_name_reserved",
            ErrorCode::LockContention => "lock_contention",
            ErrorCode::ChecksumMismatch => "checksum_mismatch",
            ErrorCode::BenchmarkArtifactModeConflict => "benchmark_artifact_mode_conflict",
            ErrorCode::CorruptTail => "corrupt_tail",
            ErrorCode::IndexCorruption => "index_corruption",
            ErrorCode::RecoveryError => "recovery_error",
            ErrorCode::ResourceExhausted => "resource_exhausted",
            ErrorCode::PermissionDenied => "permission_denied",
            ErrorCode::InvalidSearchParams => "invalid_search_params",
            ErrorCode::InvalidSearchWeights => "invalid_search_weights",
            ErrorCode::InvalidMetadataBoost => "invalid_metadata_boost",
            ErrorCode::DocumentNotFound => "document_not_found",
            ErrorCode::ChunkNotFound => "chunk_not_found",
            ErrorCode::CliUsage => "cli_usage",
        }
    }
}

#[derive(Debug)]
pub struct Error {
    code: ErrorCode,
    message: String,
    detail: Vec<(String, String)>,
    numeric_details: Vec<usize>,
    source: Option<std::io::Error>,
}

impl Error {
    pub(crate) fn resource_exhausted(
        resource: &'static str,
        operation: &'static str,
        limit_bytes: u64,
        requested_bytes: u128,
    ) -> Self {
        // Decimal strings keep budget diagnostics exact even when summing overflows u64.
        Self::new(
            ErrorCode::ResourceExhausted,
            "operation resource budget exhausted",
        )
        .with_detail("resource", resource)
        .with_detail("operation", operation)
        .with_detail("limit_bytes", limit_bytes)
        .with_detail("requested_bytes", requested_bytes)
    }

    pub fn new(code: ErrorCode, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
            detail: Vec::new(),
            numeric_details: Vec::new(),
            source: None,
        }
    }

    pub fn with_detail(mut self, key: impl Into<String>, value: impl ToString) -> Self {
        self.detail.push((key.into(), value.to_string()));
        self
    }

    pub(crate) fn with_u64_detail(mut self, key: impl Into<String>, value: u64) -> Self {
        // JsonValue stores numbers as f64; parsed integer diagnostics must round-trip exactly.
        debug_assert_eq!((value as f64) as u128, u128::from(value));
        self.numeric_details.push(self.detail.len());
        self.detail.push((key.into(), value.to_string()));
        self
    }

    pub fn code(&self) -> ErrorCode {
        self.code
    }

    pub fn code_str(&self) -> &'static str {
        self.code.as_str()
    }

    pub fn message(&self) -> &str {
        &self.message
    }

    pub fn detail(&self) -> &[(String, String)] {
        &self.detail
    }

    pub(crate) fn map_path_details(
        mut self,
        mut public_path: impl FnMut(&str) -> Option<String>,
    ) -> Self {
        let numeric = std::mem::take(&mut self.numeric_details);
        let mut original = 0;
        let mut retained = 0;
        self.detail.retain_mut(|(key, value)| {
            let keep = if matches!(key.as_str(), "path" | "manifest_path") {
                match public_path(value) {
                    Some(public) => {
                        *value = public;
                        true
                    }
                    None => false,
                }
            } else {
                true
            };
            if keep {
                if numeric.contains(&original) {
                    self.numeric_details.push(retained);
                }
                retained += 1;
            }
            original += 1;
            keep
        });
        self
    }

    pub fn to_json(&self) -> JsonValue {
        object_from_pairs([
            ("code", JsonValue::String(self.code_str().into())),
            ("message", JsonValue::String(self.message.clone())),
            ("recoverable", JsonValue::Bool(self.recoverable())),
            (
                "suggested_action",
                self.suggested_action()
                    .map(|action| JsonValue::String(action.into()))
                    .unwrap_or(JsonValue::Null),
            ),
            (
                "detail",
                object_from_pairs(self.detail.iter().enumerate().map(|(index, (key, value))| {
                    let value = if self.numeric_details.contains(&index) {
                        JsonValue::Number(value.parse::<u64>().expect("numeric detail") as f64)
                    } else {
                        JsonValue::String(value.clone())
                    };
                    (key.clone(), value)
                })),
            ),
        ])
    }

    fn recoverable(&self) -> bool {
        if crate::json::is_limit_error(self) {
            return self.code != ErrorCode::RecoveryError;
        }
        matches!(
            self.code,
            ErrorCode::DuplicateExternalId
                | ErrorCode::CollectionAlreadyExists
                | ErrorCode::CollectionDeletedNameReserved
                | ErrorCode::LockContention
                | ErrorCode::IndexCorruption
                | ErrorCode::ResourceExhausted
        )
    }

    fn suggested_action(&self) -> Option<&'static str> {
        if crate::json::is_limit_error(self) {
            return Some(if self.code == ErrorCode::RecoveryError {
                "preserve the original file and use a compatible version for explicit migration"
            } else {
                "reduce the JSON value or processing footprint and retry"
            });
        }
        match self.code {
            ErrorCode::DuplicateExternalId => {
                Some("rerun import with --on-conflict skip or replace")
            }
            ErrorCode::LockContention => Some("retry after the active writer finishes"),
            ErrorCode::IndexCorruption => {
                Some("run coffer rebuild-index for the affected collection")
            }
            ErrorCode::ResourceExhausted => {
                Some("release resources or reduce the operation footprint; verify committed progress before retrying writes")
            }
            _ => None,
        }
    }
}

impl Display for Error {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {}", self.code.as_str(), self.message)?;
        for (key, value) in &self.detail {
            write!(f, " ({key}={value})")?;
        }
        Ok(())
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        self.source
            .as_ref()
            .map(|err| err as &(dyn std::error::Error + 'static))
    }
}

impl From<std::io::Error> for Error {
    fn from(source: std::io::Error) -> Self {
        let code = if source.kind() == std::io::ErrorKind::PermissionDenied {
            ErrorCode::PermissionDenied
        } else {
            ErrorCode::IoError
        };
        Self {
            code,
            message: source.to_string(),
            detail: Vec::new(),
            numeric_details: Vec::new(),
            source: Some(source),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn path_redaction_preserves_core_fields_sources_and_numeric_detail_positions() {
        for seed in 0..100_u64 {
            let error = Error::from(std::io::Error::from(std::io::ErrorKind::PermissionDenied))
                .with_u64_detail("line", seed + 1)
                .with_detail("path", "/private/generated.json")
                .with_u64_detail("expected_dim", 768)
                .with_detail("manifest_path", "user/dataset.json")
                .with_detail("requested_bytes", u128::MAX)
                .with_u64_detail("actual_vector_length", seed);
            let before = error.to_json();
            let mapped = error
                .map_path_details(|path| (path == "user/dataset.json").then(|| path.to_owned()));
            let after = mapped.to_json();
            for field in ["code", "message", "recoverable", "suggested_action"] {
                assert_eq!(before.get(field), after.get(field));
            }
            let detail = after.get("detail").unwrap();
            for field in [
                "line",
                "expected_dim",
                "manifest_path",
                "requested_bytes",
                "actual_vector_length",
            ] {
                assert_eq!(before.get("detail").unwrap().get(field), detail.get(field));
            }
            assert!(detail.get("path").is_none());
            assert!(std::error::Error::source(&mapped).is_some());
            assert!(!after
                .to_compact_string()
                .contains("/private/generated.json"));
        }
    }

    #[test]
    fn resource_errors_preserve_exact_budget_details_without_claiming_corruption() {
        for limit in [0, 1, 4096, (1 << 53) + 1, u64::MAX] {
            for additional in 1..=100 {
                let requested = u128::from(limit) + additional;
                let error = Error::resource_exhausted(
                    "temporary_disk",
                    "temporary_staging",
                    limit,
                    requested,
                );
                assert_eq!(error.code(), ErrorCode::ResourceExhausted);
                assert!(!crate::json::is_limit_error(&error));
                let json = error.to_json();
                assert_eq!(
                    json.get("code").and_then(JsonValue::as_str),
                    Some("resource_exhausted")
                );
                assert_eq!(json.get("recoverable"), Some(&JsonValue::Bool(true)));
                assert_eq!(
                    json.get("suggested_action").and_then(JsonValue::as_str),
                    Some("release resources or reduce the operation footprint; verify committed progress before retrying writes")
                );
                let detail = json.get("detail").unwrap().as_object().unwrap();
                assert_eq!(detail.len(), 4);
                for (key, expected) in [
                    ("resource", "temporary_disk".to_owned()),
                    ("operation", "temporary_staging".to_owned()),
                    ("limit_bytes", limit.to_string()),
                    ("requested_bytes", requested.to_string()),
                ] {
                    assert_eq!(
                        detail.get(key).and_then(JsonValue::as_str),
                        Some(expected.as_str())
                    );
                }
                assert_eq!(crate::parse_json(&json.to_compact_string()).unwrap(), json);
            }
        }
    }

    #[test]
    fn numeric_details_keep_json_types_and_exactly_representable_large_integers() {
        let values = (0..100).chain([
            (1 << 53) - 1,
            1 << 53,
            (1 << 53) + 2,
            1 << 60,
            u64::MAX - 2047,
        ]);
        for value in values {
            let error = Error::new(ErrorCode::InvalidVectorLength, "test integer")
                .with_detail("query_id", "q")
                .with_u64_detail("row_dim", value)
                .with_detail("string_value", value);
            let wire = error.to_json().to_compact_string();
            let json = crate::parse_json(&wire).unwrap();
            let detail = json.get("detail").unwrap();
            assert_eq!(
                detail.get("row_dim").and_then(JsonValue::as_f64),
                Some(value as f64)
            );
            assert_eq!(
                detail.get("query_id").and_then(JsonValue::as_str),
                Some("q")
            );
            assert_eq!(
                detail.get("string_value").and_then(JsonValue::as_str),
                Some(value.to_string().as_str())
            );
            assert_eq!(error.detail()[1].1, value.to_string());
        }
    }

    #[test]
    fn error_json_is_machine_readable() {
        let err = Error::new(ErrorCode::DuplicateExternalId, "external_id already exists")
            .with_detail("collection", "notes")
            .with_detail("external_id", "doc-1");

        let json = err.to_json();
        assert_eq!(
            json.get("code").and_then(JsonValue::as_str),
            Some("duplicate_external_id")
        );
        assert_eq!(
            json.get("message").and_then(JsonValue::as_str),
            Some("external_id already exists")
        );
        assert_eq!(
            json.get("recoverable").and_then(JsonValue::as_bool),
            Some(true)
        );
        assert_eq!(
            json.get("suggested_action").and_then(JsonValue::as_str),
            Some("rerun import with --on-conflict skip or replace")
        );
        let detail = json.get("detail").and_then(JsonValue::as_object).unwrap();
        assert_eq!(
            detail.get("collection").and_then(JsonValue::as_str),
            Some("notes")
        );
        assert_eq!(
            detail.get("external_id").and_then(JsonValue::as_str),
            Some("doc-1")
        );
    }
}
