use crate::error::{Error, ErrorCode};

pub(crate) struct JsonProcessingPolicyV1;

impl JsonProcessingPolicyV1 {
    pub(crate) const MAX_CONTAINER_DEPTH: usize = 128;
    pub(crate) const MAX_UNIT_CANONICAL_BYTES: usize = 8 * 1024 * 1024;
    pub(crate) const WORK_BUDGET_BYTES: usize = 32 * 1024 * 1024;
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum JsonInputKind {
    Manifest,
    CollectionConfig,
    RuntimeConfig,
    DocumentMetadata,
    ChunkMetadata,
    Filter,
    Weights,
    MetadataBoost,
    QueryEmbedding,
    JsonlRecord,
    BenchmarkArtifact,
    StoredJson,
}

impl JsonInputKind {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Manifest => "manifest",
            Self::CollectionConfig => "collection_config",
            Self::RuntimeConfig => "runtime_config",
            Self::DocumentMetadata => "document_metadata",
            Self::ChunkMetadata => "chunk_metadata",
            Self::Filter => "filter",
            Self::Weights => "weights",
            Self::MetadataBoost => "metadata_boost",
            Self::QueryEmbedding => "query_embedding",
            Self::JsonlRecord => "jsonl_record",
            Self::BenchmarkArtifact => "benchmark_artifact",
            Self::StoredJson => "stored_json",
        }
    }

    pub(crate) fn error_code(self) -> ErrorCode {
        match self {
            Self::Filter => ErrorCode::InvalidMetadataFilter,
            Self::Weights => ErrorCode::InvalidSearchWeights,
            Self::MetadataBoost => ErrorCode::InvalidMetadataBoost,
            Self::StoredJson => ErrorCode::RecoveryError,
            _ => ErrorCode::InvalidJsonlSchema,
        }
    }

    pub(crate) fn limit_error(
        self,
        reason: JsonLimitReason,
        limit: usize,
        actual: Option<usize>,
    ) -> Error {
        let mut error = Error::new(self.error_code(), reason.message())
            .with_detail("json_limit_reason", reason.as_str())
            .with_detail("json_input_kind", self.as_str())
            .with_u64_detail("limit", limit as u64)
            .with_detail("limit_unit", reason.unit());
        if let Some(actual) = actual.filter(|actual| (*actual as u128) <= (1u128 << 53)) {
            error = error.with_u64_detail("actual", actual as u64);
        }
        match self {
            Self::Weights => error.with_detail("weights_reason", "json_processing_limit"),
            Self::MetadataBoost => {
                error.with_detail("metadata_boost_reason", "json_processing_limit")
            }
            _ => error,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) enum JsonLimitReason {
    Depth,
    UnitBytes,
    WorkBudget,
}

impl JsonLimitReason {
    fn as_str(self) -> &'static str {
        match self {
            Self::Depth => "depth_exceeded",
            Self::UnitBytes => "unit_bytes_exceeded",
            Self::WorkBudget => "work_budget_exceeded",
        }
    }

    fn message(self) -> &'static str {
        match self {
            Self::Depth => "JSON container depth exceeds the processing limit",
            Self::UnitBytes => "JSON canonical bytes exceed the processing limit",
            Self::WorkBudget => "JSON processing exceeds the available work budget",
        }
    }

    fn unit(self) -> &'static str {
        match self {
            Self::Depth => "container_levels",
            _ => "bytes",
        }
    }
}

pub(crate) fn is_limit_error(error: &Error) -> bool {
    error.detail().iter().any(|(key, value)| {
        key == "json_limit_reason"
            && matches!(
                value.as_str(),
                "depth_exceeded" | "unit_bytes_exceeded" | "work_budget_exceeded"
            )
    })
}

pub(crate) fn is_memory_budget_error(error: &Error) -> bool {
    (error.code() == ErrorCode::ResourceExhausted
        && error
            .detail()
            .iter()
            .any(|(key, value)| key == "resource" && value == "memory"))
        || error
            .detail()
            .iter()
            .any(|(key, value)| key == "json_limit_reason" && value == "work_budget_exceeded")
}

// The attempt must be safe to repeat before IO, retaining any live input credit.
pub(crate) fn retry_after_memory_reclaim<T>(
    mut attempt: impl FnMut() -> crate::Result<T>,
    reclaim: impl FnOnce() -> bool,
) -> crate::Result<T> {
    match attempt() {
        Err(error) if is_memory_budget_error(&error) && reclaim() => attempt(),
        result => result,
    }
}

#[derive(Clone, Copy)]
pub(crate) enum JsonSchema {
    Value(JsonInputKind),
    DocumentRow,
    ChunkRow,
    EmbeddingRow,
    QueryRow,
    QueryEmbeddingObject,
    BenchmarkRecord,
    StoredRecord,
    RecordField(JsonInputKind),
}

impl JsonSchema {
    pub(crate) fn kind(self) -> JsonInputKind {
        match self {
            Self::Value(kind) | Self::RecordField(kind) => kind,
            Self::DocumentRow | Self::ChunkRow | Self::EmbeddingRow => JsonInputKind::JsonlRecord,
            Self::QueryRow | Self::BenchmarkRecord => JsonInputKind::BenchmarkArtifact,
            Self::QueryEmbeddingObject => JsonInputKind::QueryEmbedding,
            Self::StoredRecord => JsonInputKind::StoredJson,
        }
    }

    pub(crate) fn is_record(self) -> bool {
        !matches!(self, Self::Value(_) | Self::RecordField(_))
    }

    pub(crate) fn field(self, key: &str) -> Self {
        let kind = match (self, key) {
            (Self::DocumentRow, "metadata") => JsonInputKind::DocumentMetadata,
            (Self::ChunkRow, "metadata") => JsonInputKind::ChunkMetadata,
            (Self::QueryRow, "filter") => JsonInputKind::Filter,
            (Self::QueryRow, "weights") => JsonInputKind::Weights,
            (Self::QueryRow, "metadata_boost") => JsonInputKind::MetadataBoost,
            (Self::QueryRow, "embedding") | (Self::QueryEmbeddingObject, "vector") => {
                JsonInputKind::QueryEmbedding
            }
            (Self::StoredRecord, "metadata" | "config" | "tokenizer") => JsonInputKind::StoredJson,
            (Self::Value(_) | Self::RecordField(_), _) => return self,
            _ => return Self::RecordField(self.kind()),
        };
        Self::Value(kind)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::JsonValue;

    #[test]
    fn memory_reclaim_attempts_are_bounded_and_preserve_errors() {
        for resource in ["memory", "catalog", "temporary_disk"] {
            for released in [false, true] {
                let mut attempts = 0;
                let mut reclaims = 0;
                let error = retry_after_memory_reclaim(
                    || {
                        attempts += 1;
                        Err::<(), _>(Error::resource_exhausted(
                            resource,
                            "batch_prepare",
                            100,
                            100 + attempts,
                        ))
                    },
                    || {
                        reclaims += 1;
                        released
                    },
                )
                .unwrap_err();
                assert_eq!(attempts, 1 + u128::from(resource == "memory" && released));
                assert_eq!(reclaims, usize::from(resource == "memory"));
                let expected =
                    Error::resource_exhausted(resource, "batch_prepare", 100, 100 + attempts);
                assert_eq!(error.code(), expected.code());
                assert_eq!(error.detail(), expected.detail());
            }
        }
        let value = retry_after_memory_reclaim(|| Ok(42), || panic!("success must retain cache"));
        assert_eq!(value.unwrap(), 42);
        for error in [
            Error::new(ErrorCode::IoError, "original IO failure"),
            JsonInputKind::DocumentMetadata.limit_error(JsonLimitReason::Depth, 128, Some(129)),
        ] {
            let expected = error.to_json();
            let mut error = Some(error);
            let actual = retry_after_memory_reclaim(
                || Err::<(), _>(error.take().expect("must not retry")),
                || panic!("must not reclaim"),
            )
            .unwrap_err();
            assert_eq!(actual.to_json(), expected);
        }
    }

    #[test]
    fn memory_reclaim_does_not_retry_other_resource_or_storage_failures() {
        for resource in ["memory", "catalog", "temporary_disk"] {
            let error = Error::resource_exhausted(resource, "catalog_build", 100, 101);
            assert_eq!(is_memory_budget_error(&error), resource == "memory");
        }
        for code in [
            ErrorCode::IoError,
            ErrorCode::ChecksumMismatch,
            ErrorCode::RecoveryError,
            ErrorCode::FileFormatError,
        ] {
            assert!(!is_memory_budget_error(&Error::new(
                code,
                "not a budget rejection"
            )));
        }
    }

    #[test]
    fn all_json_input_kinds_have_typed_limit_details_and_domain_codes() {
        for (kind, name, code) in [
            (JsonInputKind::Manifest, "manifest", "invalid_jsonl_schema"),
            (
                JsonInputKind::CollectionConfig,
                "collection_config",
                "invalid_jsonl_schema",
            ),
            (
                JsonInputKind::RuntimeConfig,
                "runtime_config",
                "invalid_jsonl_schema",
            ),
            (
                JsonInputKind::DocumentMetadata,
                "document_metadata",
                "invalid_jsonl_schema",
            ),
            (
                JsonInputKind::ChunkMetadata,
                "chunk_metadata",
                "invalid_jsonl_schema",
            ),
            (JsonInputKind::Filter, "filter", "invalid_metadata_filter"),
            (JsonInputKind::Weights, "weights", "invalid_search_weights"),
            (
                JsonInputKind::MetadataBoost,
                "metadata_boost",
                "invalid_metadata_boost",
            ),
            (
                JsonInputKind::QueryEmbedding,
                "query_embedding",
                "invalid_jsonl_schema",
            ),
            (
                JsonInputKind::JsonlRecord,
                "jsonl_record",
                "invalid_jsonl_schema",
            ),
            (
                JsonInputKind::BenchmarkArtifact,
                "benchmark_artifact",
                "invalid_jsonl_schema",
            ),
            (JsonInputKind::StoredJson, "stored_json", "recovery_error"),
        ] {
            for (reason, limit, unit) in [
                (JsonLimitReason::Depth, 128, "container_levels"),
                (
                    JsonLimitReason::UnitBytes,
                    JsonProcessingPolicyV1::MAX_UNIT_CANONICAL_BYTES,
                    "bytes",
                ),
                (JsonLimitReason::WorkBudget, 4096, "bytes"),
            ] {
                let error = kind.limit_error(reason, limit, Some(limit + 1));
                assert!(is_limit_error(&error));
                assert_eq!(
                    is_memory_budget_error(&error),
                    matches!(reason, JsonLimitReason::WorkBudget)
                );
                assert_eq!(error.code_str(), code);
                let json = error.to_json();
                crate::test_error_contract::validate_error(&json).unwrap();
                crate::test_error_contract::validate_error(
                    &kind.limit_error(reason, limit, None).to_json(),
                )
                .unwrap();
                let detail = json.get("detail").unwrap();
                assert_eq!(detail.get("limit"), Some(&JsonValue::Number(limit as f64)));
                assert_eq!(
                    detail.get("actual"),
                    Some(&JsonValue::Number((limit + 1) as f64))
                );
                assert_eq!(
                    detail.get("json_input_kind").and_then(JsonValue::as_str),
                    Some(name)
                );
                assert_eq!(
                    detail.get("limit_unit").and_then(JsonValue::as_str),
                    Some(unit)
                );
                let reason_field = match kind {
                    JsonInputKind::Weights => Some("weights_reason"),
                    JsonInputKind::MetadataBoost => Some("metadata_boost_reason"),
                    _ => None,
                };
                if let Some(field) = reason_field {
                    assert_eq!(
                        detail.get(field).and_then(JsonValue::as_str),
                        Some("json_processing_limit")
                    );
                }
            }
        }
    }
}
