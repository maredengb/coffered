use super::{
    json_error, shallow_canonical_bytes, work, JsonInputKind, JsonLimitReason, JsonObject,
    JsonOperationContext, JsonProcessingPolicyV1, JsonValue, ProcessedJson,
};
use crate::Result;

enum Frame<'a> {
    Array {
        source: std::slice::Iter<'a, JsonValue>,
        values: Vec<JsonValue>,
    },
    Object {
        source: std::collections::btree_map::Iter<'a, String, JsonValue>,
        values: JsonObject,
        key: Option<String>,
    },
}

impl<'a> Frame<'a> {
    fn next(&mut self) -> Result<Option<&'a JsonValue>> {
        match self {
            Self::Array { source, .. } => Ok(source.next()),
            Self::Object { source, key, .. } => {
                let Some((name, value)) = source.next() else {
                    return Ok(None);
                };
                *key = Some(work::copy_precharged_string(name)?);
                Ok(Some(value))
            }
        }
    }

    fn push(&mut self, value: JsonValue) {
        match self {
            Self::Array { values, .. } => values.push(value),
            Self::Object { values, key, .. } => {
                values.insert(key.take().unwrap(), value);
            }
        }
    }

    fn finish(self) -> JsonValue {
        match self {
            Self::Array { values, .. } => JsonValue::Array(values),
            Self::Object { values, .. } => JsonValue::Object(values),
        }
    }
}

impl JsonValue {
    pub(crate) fn try_clone_in(
        &self,
        kind: JsonInputKind,
        context: &JsonOperationContext,
    ) -> Result<ProcessedJson> {
        self.clone_in(kind, context, true)
    }

    pub(super) fn clone_output_in(
        &self,
        kind: JsonInputKind,
        context: &JsonOperationContext,
    ) -> Result<ProcessedJson> {
        self.clone_in(kind, context, false)
    }

    fn clone_in(
        &self,
        kind: JsonInputKind,
        context: &JsonOperationContext,
        business_root: bool,
    ) -> Result<ProcessedJson> {
        let mut work = context.aggregate_reservation();
        let mut frames = Vec::<Frame<'_>>::new();
        let mut next = Some(self);
        let mut ready = None;
        let mut canonical_bytes = 0usize;
        loop {
            if let Some(value) = next.take() {
                if matches!(value, Self::Number(number) if !number.is_finite()) {
                    return Err(json_error("JSON numbers must be finite"));
                }
                let container = matches!(value, Self::Array(_) | Self::Object(_));
                if business_root
                    && container
                    && frames.len() >= JsonProcessingPolicyV1::MAX_CONTAINER_DEPTH
                {
                    return Err(kind.limit_error(
                        JsonLimitReason::Depth,
                        JsonProcessingPolicyV1::MAX_CONTAINER_DEPTH,
                        Some(frames.len() + 1),
                    ));
                }
                if business_root {
                    canonical_bytes =
                        canonical_bytes.saturating_add(shallow_canonical_bytes(value));
                }
                if business_root
                    && canonical_bytes > JsonProcessingPolicyV1::MAX_UNIT_CANONICAL_BYTES
                {
                    return Err(kind.limit_error(
                        JsonLimitReason::UnitBytes,
                        JsonProcessingPolicyV1::MAX_UNIT_CANONICAL_BYTES,
                        Some(canonical_bytes),
                    ));
                }
                // Only active ancestors occupy the traversal stack, even for wide arrays.
                if container {
                    work.reserve_vec(&mut frames, 1, kind)?;
                }
                match value {
                    Self::Array(source) => {
                        let mut values = Vec::new();
                        work.reserve_vec(&mut values, source.len(), kind)?;
                        frames.push(Frame::Array {
                            source: source.iter(),
                            values,
                        });
                    }
                    Self::Object(source) => {
                        let keys_work =
                            work.reserve_object_keys(source.keys().map(String::as_str), kind)?;
                        work.merge(keys_work);
                        frames.push(Frame::Object {
                            source: source.iter(),
                            values: JsonObject::new(),
                            key: None,
                        });
                    }
                    Self::Null => ready = Some(Self::Null),
                    Self::Bool(value) => ready = Some(Self::Bool(*value)),
                    Self::Number(value) => ready = Some(Self::Number(*value)),
                    Self::String(value) => {
                        ready = Some(Self::String(work.copy_string(value, kind)?))
                    }
                }
            }
            if let Some(value) = ready.take() {
                if let Some(parent) = frames.last_mut() {
                    parent.push(value);
                } else {
                    let frame_bytes = frames.capacity() * std::mem::size_of::<Frame<'_>>();
                    drop(frames);
                    work.release(frame_bytes);
                    work.finish_aggregate();
                    return Ok(ProcessedJson {
                        value,
                        _reservation: work,
                    });
                }
            }
            let frame = frames.last_mut().unwrap();
            if let Some(value) = frame.next()? {
                next = Some(value);
            } else {
                ready = Some(frames.pop().unwrap().finish());
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::json::object_from_pairs;

    #[test]
    fn object_copy_key_failure_releases_only_the_copy_stage() {
        let context = JsonOperationContext::standalone();
        let first = "a".repeat(4097);
        let second = "b".repeat(8193);
        let value = object_from_pairs([
            (first.as_str(), JsonValue::String("payload".into())),
            (second.as_str(), JsonValue::Bool(true)),
        ]);
        let kind = JsonInputKind::StoredJson;
        let input = value.reserve_retained_in(kind, &context).unwrap();
        let mut existing = context.reservation();
        existing.grow(17, kind).unwrap();
        let baseline = context.available_workspace_bytes();
        let pointer = value.get(&first).unwrap().as_str().unwrap().as_ptr();
        let error = crate::test_alloc::fail_byte_allocation_once(second.len(), || {
            value.try_clone_in(kind, &context)
        })
        .err()
        .unwrap();
        assert_eq!(error.code(), crate::ErrorCode::IoError);
        assert_eq!(context.available_workspace_bytes(), baseline);
        assert_eq!(existing.reserved_bytes(), 17);
        assert_eq!(
            value.get(&first).unwrap().as_str().unwrap().as_ptr(),
            pointer
        );
        let (copy, copy_work) = value.try_clone_in(kind, &context).unwrap().into_parts();
        assert_eq!(copy, value);
        let measured = copy.reserve_retained_in(kind, &context).unwrap();
        assert_eq!(copy_work.reserved_bytes(), measured.reserved_bytes());
        drop(measured);
        drop(copy);
        drop(copy_work);
        assert_eq!(context.available_workspace_bytes(), baseline);
        drop(value);
        drop((input, existing));
        assert_eq!(
            context.available_workspace_bytes(),
            JsonProcessingPolicyV1::WORK_BUDGET_BYTES
        );
    }

    #[test]
    fn copies_match_canonical_values_and_keep_exact_retained_reservations() {
        let context = JsonOperationContext::standalone();
        for seed in 0..100 {
            let mut value = object_from_pairs([
                (
                    "escaped",
                    JsonValue::String(format!("\"\\\n\u{7f}\u{1f600}{seed}")),
                ),
                ("number", JsonValue::Number((seed as f64 - 50.0) / 7.0)),
                ("empty", JsonValue::Object(JsonObject::new())),
            ]);
            for depth in 0..seed % 12 {
                value = if depth % 2 == 0 {
                    JsonValue::Array(vec![JsonValue::Null, value, JsonValue::Bool(true)])
                } else {
                    object_from_pairs([("nested", value)])
                };
            }
            let (_input, _) = value.guard_in(JsonInputKind::Filter, &context).unwrap();
            let (copy, lease) = value
                .try_clone_in(JsonInputKind::Filter, &context)
                .unwrap()
                .into_parts();
            assert_eq!(copy, value);
            assert_eq!(copy.to_compact_string(), value.to_compact_string());
            let (measured, _) = copy.guard_in(JsonInputKind::Filter, &context).unwrap();
            assert_eq!(measured.reserved_bytes(), lease.reserved_bytes());
        }
    }

    #[test]
    fn depth_rejection_and_cleanup_use_a_small_stack() {
        std::thread::Builder::new()
            .stack_size(64 * 1024)
            .spawn(|| {
                let context = JsonOperationContext::standalone();
                let mut value = JsonValue::Null;
                for _ in 0..128 {
                    value = JsonValue::Array(vec![value]);
                }
                let copied = value
                    .try_clone_in(JsonInputKind::StoredJson, &context)
                    .unwrap();
                drop(copied);
                for _ in 128..30_000 {
                    value = JsonValue::Array(vec![value]);
                }
                let error = value
                    .try_clone_in(JsonInputKind::StoredJson, &context)
                    .err()
                    .unwrap();
                assert_eq!(error.code(), crate::ErrorCode::RecoveryError);
                assert!(error
                    .detail()
                    .iter()
                    .any(|(key, value)| key == "json_limit_reason" && value == "depth_exceeded"));
                drop(value);
            })
            .unwrap()
            .join()
            .unwrap();
    }

    #[test]
    fn copies_share_input_budget_and_release_failed_allocations() {
        let context = JsonOperationContext::standalone();
        let value = JsonValue::Array(vec![JsonValue::Null; 600_000]);
        let (input, _) = value.guard_in(JsonInputKind::Filter, &context).unwrap();
        let error = value
            .try_clone_in(JsonInputKind::Filter, &context)
            .err()
            .unwrap();
        assert!(error
            .detail()
            .iter()
            .any(|(key, value)| key == "json_limit_reason" && value == "work_budget_exceeded"));
        drop(input);
        let (copy, lease) = value
            .try_clone_in(JsonInputKind::Filter, &context)
            .unwrap()
            .into_parts();
        assert_eq!(copy.as_array().unwrap().len(), 600_000);
        drop(copy);
        drop(lease);
        let mut all = context.reservation();
        all.grow(
            JsonProcessingPolicyV1::WORK_BUDGET_BYTES,
            JsonInputKind::Filter,
        )
        .unwrap();
    }

    #[test]
    fn unit_and_number_validation_precede_string_copy() {
        let context = JsonOperationContext::standalone();
        let exact =
            JsonValue::String("x".repeat(JsonProcessingPolicyV1::MAX_UNIT_CANONICAL_BYTES - 2));
        drop(
            exact
                .try_clone_in(JsonInputKind::DocumentMetadata, &context)
                .unwrap(),
        );
        let oversized =
            JsonValue::String("\n".repeat(JsonProcessingPolicyV1::MAX_UNIT_CANONICAL_BYTES / 2));
        let error = oversized
            .try_clone_in(JsonInputKind::DocumentMetadata, &context)
            .err()
            .unwrap();
        assert!(error
            .detail()
            .iter()
            .any(|(key, value)| key == "json_limit_reason" && value == "unit_bytes_exceeded"));
        assert!(JsonValue::Number(f64::NAN)
            .try_clone_in(JsonInputKind::Filter, &context)
            .is_err());
        let mut all = context.reservation();
        all.grow(
            JsonProcessingPolicyV1::WORK_BUDGET_BYTES,
            JsonInputKind::Filter,
        )
        .unwrap();
    }
}
