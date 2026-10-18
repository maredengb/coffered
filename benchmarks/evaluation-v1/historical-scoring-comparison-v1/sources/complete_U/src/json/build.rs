use std::cell::RefCell;
use std::collections::BTreeMap;

use super::{
    work::object_workspace, JsonInputKind, JsonReservation, JsonValue, WorkspaceReservation,
};
use crate::Result;

pub(crate) struct JsonBuilder<'a> {
    work: Option<RefCell<&'a mut JsonReservation>>,
    owned_strings: Option<RefCell<&'a mut WorkspaceReservation>>,
    kind: JsonInputKind,
}

impl<'a> JsonBuilder<'a> {
    pub(crate) fn budgeted(work: &'a mut JsonReservation, kind: JsonInputKind) -> Self {
        Self {
            work: Some(RefCell::new(work)),
            owned_strings: None,
            kind,
        }
    }

    pub(crate) fn budgeted_owned_strings(
        work: &'a mut JsonReservation,
        strings: Option<&'a mut WorkspaceReservation>,
        kind: JsonInputKind,
    ) -> Self {
        Self {
            work: Some(RefCell::new(work)),
            owned_strings: strings.map(RefCell::new),
            kind,
        }
    }

    // Preserve the existing infallible helpers for host-owned public DTOs.
    // Database/CLI operations use budgeted construction instead.
    pub(crate) fn host() -> Self {
        Self {
            work: None,
            owned_strings: None,
            kind: JsonInputKind::JsonlRecord,
        }
    }

    pub(crate) fn string(&self, value: &str) -> Result<JsonValue> {
        let value = match &self.work {
            Some(work) => work.borrow_mut().copy_string(value, self.kind)?,
            None => value.to_owned(),
        };
        Ok(JsonValue::String(value))
    }

    pub(crate) fn display(&self, value: impl std::fmt::Display) -> Result<JsonValue> {
        let value = match &self.work {
            Some(work) => work
                .borrow_mut()
                .format_string(self.kind, |output| write!(output, "{value}"))?,
            None => value.to_string(),
        };
        Ok(JsonValue::String(value))
    }

    // Library DTO wrappers are not new business roots; their values were already validated.
    pub(crate) fn copy_validated_value(&self, value: &JsonValue) -> Result<JsonValue> {
        match &self.work {
            Some(work) => work.borrow_mut().copy_validated_value(value, self.kind),
            None => Ok(value.clone()),
        }
    }

    pub(crate) fn owned_string(&self, value: String) -> Result<JsonValue> {
        if let Some(work) = &self.work {
            match &self.owned_strings {
                Some(strings) => work.borrow_mut().merge(
                    strings
                        .borrow_mut()
                        .take_json(value.capacity(), self.kind)?,
                ),
                None => work.borrow_mut().grow(value.capacity(), self.kind)?,
            }
        }
        Ok(JsonValue::String(value))
    }

    pub(crate) fn optional_string(&self, value: Option<String>) -> Result<JsonValue> {
        value
            .map(|value| self.owned_string(value))
            .transpose()
            .map(|value| value.unwrap_or_default())
    }

    pub(crate) fn optional_str(&self, value: Option<&str>) -> Result<JsonValue> {
        value
            .map(|value| self.string(value))
            .transpose()
            .map(|value| value.unwrap_or_default())
    }

    pub(crate) fn optional_u64(&self, value: Option<u64>) -> Result<JsonValue> {
        let Some(value) = value else {
            return Ok(JsonValue::Null);
        };
        if value <= 9_007_199_254_740_991 {
            return Ok(JsonValue::Number(value as f64));
        }
        use std::io::Write;
        let mut bytes = [0; 20];
        let mut output = std::io::Cursor::new(bytes.as_mut_slice());
        write!(output, "{value}").expect("u64 decimal fits in 20 bytes");
        let length = output.position() as usize;
        self.string(std::str::from_utf8(&bytes[..length]).expect("decimal digits are UTF-8"))
    }

    pub(crate) fn object<const N: usize>(
        &self,
        fields: [(&str, JsonValue); N],
    ) -> Result<JsonValue> {
        let keys_work = self
            .work
            .as_ref()
            .map(|work| {
                work.borrow_mut()
                    .reserve_object_keys(fields.iter().map(|(key, _)| *key), self.kind)
            })
            .transpose()?;
        let mut object = BTreeMap::new();
        for (key, value) in fields {
            let key = if keys_work.is_some() {
                super::work::copy_precharged_string(key)?
            } else {
                key.to_owned()
            };
            object.insert(key, value);
        }
        if let Some(keys_work) = keys_work {
            self.work.as_ref().unwrap().borrow_mut().merge(keys_work);
        }
        Ok(JsonValue::Object(object))
    }

    pub(crate) fn array(
        &self,
        values: impl ExactSizeIterator<Item = Result<JsonValue>>,
    ) -> Result<JsonValue> {
        let mut array = Vec::new();
        match &self.work {
            Some(work) => work
                .borrow_mut()
                .reserve_vec(&mut array, values.len(), self.kind)?,
            None => array.reserve(values.len()),
        }
        for value in values {
            if let Some(work) = &self.work {
                work.borrow_mut().reserve_vec(&mut array, 1, self.kind)?;
            }
            array.push(value?);
        }
        Ok(JsonValue::Array(array))
    }

    pub(crate) fn insert_if_absent(
        &self,
        object: &mut super::JsonObject,
        key: &str,
        value: impl FnOnce() -> Result<JsonValue>,
    ) -> Result<()> {
        if object.contains_key(key) {
            return Ok(());
        }
        let value = value()?;
        let key = match &self.work {
            Some(work) => {
                let mut work = work.borrow_mut();
                work.grow(
                    object_workspace(object.len() + 1) - object_workspace(object.len()),
                    self.kind,
                )?;
                work.copy_string(key, self.kind)?
            }
            None => key.to_owned(),
        };
        object.insert(key, value);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::json::{JsonOperationContext, JsonProcessingPolicyV1};

    #[test]
    fn displayed_paths_preserve_utf8_and_lossy_os_path_rendering() {
        let mut paths = vec![std::path::PathBuf::from("x\u{4e2d}\"\\\n")];
        paths.push(std::path::PathBuf::new());
        #[cfg(unix)]
        {
            use std::os::unix::ffi::OsStringExt;
            paths.push(std::ffi::OsString::from_vec(b"bad-\xff-path".to_vec()).into());
        }
        for path in paths {
            let context = JsonOperationContext::standalone();
            let mut work = context.reservation();
            let expected = JsonValue::String(path.display().to_string());
            let actual = JsonBuilder::budgeted(&mut work, JsonInputKind::StoredJson)
                .display(path.display())
                .unwrap();
            assert_eq!(actual, expected);
            assert_eq!(actual.to_compact_string(), expected.to_compact_string());
            let JsonValue::String(text) = &actual else {
                unreachable!()
            };
            assert_eq!(work.reserved_bytes(), text.capacity());
            drop(actual);
            drop(work);
            assert_eq!(
                context.available_workspace_bytes(),
                JsonProcessingPolicyV1::WORK_BUDGET_BYTES
            );
        }
    }

    #[test]
    fn display_admission_and_partial_render_failures_preserve_existing_credit() {
        struct Changing<'a> {
            calls: &'a std::cell::Cell<usize>,
            second: &'a str,
            fail: usize,
        }
        impl std::fmt::Display for Changing<'_> {
            fn fmt(&self, out: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                let call = self.calls.get() + 1;
                self.calls.set(call);
                out.write_str(if call == 1 { "sized" } else { self.second })?;
                if call == self.fail {
                    Err(std::fmt::Error)
                } else {
                    Ok(())
                }
            }
        }
        let context = JsonOperationContext::standalone();
        let mut work = context.reservation();
        work.grow(13, JsonInputKind::StoredJson).unwrap();
        let available = context.available_workspace_bytes();
        let builder = JsonBuilder::budgeted(&mut work, JsonInputKind::StoredJson);
        for (second, fail) in [("", 0), ("much longer", 0), ("sized", 1), ("sized", 2)] {
            let calls = std::cell::Cell::new(0);
            let error = builder
                .display(Changing {
                    calls: &calls,
                    second,
                    fail,
                })
                .unwrap_err();
            assert_eq!(error.code(), crate::ErrorCode::IoError);
            assert_eq!(context.available_workspace_bytes(), available);
            assert_eq!(calls.get(), if fail == 1 { 1 } else { 2 });
        }
        let mut held = context.workspace_reservation();
        held.grow_to(available - 4, available - 4, "operation_scratch")
            .unwrap();
        let calls = std::cell::Cell::new(0);
        let error = builder
            .display(Changing {
                calls: &calls,
                second: "sized",
                fail: 0,
            })
            .unwrap_err();
        assert_eq!(
            calls.get(),
            1,
            "budget refusal must precede the allocating render pass"
        );
        assert_eq!(error.code(), crate::ErrorCode::RecoveryError);
        assert!(error
            .detail()
            .contains(&("json_limit_reason".into(), "work_budget_exceeded".into())));
        assert_eq!(context.available_workspace_bytes(), 4);
        drop(held);
        let text = "x".repeat(1009);
        let error =
            crate::test_alloc::fail_byte_allocation_once(text.len(), || builder.display(&text))
                .unwrap_err();
        assert_eq!(error.code(), crate::ErrorCode::IoError);
        assert_eq!(context.available_workspace_bytes(), available);
        let value = builder.display(&text).unwrap();
        assert_eq!(value.as_str(), Some(text.as_str()));
        assert_eq!(context.available_workspace_bytes(), available - text.len());
        drop(value);
        drop(work);
        assert_eq!(
            context.available_workspace_bytes(),
            JsonProcessingPolicyV1::WORK_BUDGET_BYTES
        );
    }

    #[test]
    fn object_key_copy_failure_releases_the_whole_object_stage() {
        let context = JsonOperationContext::standalone();
        let mut work = context.reservation();
        work.grow(13, JsonInputKind::StoredJson).unwrap();
        let baseline = context.available_workspace_bytes();
        let first = "a".repeat(4097);
        let second = "b".repeat(8193);
        let builder = JsonBuilder::budgeted(&mut work, JsonInputKind::StoredJson);
        let error = crate::test_alloc::fail_byte_allocation_once(second.len(), || {
            builder.object([
                (first.as_str(), JsonValue::Bool(true)),
                (second.as_str(), JsonValue::Number(7.0)),
            ])
        })
        .unwrap_err();
        assert_eq!(error.code(), crate::ErrorCode::IoError);
        assert_eq!(context.available_workspace_bytes(), baseline);
        let value = builder
            .object([
                (first.as_str(), JsonValue::Bool(true)),
                (second.as_str(), JsonValue::Number(7.0)),
            ])
            .unwrap();
        assert_eq!(value.get(&first).and_then(JsonValue::as_bool), Some(true));
        assert_eq!(value.get(&second).and_then(JsonValue::as_f64), Some(7.0));
        let expected = value
            .reserve_retained_in(JsonInputKind::StoredJson, &context)
            .unwrap();
        assert_eq!(work.reserved_bytes(), 13 + expected.reserved_bytes());
    }

    #[test]
    fn duplicate_dto_keys_keep_last_value_under_the_aggregate_upper_bound() {
        let context = JsonOperationContext::standalone();
        let mut work = context.reservation();
        let value = JsonBuilder::budgeted(&mut work, JsonInputKind::StoredJson)
            .object([
                ("key", JsonValue::Bool(false)),
                ("key", JsonValue::Bool(true)),
            ])
            .unwrap();
        assert_eq!(value.get("key").and_then(JsonValue::as_bool), Some(true));
        assert_eq!(value.as_object().unwrap().len(), 1);
        let measured = value
            .reserve_retained_in(JsonInputKind::StoredJson, &context)
            .unwrap();
        assert!(work.reserved_bytes() >= measured.reserved_bytes());
        drop(measured);
        drop(value);
        drop(work);
        assert_eq!(
            context.available_workspace_bytes(),
            JsonProcessingPolicyV1::WORK_BUDGET_BYTES
        );
    }

    #[test]
    fn fixed_dto_builder_accounts_for_actual_capacity_without_copying_owned_strings() {
        for seed in 0..100 {
            let context = JsonOperationContext::standalone();
            let mut work = context.reservation();
            let builder = JsonBuilder::budgeted(&mut work, JsonInputKind::StoredJson);
            let mut owned = String::with_capacity(256 + seed);
            owned.push_str("owned");
            let pointer = owned.as_ptr();
            let value = builder
                .object([
                    ("owned", builder.owned_string(owned).unwrap()),
                    (
                        "array",
                        builder
                            .array(
                                (0..seed)
                                    .map(|i| builder.object([("i", JsonValue::Number(i as f64))])),
                            )
                            .unwrap(),
                    ),
                    ("text", builder.string("\u{4e2d}\n").unwrap()),
                    ("none", builder.optional_string(None).unwrap()),
                ])
                .unwrap();
            assert_eq!(
                value.get("owned").unwrap().as_str().unwrap().as_ptr(),
                pointer
            );
            let expected = value
                .reserve_retained_in(JsonInputKind::StoredJson, &context)
                .unwrap();
            assert_eq!(work.reserved_bytes(), expected.reserved_bytes());
        }
    }

    #[test]
    fn validated_dto_copies_preserve_values_and_account_before_allocation() {
        for seed in 0..100 {
            let source = JsonValue::Array(vec![
                JsonValue::String("x".repeat(13 * seed)),
                super::super::object_from_pairs([("n", JsonValue::Number(seed as f64))]),
            ]);
            let context = JsonOperationContext::standalone();
            let mut work = context.reservation();
            let copy = JsonBuilder::budgeted(&mut work, JsonInputKind::BenchmarkArtifact)
                .copy_validated_value(&source)
                .unwrap();
            assert_eq!(copy, source);
            let expected = copy
                .reserve_retained_in(JsonInputKind::BenchmarkArtifact, &context)
                .unwrap();
            assert_eq!(work.reserved_bytes(), expected.reserved_bytes());
            drop(expected);
            drop(copy);
            drop(work);
            let mut busy = context.reservation();
            busy.grow(
                JsonProcessingPolicyV1::WORK_BUDGET_BYTES,
                JsonInputKind::BenchmarkArtifact,
            )
            .unwrap();
            let mut work = context.reservation();
            let error = JsonBuilder::budgeted(&mut work, JsonInputKind::BenchmarkArtifact)
                .copy_validated_value(&source)
                .unwrap_err();
            assert_eq!(error.code(), crate::ErrorCode::InvalidJsonlSchema);
            assert_eq!(work.reserved_bytes(), 0);
            drop(busy);
            JsonBuilder::budgeted(&mut work, JsonInputKind::BenchmarkArtifact)
                .copy_validated_value(&source)
                .unwrap();
        }
    }

    #[test]
    fn validated_dto_copy_does_not_reapply_business_limits_to_wrappers() {
        let context = JsonOperationContext::standalone();
        let mut work = context.reservation();
        let builder = JsonBuilder::budgeted(&mut work, JsonInputKind::BenchmarkArtifact);
        let mut deep = JsonValue::Null;
        for _ in 0..JsonProcessingPolicyV1::MAX_CONTAINER_DEPTH + 3 {
            deep = JsonValue::Array(vec![deep]);
        }
        let copy = builder.copy_validated_value(&deep).unwrap();
        assert_eq!(copy.to_compact_string(), deep.to_compact_string());
        assert!(deep
            .try_clone_in(JsonInputKind::BenchmarkArtifact, &context)
            .is_err());
        let wide = JsonValue::Array(vec![
            JsonValue::String("a".repeat(JsonProcessingPolicyV1::MAX_UNIT_CANONICAL_BYTES / 2)),
            JsonValue::String("b".repeat(JsonProcessingPolicyV1::MAX_UNIT_CANONICAL_BYTES / 2)),
        ]);
        let copy = builder.copy_validated_value(&wide).unwrap();
        assert_eq!(copy, wide);
        assert!(wide
            .try_clone_in(JsonInputKind::BenchmarkArtifact, &context)
            .is_err());
    }

    #[test]
    fn lazy_fields_keep_existing_values_and_charge_only_added_allocations() {
        for count in 0..100 {
            let context = JsonOperationContext::standalone();
            let mut work = context.reservation();
            let builder = JsonBuilder::budgeted(&mut work, JsonInputKind::StoredJson);
            let mut object = BTreeMap::new();
            builder
                .insert_if_absent(&mut object, "existing", || builder.string("original"))
                .unwrap();
            for index in 0..count {
                builder
                    .insert_if_absent(&mut object, &format!("field-{index}"), || {
                        builder.string("value")
                    })
                    .unwrap();
            }
            builder
                .insert_if_absent(&mut object, "existing", || {
                    panic!("existing fields must not be constructed again")
                })
                .unwrap();
            let value = JsonValue::Object(object);
            let measured = value
                .reserve_retained_in(JsonInputKind::StoredJson, &context)
                .unwrap();
            assert_eq!(work.reserved_bytes(), measured.reserved_bytes());
            assert_eq!(
                value.get("existing").and_then(JsonValue::as_str),
                Some("original")
            );
        }
    }

    #[test]
    fn fixed_dto_builder_rejects_before_advancing_an_oversized_array() {
        let context = JsonOperationContext::standalone();
        let mut work = context.reservation();
        work.grow(
            JsonProcessingPolicyV1::WORK_BUDGET_BYTES - 64,
            JsonInputKind::StoredJson,
        )
        .unwrap();
        let builder = JsonBuilder::budgeted(&mut work, JsonInputKind::StoredJson);
        let calls = std::cell::Cell::new(0);
        let values = (0..100).map(|_| {
            calls.set(calls.get() + 1);
            Ok(JsonValue::Null)
        });
        assert!(builder.array(values).is_err());
        assert_eq!(calls.get(), 0);
        assert!(builder.string(&"x".repeat(128)).is_err());
        drop(work);
        context
            .reservation()
            .grow(
                JsonProcessingPolicyV1::WORK_BUDGET_BYTES,
                JsonInputKind::StoredJson,
            )
            .unwrap();
    }

    #[test]
    fn full_width_integer_output_is_exact_and_precharged() {
        for value in [
            None,
            Some(0),
            Some(9_007_199_254_740_991),
            Some(9_007_199_254_740_992),
        ]
        .into_iter()
        .chain((0..100).map(|offset| Some(u64::MAX - offset)))
        {
            let context = JsonOperationContext::standalone();
            let mut work = context.reservation();
            let actual = JsonBuilder::budgeted(&mut work, JsonInputKind::StoredJson)
                .optional_u64(value)
                .unwrap();
            let measured = actual
                .reserve_retained_in(JsonInputKind::StoredJson, &context)
                .unwrap();
            assert_eq!(work.reserved_bytes(), measured.reserved_bytes());
            match value {
                None => assert!(actual.is_null()),
                Some(value) if value <= 9_007_199_254_740_991 => {
                    assert_eq!(actual.as_f64(), Some(value as f64))
                }
                Some(value) => assert_eq!(actual.as_str(), Some(value.to_string().as_str())),
            }
        }
    }
}
