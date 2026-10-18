use super::ImportWarning;
use crate::json::{
    copy_precharged_string, JsonOperationContext, ReservedVec, WorkspaceReservation,
};
use crate::Result;

// Array capacity and all warning strings remain owned until the summary is handed off.
pub(crate) struct ImportWarnings {
    values: ReservedVec<ImportWarning>,
    payload: WorkspaceReservation,
}

impl ImportWarnings {
    pub(crate) const SOURCE_SPAN_CONFLICT: [&'static str; 4] = [
        "source_span_metadata_conflict",
        "source_span",
        "chunks",
        "source_span conflicts with metadata byte/line boundary fields; source_span was used",
    ];

    pub(crate) fn new(context: &JsonOperationContext) -> Result<Self> {
        Ok(Self {
            values: ReservedVec::with_capacity(0, context, "operation_scratch")?,
            payload: context.workspace_reservation(),
        })
    }

    pub(crate) fn push(&mut self, fields: [&str; 4]) -> Result<()> {
        let mut work = self.payload.context().workspace_reservation();
        let bytes = fields.iter().map(|value| value.len() as u128).sum();
        work.grow_to_wide(bytes, bytes, "operation_scratch")?;
        let [code, field, artifact_kind, message] = [
            copy_precharged_string(fields[0])?,
            copy_precharged_string(fields[1])?,
            copy_precharged_string(fields[2])?,
            copy_precharged_string(fields[3])?,
        ];
        self.values.try_push(ImportWarning {
            code,
            field,
            artifact_kind,
            message,
        })?;
        self.payload.merge(work);
        Ok(())
    }

    pub(crate) fn into_parts(self) -> (Vec<ImportWarning>, WorkspaceReservation) {
        let (values, mut work) = self.values.into_parts();
        work.merge(self.payload);
        (values, work)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::json::WorkLedger;
    use crate::ErrorCode;
    use std::sync::Arc;

    fn payload_bytes() -> usize {
        ImportWarnings::SOURCE_SPAN_CONFLICT
            .iter()
            .map(|s| s.len())
            .sum()
    }

    #[test]
    fn aggregate_growth_and_summary_handoff_cover_all_live_capacity() {
        let context = JsonOperationContext::standalone();
        let available = context.available_workspace_bytes();
        let mut warnings = ImportWarnings::new(&context).unwrap();
        for count in 1..=257 {
            warnings.push(ImportWarnings::SOURCE_SPAN_CONFLICT).unwrap();
            let expected = warnings.values.capacity() * std::mem::size_of::<ImportWarning>()
                + count * payload_bytes();
            assert_eq!(context.available_workspace_bytes(), available - expected);
        }
        let (values, work) = warnings.into_parts();
        assert_eq!(values.len(), 257);
        assert_eq!(
            context.available_workspace_bytes(),
            available - work.reserved_bytes()
        );
        for value in &values {
            assert_eq!(value.code, "source_span_metadata_conflict");
            assert_eq!(value.field, "source_span");
            assert_eq!(value.artifact_kind, "chunks");
            assert_eq!(value.message, ImportWarnings::SOURCE_SPAN_CONFLICT[3]);
        }
        drop(values);
        drop(work);
        assert_eq!(context.available_workspace_bytes(), available);
    }

    #[test]
    fn required_capacity_and_failed_growth_preserve_prior_warnings_on_both_axes() {
        let minimum = std::mem::size_of::<ImportWarning>() + payload_bytes();
        for parent in [false, true] {
            for limit in [minimum - 1, minimum, minimum + 1] {
                let context = JsonOperationContext::new(Arc::new(WorkLedger::with_limits(
                    if parent { 2 * minimum } else { limit },
                    0,
                    if parent { limit } else { 2 * minimum },
                )));
                let mut warnings = ImportWarnings::new(&context).unwrap();
                let result = warnings.push(ImportWarnings::SOURCE_SPAN_CONFLICT);
                if limit < minimum {
                    assert_eq!(result.unwrap_err().code(), ErrorCode::ResourceExhausted);
                    assert!(warnings.values.as_slice().is_empty());
                    assert_eq!(context.available_workspace_bytes(), limit);
                } else {
                    result.unwrap();
                    let available = context.available_workspace_bytes();
                    let old = warnings.values.as_slice().to_vec();
                    let error = warnings
                        .push(ImportWarnings::SOURCE_SPAN_CONFLICT)
                        .unwrap_err();
                    assert_eq!(error.code(), ErrorCode::ResourceExhausted);
                    assert!(error
                        .detail()
                        .contains(&("resource".into(), "memory".into())));
                    assert_eq!(warnings.values.as_slice(), old);
                    assert_eq!(context.available_workspace_bytes(), available);
                }
                drop(warnings);
                assert_eq!(context.available_workspace_bytes(), limit);
            }
        }
    }

    #[test]
    fn native_string_and_array_failures_release_partial_copies_without_mutation() {
        let context = JsonOperationContext::standalone();
        let available = context.available_workspace_bytes();
        let mut warnings = ImportWarnings::new(&context).unwrap();
        for _ in 0..4 {
            warnings.push(ImportWarnings::SOURCE_SPAN_CONFLICT).unwrap();
        }
        let retained = context.available_workspace_bytes();
        let old = warnings.values.as_slice().to_vec();
        for ordinal in [1, 4, 5] {
            let error = crate::test_alloc::fail_nth_allocation_once(ordinal, || {
                warnings.push(ImportWarnings::SOURCE_SPAN_CONFLICT)
            })
            .unwrap_err();
            assert_eq!(error.code(), ErrorCode::IoError);
            assert_eq!(warnings.values.as_slice(), old);
            assert_eq!(context.available_workspace_bytes(), retained);
        }
        warnings.push(ImportWarnings::SOURCE_SPAN_CONFLICT).unwrap();
        assert_eq!(warnings.values.as_slice().len(), 5);
        drop(warnings);
        assert_eq!(context.available_workspace_bytes(), available);
    }
}
