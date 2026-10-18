use std::fmt::Write;

use super::*;
use crate::json::JsonTextWriter;

impl FilterExpr {
    pub(crate) fn evaluate_for_chunk_in(
        &self,
        catalog: &Catalog,
        chunk: &Chunk,
        expression: JsonValue,
        context: &JsonOperationContext,
    ) -> Result<(FilterExplain, JsonReservation)> {
        let (eval, work) =
            self.evaluate_lookup_in(&|field| resolve_field(catalog, chunk, field), context)?;
        Ok((
            FilterExplain {
                matched: eval.matched,
                expression,
                reasons: eval.reasons,
            },
            work,
        ))
    }

    fn evaluate_lookup_in<'a>(
        &self,
        lookup: &impl Fn(&str) -> Option<&'a JsonValue>,
        context: &JsonOperationContext,
    ) -> Result<(Eval, JsonReservation)> {
        let kind = JsonInputKind::Filter;
        let mut scratch = context.reservation();
        let mut parents = Vec::<std::slice::Iter<'_, Self>>::new();
        let mut work = context.reservation();
        let mut reasons = Vec::new();
        let mut matched = true;
        let mut next = Some(self);
        while let Some(node) = next {
            if let Self::And(nodes) = node {
                scratch.reserve_vec(&mut parents, 1, kind)?;
                parents.push(nodes.iter());
            } else {
                let (field, op, expected) = match node {
                    Self::Eq(field, value) => (field, "==", Some(value)),
                    Self::Ne(field, value) => (field, "!=", Some(value)),
                    Self::Gt(field, value) => (field, ">", Some(value)),
                    Self::Gte(field, value) => (field, ">=", Some(value)),
                    Self::Lt(field, value) => (field, "<", Some(value)),
                    Self::Lte(field, value) => (field, "<=", Some(value)),
                    Self::Contains(field, value) => (field, "contains", Some(value)),
                    Self::In(field, _) => (field, "in", None),
                    Self::And(_) => unreachable!(),
                };
                let actual = lookup(field);
                let leaf_matched = node.matches_value(actual);
                matched &= leaf_matched;
                work.reserve_vec(&mut reasons, 1, kind)?;
                let mut out = JsonTextWriter::new(context, kind);
                let written = (|| {
                    write!(out, "{field} {op} ")?;
                    if let Self::In(_, values) = node {
                        if leaf_matched {
                            out.write_char('[')?;
                            for (index, value) in values.iter().enumerate() {
                                if index != 0 {
                                    out.write_str(", ")?;
                                }
                                write_label(&mut out, value)?;
                            }
                            return out.write_char(']');
                        }
                        out.write_str("[...]")?;
                    } else {
                        write_label(&mut out, expected.unwrap())?;
                    }
                    if !leaf_matched {
                        out.write_str(" failed")?;
                        if !matches!(node, Self::Ne(_, _)) {
                            out.write_str(": actual ")?;
                            match actual {
                                Some(value) => write_label(&mut out, value)?,
                                None => out.write_str("missing")?,
                            }
                        }
                    }
                    Ok(())
                })();
                let (reason, reserved) = out.finish(written)?;
                reasons.push(reason);
                work.merge(reserved);
            }
            next = loop {
                let Some(parent) = parents.last_mut() else {
                    break None;
                };
                if let Some(node) = parent.next() {
                    break Some(node);
                }
                parents.pop();
            };
        }
        Ok((Eval { matched, reasons }, work))
    }
}

fn write_label(out: &mut JsonTextWriter, value: &JsonValue) -> std::fmt::Result {
    match value {
        JsonValue::String(value) => out.write_str(value),
        JsonValue::Null => out.write_str("null"),
        JsonValue::Bool(value) => out.write_str(if *value { "true" } else { "false" }),
        JsonValue::Number(value) => {
            if value.fract() == 0.0 {
                write!(out, "{value:.0}")
            } else {
                write!(out, "{value}")
            }
        }
        value => out.write_validated_value(value),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::json::{object_from_pairs, JsonProcessingPolicyV1};

    #[test]
    fn streamed_trace_matches_labels_for_scalars_and_nested_actual_values() {
        let expected = [
            JsonValue::Null,
            JsonValue::Bool(true),
            JsonValue::Number(-0.0),
            JsonValue::Number(1.25),
            JsonValue::String("\u{4e2d}\n\\\"".into()),
        ];
        for seed in 0..100 {
            let mut actual =
                object_from_pairs([("value", expected[seed % expected.len()].clone())]);
            for depth in 1..=seed % 127 {
                actual = if depth % 2 == 0 {
                    object_from_pairs([("nested", actual)])
                } else {
                    JsonValue::Array(vec![actual])
                };
            }
            for expected in &expected {
                let filter = FilterExpr::And(vec![
                    FilterExpr::Eq("chunk.x".into(), expected.clone()),
                    FilterExpr::In("chunk.x".into(), vec![expected.clone()]),
                ]);
                let lookup = |_: &str| Some(&actual);
                let legacy = filter.evaluate_lookup(&lookup);
                let context = JsonOperationContext::standalone();
                let (actual, work) = filter.evaluate_lookup_in(&lookup, &context).unwrap();
                assert_eq!(actual.matched, legacy.matched);
                assert_eq!(actual.reasons, legacy.reasons);
                let expected_bytes = actual.reasons.capacity() * std::mem::size_of::<String>()
                    + actual.reasons.iter().map(String::capacity).sum::<usize>();
                assert_eq!(work.reserved_bytes(), expected_bytes);
            }
        }
    }

    #[test]
    fn streamed_in_trace_avoids_per_literal_copies_and_releases_failed_output() {
        for (count, accepted) in [(200_000, true), (600_000, false)] {
            let context = JsonOperationContext::standalone();
            let filter = FilterExpr::In("chunk.x".into(), vec![JsonValue::Null; count]);
            let input = filter
                .reserve_json_work_in(&context, JsonInputKind::Filter)
                .unwrap();
            let mut busy = context.reservation();
            busy.grow(
                JsonProcessingPolicyV1::WORK_BUDGET_BYTES
                    - input.reserved_bytes()
                    - 3 * 1024 * 1024,
                JsonInputKind::Filter,
            )
            .unwrap();
            let actual = JsonValue::Null;
            let result = filter.evaluate_lookup_in(&|_| Some(&actual), &context);
            assert_eq!(result.is_ok(), accepted);
            match result {
                Ok((eval, work)) => {
                    assert!(eval.matched);
                    assert_eq!(eval.reasons.len(), 1);
                    assert!(eval.reasons[0].starts_with("chunk.x in [null, null"));
                    assert!(eval.reasons[0].ends_with("null]"));
                    assert_eq!(
                        eval.reasons[0].len(),
                        "chunk.x in [".len() + 4 * count + 2 * (count - 1) + 1
                    );
                    drop(eval);
                    drop(work);
                }
                Err(error) => {
                    assert_eq!(error.code(), ErrorCode::InvalidMetadataFilter);
                    assert!(error
                        .detail()
                        .iter()
                        .any(|(key, value)| key == "json_limit_reason"
                            && value == "work_budget_exceeded"));
                }
            }
            context
                .reservation()
                .grow(3 * 1024 * 1024, JsonInputKind::Filter)
                .unwrap();
            drop(busy);
            drop(input);
            context
                .reservation()
                .grow(
                    JsonProcessingPolicyV1::WORK_BUDGET_BYTES,
                    JsonInputKind::Filter,
                )
                .unwrap();
        }
    }

    #[test]
    fn trace_uses_iterative_and_walk_on_a_small_stack() {
        let mut filter = FilterExpr::Eq("chunk.x".into(), JsonValue::Number(-0.0));
        for _ in 0..63 {
            filter = FilterExpr::And(vec![filter]);
        }
        std::thread::Builder::new()
            .stack_size(64 * 1024)
            .spawn(move || {
                let context = JsonOperationContext::standalone();
                let (eval, _) = filter.evaluate_lookup_in(&|_| None, &context).unwrap();
                assert!(!eval.matched);
                assert_eq!(eval.reasons, ["chunk.x == -0 failed: actual missing"]);
            })
            .unwrap()
            .join()
            .unwrap();
    }
}
