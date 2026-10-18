#[cfg(test)]
use super::normalized::lowercase_layout;
use super::{
    normalized::{normalize, Token},
    symbol_parts, symbols,
};
use crate::json::{
    btree_workspace, is_memory_budget_error, JsonOperationContext, WorkspaceReservation,
};
use crate::Result;
use std::borrow::Cow;
use std::collections::BTreeMap;

pub(crate) struct QueryTerms<'a> {
    terms: BTreeMap<Cow<'a, str>, usize>,
    work: WorkspaceReservation,
}

impl QueryTerms<'_> {
    pub(crate) fn len(&self) -> usize {
        self.terms.len()
    }

    pub(crate) fn iter(&self) -> impl Iterator<Item = &str> + Clone {
        self.terms.keys().map(Cow::as_ref)
    }

    pub(crate) fn position(&self, term: &str) -> Option<usize> {
        self.terms.get(term).copied()
    }
}

pub(crate) fn query_terms_in<'a>(
    query: &'a str,
    context: &JsonOperationContext,
) -> Result<QueryTerms<'a>> {
    let mut result = QueryTerms {
        terms: BTreeMap::new(),
        work: context.workspace_reservation(),
    };
    for raw in symbols(query).flat_map(|symbol| {
        std::iter::once(symbol).chain(symbol_parts(symbol).filter(move |part| *part != symbol))
    }) {
        // Small queries can discard repeated ASCII casing without making a copy.
        if raw.is_ascii()
            && result.len() <= 16
            && result.iter().any(|term| term.eq_ignore_ascii_case(raw))
        {
            continue;
        }
        let token = match normalize(raw, context) {
            Ok(Some(token)) => token,
            Ok(None) => continue,
            Err(error)
                if raw.is_ascii()
                    && is_memory_budget_error(&error)
                    && result.iter().any(|term| term.eq_ignore_ascii_case(raw)) =>
            {
                continue
            }
            Err(error) => return Err(error),
        };
        if result.terms.contains_key(token.value.as_ref()) {
            continue;
        }
        let previous = btree_workspace::<Cow<'_, str>, usize>(result.len());
        let next = btree_workspace::<Cow<'_, str>, usize>(result.len() + 1);
        let needed = result.work.reserved_bytes() + next - previous;
        result.work.grow_to(needed, needed, "operation_scratch")?;
        let Token { value, work } = token;
        result.terms.insert(value, 0);
        result.work.merge(work);
    }
    for (position, (_, slot)) in result.terms.iter_mut().enumerate() {
        *slot = position;
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::json::WorkLedger;
    use crate::{Error, ErrorCode};
    use std::sync::Arc;

    fn node_bytes(count: usize) -> usize {
        if count == 0 {
            return 0;
        }
        count.div_ceil(5)
            * (11 * std::mem::size_of::<(Cow<'_, str>, usize)>()
                + 16 * std::mem::size_of::<usize>())
    }

    fn assert_released(context: &JsonOperationContext, limit: usize) {
        let mut probe = context.workspace_reservation();
        probe.grow_to(limit, limit, "operation_scratch").unwrap();
        assert!(probe
            .grow_to(limit + 1, limit + 1, "operation_scratch")
            .is_err());
    }

    #[test]
    fn original_order_casing_and_code_parts_match_the_frozen_iterator() {
        for seed in 0..100 {
            let query = format!("the and THE AlphaBeta alphaBeta fooFoo foo_bar_foo {} {} \u{39f}\u{3a3}Value \u{0130} caf\u{00e9} \u{10428} 42", "REPEAT ".repeat(seed % 7), (0..seed).map(|i| format!("unique{i} ")).collect::<String>());
            let expected =
                super::super::query_tokens(&query).collect::<std::collections::BTreeSet<_>>();
            let context = JsonOperationContext::standalone();
            let available = context.available_workspace_bytes();
            let terms = query_terms_in(&query, &context).unwrap();
            assert_eq!(
                terms.iter().collect::<Vec<_>>(),
                expected.iter().map(Cow::as_ref).collect::<Vec<_>>()
            );
            for (position, term) in terms.iter().enumerate() {
                assert_eq!(terms.position(term), Some(position));
            }
            assert_eq!(terms.position("absent"), None);
            let payload = terms
                .terms
                .keys()
                .map(|term| {
                    if let Cow::Owned(value) = term {
                        value.capacity()
                    } else {
                        0
                    }
                })
                .sum::<usize>();
            assert_eq!(
                terms.work.reserved_bytes(),
                node_bytes(terms.len()) + payload
            );
            assert_eq!(
                context.available_workspace_bytes(),
                available - node_bytes(terms.len()) - payload
            );
            let mut original_operation = context.workspace_reservation();
            original_operation.merge(terms.work.context().workspace_reservation());
            drop(terms);
            assert_released(&context, available);
        }
    }

    #[test]
    fn query_node_and_owned_payload_minimum_is_admitted_on_each_independent_ledger() {
        let minimum = node_bytes(1) + "ALPHA".len();
        for parent in [false, true] {
            for delta in [-1isize, 0, 1] {
                let limit = (minimum as isize + delta) as usize;
                let context = JsonOperationContext::new(Arc::new(WorkLedger::with_limits(
                    if parent { limit * 2 } else { limit },
                    limit * 2,
                    if parent { limit } else { limit * 2 },
                )));
                let result = query_terms_in("ALPHA", &context);
                if delta < 0 {
                    let error = result.err().unwrap();
                    assert_eq!(error.code(), ErrorCode::ResourceExhausted);
                    for (key, expected) in [
                        ("resource", "memory".to_owned()),
                        ("operation", "operation_scratch".to_owned()),
                        ("limit_bytes", limit.to_string()),
                        ("requested_bytes", minimum.to_string()),
                    ] {
                        assert_eq!(
                            error.detail().iter().find(|item| item.0 == key).unwrap().1,
                            expected
                        );
                    }
                } else {
                    let terms = result.unwrap();
                    assert_eq!(terms.iter().collect::<Vec<_>>(), ["alpha"]);
                    assert_eq!(terms.work.reserved_bytes(), minimum);
                    assert_eq!(context.available_workspace_bytes(), delta as usize);
                    drop(terms);
                }
                assert_released(&context, limit);
            }
        }
    }

    #[test]
    fn ascii_normalization_reserves_before_allocating_and_borrowed_terms_need_no_payload() {
        let raw = "A".repeat(65537);
        let expected = "a".repeat(raw.len());
        for parent in [false, true] {
            for delta in [-1isize, 0, 1] {
                let limit = (raw.len() as isize + delta) as usize;
                let context = JsonOperationContext::new(Arc::new(WorkLedger::with_limits(
                    if parent { 2 * limit } else { limit },
                    2 * limit,
                    if parent { limit } else { 2 * limit },
                )));
                let (result, allocation) = crate::test_alloc::key_allocations(raw.len(), || {
                    let token = normalize(&raw, &context)?;
                    if let Some(token) = token {
                        assert_eq!(token.value.as_ref(), expected);
                        assert_eq!(token.work.reserved_bytes(), raw.len());
                        drop(token);
                    }
                    Ok::<(), Error>(())
                });
                if delta < 0 {
                    assert_eq!(result.unwrap_err().code(), ErrorCode::ResourceExhausted);
                    assert_eq!(allocation.allocated, 0);
                } else {
                    result.unwrap();
                    assert_eq!(allocation.allocated, 1);
                    assert_eq!(allocation.peak, 1);
                }
                assert_eq!(allocation.live, 0);
                assert!(!allocation.overflow);
                assert_released(&context, limit);
            }
        }
        let context = JsonOperationContext::new(Arc::new(WorkLedger::with_limits(0, 0, 0)));
        for raw in ["unchanged", "caf\u{00e9}", "\u{10428}"] {
            let token = normalize(raw, &context).unwrap().unwrap();
            assert!(matches!(token.value, Cow::Borrowed(value) if value.as_ptr() == raw.as_ptr()));
            assert_eq!(token.work.reserved_bytes(), 0);
        }
        for raw in ["the", "THE", "An", "AND"] {
            assert!(normalize(raw, &context).unwrap().is_none());
        }
    }

    #[test]
    fn repeated_queries_do_not_require_capacity_for_raw_occurrence_count() {
        let query = "ALPHA Beta ALPHA beta ".repeat(20000);
        let minimum = node_bytes(2) + "alpha".len() + "beta".len();
        let context =
            JsonOperationContext::new(Arc::new(WorkLedger::with_limits(minimum, minimum, minimum)));
        let terms = query_terms_in(&query, &context).unwrap();
        assert_eq!(terms.iter().collect::<Vec<_>>(), ["alpha", "beta"]);
        assert_eq!(terms.work.reserved_bytes(), minimum);
        drop(terms);
        assert_released(&context, minimum);
    }

    #[test]
    fn duplicate_casing_survives_zero_spare_credit_on_both_sides_of_the_scan_threshold() {
        for count in [16usize, 17] {
            let query = (0..count).map(|i| format!("word{i} ")).collect::<String>();
            let minimum = node_bytes(count);
            for parent in [false, true] {
                let context = JsonOperationContext::new(Arc::new(WorkLedger::with_limits(
                    if parent { 2 * minimum } else { minimum },
                    2 * minimum,
                    if parent { minimum } else { 2 * minimum },
                )));
                let repeated = format!("{query}WORD0");
                let terms = query_terms_in(&repeated, &context).unwrap();
                assert_eq!(terms.len(), count);
                assert_eq!(terms.position("word0"), Some(0));
                assert_eq!(terms.work.reserved_bytes(), minimum);
                assert_eq!(context.available_workspace_bytes(), 0);
                assert!(terms
                    .terms
                    .keys()
                    .all(|term| matches!(term, Cow::Borrowed(_))));
                drop(terms);
                assert_released(&context, minimum);

                let distinct = format!("{query}ABSENT");
                let error = query_terms_in(&distinct, &context).err().unwrap();
                assert_eq!(error.code(), ErrorCode::ResourceExhausted);
                for (key, expected) in [
                    ("resource", "memory".to_owned()),
                    ("operation", "operation_scratch".to_owned()),
                    ("limit_bytes", minimum.to_string()),
                    ("requested_bytes", (minimum + "ABSENT".len()).to_string()),
                ] {
                    assert_eq!(
                        error.detail().iter().find(|item| item.0 == key).unwrap().1,
                        expected
                    );
                }
                assert_released(&context, minimum);
            }
        }
    }

    #[test]
    fn large_query_dictionary_is_sorted_and_shared_credit_outlives_the_context_handle() {
        let query = (0..10000)
            .rev()
            .map(|i| format!("unique{i} "))
            .collect::<String>();
        let limit = node_bytes(10000);
        let ledger = Arc::new(WorkLedger::with_limits(limit, 2 * limit, 2 * limit));
        let context = JsonOperationContext::new(ledger.clone());
        let other = JsonOperationContext::new(ledger);
        let terms = query_terms_in(&query, &context).unwrap();
        assert_eq!(terms.len(), 10000);
        let values = terms.iter().collect::<Vec<_>>();
        assert!(values.windows(2).all(|pair| pair[0] < pair[1]));
        assert_eq!(context.available_workspace_bytes(), 0);
        drop(context);
        assert!(other
            .workspace_reservation()
            .grow_to(1, 1, "operation_scratch")
            .is_err());
        drop(terms);
        assert_released(&other, limit);
    }

    #[test]
    fn unicode_results_keep_standard_contextual_casing_and_actual_resident_capacity() {
        for (raw, capacity, peak) in [
            ("\u{39f}\u{3a3}", 4, 4),
            ("\u{39f}\u{3a3}Value", 9, 9),
            ("\u{39f}\u{0345}\u{3a3}", 6, 6),
            ("\u{39f}\u{3a3}_\u{3a3}", 7, 7),
            ("\u{0130}", 8, 10),
            ("\u{212a}ELVIN", 8, 8),
            ("\u{10400}", 4, 4),
        ] {
            let expected = raw.to_lowercase();
            assert_eq!(expected.capacity(), capacity);
            let context =
                JsonOperationContext::new(Arc::new(WorkLedger::with_limits(peak, peak, peak)));
            let token = normalize(raw, &context).unwrap().unwrap();
            assert_eq!(token.value.as_ref(), expected);
            assert_eq!(token.work.reserved_bytes(), capacity);
            assert_eq!(context.available_workspace_bytes(), peak - capacity);
            drop(token);
            assert_released(&context, peak);
        }
    }

    #[test]
    fn unicode_expansion_peak_is_admitted_before_target_allocation_on_each_ledger() {
        for count in [1usize, 2, 3, 9, 4097, 16384] {
            let raw = "\u{0130}".repeat(count);
            let expected = "i\u{0307}".repeat(count);
            let capacity = (4 * count).max(8);
            let minimum = 2 * count + capacity;
            for parent in [false, true] {
                for delta in [-1isize, 0, 1] {
                    let limit = (minimum as isize + delta) as usize;
                    let context = JsonOperationContext::new(Arc::new(WorkLedger::with_limits(
                        if parent { 2 * minimum } else { limit },
                        2 * minimum,
                        if parent { limit } else { 2 * minimum },
                    )));
                    let (result, allocations) =
                        crate::test_alloc::key_allocations(raw.len(), || {
                            let result = normalize(&raw, &context);
                            if let Ok(token) = &result {
                                let token = token.as_ref().expect("dotted I is not a stop word");
                                assert_eq!(token.value.as_ref(), expected);
                                assert_eq!(token.work.reserved_bytes(), capacity);
                                assert_eq!(context.available_workspace_bytes(), limit - capacity);
                            }
                            if let Err(error) = &result {
                                assert_eq!(error.code(), ErrorCode::ResourceExhausted);
                                for (key, value) in [
                                    ("resource", "memory".into()),
                                    ("operation", "operation_scratch".into()),
                                    ("limit_bytes", limit.to_string()),
                                    ("requested_bytes", minimum.to_string()),
                                ] {
                                    assert!(
                                        error.detail().contains(&(key.into(), value)),
                                        "{error:?}"
                                    );
                                }
                            }
                            result.map(drop).map_err(drop)
                        });
                    if delta < 0 {
                        assert!(result.is_err());
                        // Small sizes also match error-detail strings; large inputs isolate
                        // the initial normalization allocation from diagnostics.
                        if count >= 4097 {
                            assert_eq!(allocations.allocated, 0);
                        }
                    } else {
                        result.unwrap();
                        if count >= 4097 {
                            assert_eq!(allocations.allocated, 1);
                            assert_eq!(allocations.peak, 1);
                        }
                    }
                    assert_eq!(allocations.live, 0);
                    assert!(!allocations.overflow);
                    assert_released(&context, limit);
                }
            }
        }
    }

    #[test]
    fn unicode_query_minimum_is_maximum_of_conversion_and_dictionary_phases() {
        let count = 4097;
        let raw = "\u{0130}".repeat(count);
        let query = format!("anchor {raw}");
        let expected = "i\u{0307}".repeat(count);
        let capacity = 4 * count;
        let conversion_peak = 2 * count + capacity;
        let previous = node_bytes(1);
        let minimum = previous + conversion_peak.max(capacity + node_bytes(2) - previous);
        let resident = node_bytes(2) + capacity;
        for parent in [false, true] {
            for delta in [-1isize, 0, 1] {
                let limit = (minimum as isize + delta) as usize;
                let context = JsonOperationContext::new(Arc::new(WorkLedger::with_limits(
                    if parent { 2 * minimum } else { limit },
                    2 * minimum,
                    if parent { limit } else { 2 * minimum },
                )));
                let (result, allocations) = crate::test_alloc::key_allocations(raw.len(), || {
                    let result = query_terms_in(&query, &context);
                    if let Ok(terms) = &result {
                        assert_eq!(terms.iter().collect::<Vec<_>>(), ["anchor", &expected]);
                        assert_eq!(terms.work.reserved_bytes(), resident);
                        assert_eq!(context.available_workspace_bytes(), limit - resident);
                    }
                    result.map(drop)
                });
                if delta < 0 {
                    let error = result.unwrap_err();
                    assert_eq!(error.code(), ErrorCode::ResourceExhausted);
                    assert!(error
                        .detail()
                        .contains(&("requested_bytes".into(), minimum.to_string())));
                    assert!(error
                        .detail()
                        .contains(&("limit_bytes".into(), limit.to_string())));
                    assert_eq!(allocations.allocated, 0);
                } else {
                    result.unwrap();
                    assert_eq!(allocations.allocated, 1);
                }
                assert_eq!(allocations.live, 0);
                assert!(!allocations.overflow);
                assert_released(&context, limit);
            }
        }
    }

    #[test]
    fn unicode_dictionary_growth_refusal_follows_successful_conversion_and_releases_credit() {
        let query = "word0 word1 word2 word3 word4 \u{0130}";
        let previous = node_bytes(5);
        let minimum = previous + 10usize.max(8 + node_bytes(6) - previous);
        assert_eq!(minimum, node_bytes(6) + 8);
        assert!(previous + 10 < minimum - 1);
        for parent in [false, true] {
            for delta in [-1isize, 0, 1] {
                let limit = (minimum as isize + delta) as usize;
                let context = JsonOperationContext::new(Arc::new(WorkLedger::with_limits(
                    if parent { 2 * minimum } else { limit },
                    2 * minimum,
                    if parent { limit } else { 2 * minimum },
                )));
                let result = query_terms_in(query, &context);
                if delta < 0 {
                    let error = result.err().unwrap();
                    assert_eq!(error.code(), ErrorCode::ResourceExhausted);
                    for (key, value) in [
                        ("resource", "memory".into()),
                        ("operation", "operation_scratch".into()),
                        ("limit_bytes", limit.to_string()),
                        ("requested_bytes", minimum.to_string()),
                    ] {
                        assert!(error.detail().contains(&(key.into(), value)), "{error:?}");
                    }
                } else {
                    let terms = result.unwrap();
                    assert_eq!(
                        terms.iter().collect::<Vec<_>>(),
                        ["i\u{0307}", "word0", "word1", "word2", "word3", "word4"]
                    );
                    assert_eq!(terms.work.reserved_bytes(), minimum);
                    assert_eq!(context.available_workspace_bytes(), delta as usize);
                    drop(terms);
                }
                assert_released(&context, limit);
            }
        }
    }

    #[test]
    fn unicode_scalar_casing_matches_standard_and_verified_capacity_envelope() {
        let context = JsonOperationContext::standalone();
        let available = context.available_workspace_bytes();
        let mut changed = 0;
        for value in 128..=0x10ffff {
            let Some(ch) = char::from_u32(value) else {
                continue;
            };
            if ch.to_lowercase().eq(std::iter::once(ch)) {
                continue;
            }
            let mut encoded = [0; 4];
            let raw = ch.encode_utf8(&mut encoded);
            let expected = raw.to_lowercase();
            let token = normalize(raw, &context).unwrap().unwrap();
            assert_eq!(token.value.as_ref(), expected, "U+{value:04X}");
            assert_eq!(token.work.reserved_bytes(), expected.capacity());
            assert!(expected.capacity() as u128 <= lowercase_layout(raw).0);
            drop(token);
            assert_eq!(context.available_workspace_bytes(), available);
            changed += 1;
        }
        assert!(changed > 1000);
    }

    #[test]
    fn partial_tree_and_owned_key_growth_failures_return_exact_credit() {
        for count in 1..=100usize {
            let query = (0..count).map(|i| format!("WORD{i} ")).collect::<String>();
            let payload = (0..count).map(|i| format!("word{i}").len()).sum::<usize>();
            let minimum = node_bytes(count) + payload;
            for parent in [false, true] {
                for delta in [-1isize, 0, 1] {
                    let limit = (minimum as isize + delta) as usize;
                    let context = JsonOperationContext::new(Arc::new(WorkLedger::with_limits(
                        if parent { 2 * limit } else { limit },
                        2 * limit,
                        if parent { limit } else { 2 * limit },
                    )));
                    let result = query_terms_in(&query, &context);
                    if delta < 0 {
                        let error = result.err().unwrap();
                        assert_eq!(error.code(), ErrorCode::ResourceExhausted);
                        assert_eq!(
                            error
                                .detail()
                                .iter()
                                .find(|item| item.0 == "requested_bytes")
                                .unwrap()
                                .1,
                            minimum.to_string()
                        );
                    } else {
                        let terms = result.unwrap();
                        assert_eq!(terms.len(), count);
                        assert_eq!(terms.work.reserved_bytes(), minimum);
                        assert_eq!(context.available_workspace_bytes(), delta as usize);
                        drop(terms);
                    }
                    assert_released(&context, limit);
                }
            }
        }
    }
}
