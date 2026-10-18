use super::*;
use crate::json::{JsonProcessingPolicyV1, WorkspaceReservation};
use std::cell::RefCell;
use std::mem::size_of;
use std::rc::Rc;

struct HookReset;

impl Drop for HookReset {
    fn drop(&mut self) {
        QUERY_DTO_READY.with(|slot| slot.borrow_mut().take());
    }
}

fn install(stage: QueryDtoStage, hook: QueryDtoHook) -> HookReset {
    QUERY_DTO_READY.with(|slot| {
        assert!(slot.borrow().is_none());
        *slot.borrow_mut() = Some((stage, hook));
    });
    HookReset
}

fn layout(value: &JsonValue) -> usize {
    match value {
        JsonValue::String(value) => value.capacity(),
        JsonValue::Array(values) => {
            values.capacity() * size_of::<JsonValue>() + values.iter().map(layout).sum::<usize>()
        }
        JsonValue::Object(values) => {
            let nodes = if values.is_empty() {
                0
            } else {
                1 + (values.len() - 1) / 5
            };
            nodes * (11 * size_of::<(String, JsonValue)>() + 16 * size_of::<usize>())
                + values
                    .iter()
                    .map(|(key, value)| key.capacity() + layout(value))
                    .sum::<usize>()
        }
        _ => 0,
    }
}

fn assert_retained(result: &CommandResult) {
    let (_, work, _) = result.retained.as_ref().unwrap();
    assert_eq!(work.reserved_bytes(), layout(&result.value));
}

fn args(path: &std::path::Path, why_not: bool, extra: Vec<String>) -> Args {
    let mut args = vec![path.display().to_string()];
    if why_not {
        args.extend([
            "--chunk-external-id".into(),
            "doc#1".into(),
            "--query".into(),
            "stable".into(),
        ]);
    } else {
        args.push("stable".into());
    }
    args.extend([
        "--collection".into(),
        "notes".into(),
        "--filter".into(),
        r#"{"$and":[{}]}"#.into(),
    ]);
    args.extend(extra);
    Args::new(args)
}

#[test]
fn actual_query_commands_close_database_before_retained_conversion() {
    let (path, _, _) = create_cli_schema_fixture("dto-db-close");
    {
        let db = Database::open(&path).unwrap();
        db.create_collection(CollectionConfig {
            name: "unused".into(),
            embedding_dim: 2,
            tokenizer: Default::default(),
            distance_metric: DistanceMetric::Cosine,
            embedding_model: Some("x".repeat(128 * 1024)),
            vectors_normalized: false,
            vector_index: VectorIndexConfig::Exact,
        })
        .unwrap();
    }
    let original = fs::read(&path).unwrap();
    for why_not in [false, true] {
        let command = if why_not { "why-not" } else { "search" };
        let reference = run_command(command, args(&path, why_not, vec![])).unwrap();
        let expected = reference.value.clone();
        drop(reference);
        let held = Rc::new(RefCell::new(None));
        let hook_held = Rc::clone(&held);
        let fired = Rc::new(std::cell::Cell::new(0));
        let hook_fired = Rc::clone(&fired);
        let reset = install(
            QueryDtoStage::DatabaseClose,
            Box::new(move |context, _| {
                hook_fired.set(hook_fired.get() + 1);
                let mut blocker = context.cache_reservation(0, usize::MAX).unwrap();
                // Fill the actual parent ledger, without introducing a test DB API
                // or assuming that its Catalog/cache footprint is a fixed size.
                for bit in (0..usize::BITS).rev() {
                    blocker.try_grow((1usize << bit) as u128, usize::MAX);
                }
                assert!(!blocker.try_grow(1, usize::MAX));
                assert_eq!(context.available_workspace_bytes(), 0);
                *hook_held.borrow_mut() = Some(blocker);
            }),
        );
        let result = run_command(command, args(&path, why_not, vec![])).unwrap();
        assert_eq!(fired.get(), 1);
        assert_eq!(result.value, expected);
        assert_retained(&result);
        let context = result.retained.as_ref().unwrap().0.clone();
        let output = success_output(command, result, 0).unwrap();
        assert_eq!(parse(&output).unwrap().get("result"), Some(&expected));
        drop(held.borrow_mut().take());
        assert_eq!(
            context.available_workspace_bytes(),
            JsonProcessingPolicyV1::WORK_BUDGET_BYTES
        );
        drop(reset);
    }
    assert_eq!(fs::read(&path).unwrap(), original);
    fs::remove_file(path).unwrap();
}

#[test]
fn actual_why_not_releases_dead_embedding_and_boost_credit_after_database_close() {
    let (path, _, _) = create_cli_schema_fixture("dto-input-release");
    let embedding_path = temp_path("dto-embedding-input").with_extension("json");
    fs::write(
        &embedding_path,
        format!(
            r#"{{"model_name":"{}","dim":2,"vector":[1,0]}}"#,
            "m".repeat(32769)
        ),
    )
    .unwrap();
    let boost = format!(
        r#"{{"schema_version":"0.1","combine":"sum","max_boost":0.3,"rules":[{{"when":{{"doc.tag":"{}"}},"boost":0.1}}]}}"#,
        "v".repeat(32769)
    );
    let original = fs::read(&path).unwrap();
    for extra in [
        vec!["--embedding".into(), embedding_path.display().to_string()],
        vec!["--metadata-boost".into(), boost],
    ] {
        let reference = run_command("why-not", args(&path, true, extra.clone())).unwrap();
        let expected = reference.value.clone();
        drop(reference);
        let held: Rc<RefCell<Option<WorkspaceReservation>>> = Rc::new(RefCell::new(None));
        let hook_held = Rc::clone(&held);
        let observed = Rc::new(RefCell::new(None));
        let hook_observed = Rc::clone(&observed);
        let reset = install(
            QueryDtoStage::InputRelease,
            Box::new(move |context, work| {
                let mut blocker = context.workspace_reservation();
                let available = context.available_workspace_bytes();
                blocker
                    .grow_to(available, available, "operation_scratch")
                    .unwrap();
                assert_eq!(context.available_workspace_bytes(), 0);
                *hook_observed.borrow_mut() =
                    Some((context.clone(), work.json_bytes(), work.workspace_bytes()));
                *hook_held.borrow_mut() = Some(blocker);
            }),
        );
        let result = run_command("why-not", args(&path, true, extra)).unwrap();
        assert_eq!(result.value, expected);
        assert_retained(&result);
        let (context, json, workspace) = observed.borrow_mut().take().unwrap();
        assert!(json > 0 && workspace > 0);
        assert!(context.available_workspace_bytes() > 0);
        drop(held.borrow_mut().take());
        let output = success_output("why-not", result, 0).unwrap();
        assert_eq!(parse(&output).unwrap().get("result"), Some(&expected));
        assert_eq!(
            context.available_workspace_bytes(),
            JsonProcessingPolicyV1::WORK_BUDGET_BYTES
        );
        drop(reset);
    }
    assert_eq!(fs::read(&path).unwrap(), original);
    fs::remove_file(path).unwrap();
    fs::remove_file(embedding_path).unwrap();
}

#[test]
fn native_dto_conversion_failures_release_both_owned_query_results() {
    let (path, _, chunk_id) = create_cli_schema_fixture("dto-native");
    let original = fs::read(&path).unwrap();
    for why_not in [false, true] {
        let db = Database::open_read_only(&path).unwrap();
        let context = db.json_context();
        let filter = parse(r#"{"$and":[{}]}"#).unwrap();
        let (filter_work, _) = filter.guard_in(JsonInputKind::Filter, &context).unwrap();
        let mut attempts = 0;
        let error = if why_not {
            let (result, mut work) = db
                .why_not_debug_with_metadata_boost_in(
                    "notes",
                    &chunk_id,
                    Some("stable".into()),
                    None,
                    filter,
                    None,
                    false,
                    None,
                    10,
                    None,
                    Some(context.clone()),
                )
                .unwrap();
            work.merge(filter_work);
            drop(db);
            let error = crate::test_alloc::fail_nth_allocation_once(1, || {
                attempts += 1;
                result.into_json_retained(&mut work)
            })
            .unwrap_err();
            drop(work);
            error
        } else {
            let (result, mut work) = db
                .collection("notes")
                .unwrap()
                .search()
                .text("stable")
                .filter(filter)
                .execute_retained(Some(context.clone()))
                .unwrap();
            drop(filter_work);
            drop(db);
            let error = crate::test_alloc::fail_nth_allocation_once(1, || {
                attempts += 1;
                result.into_json_retained(&mut work)
            })
            .unwrap_err();
            drop(work);
            error
        };
        assert_eq!(error.code(), ErrorCode::IoError);
        assert!(error.detail().is_empty());
        assert_eq!(attempts, 1);
        assert_eq!(
            context.available_workspace_bytes(),
            JsonProcessingPolicyV1::WORK_BUDGET_BYTES
        );
    }
    assert_eq!(fs::read(&path).unwrap(), original);
    fs::remove_file(path).unwrap();
}

#[test]
fn retained_filter_moves_into_diagnosis_without_recopying_or_refunding_live_data() {
    let (path, _, chunk_id) = create_cli_schema_fixture("dto-filter-transfer");
    let original = fs::read(&path).unwrap();
    let db = Database::open_read_only(&path).unwrap();
    let context = db.json_context();
    let mut input_work = context.reservation();
    let mut filter_work = context.reservation();
    let mut args = Args::new(vec![
        "--filter".into(),
        format!(r#"{{"doc.tag":"{}"}}"#, "v".repeat(4097)),
    ]);
    let QueryJsonOptions { filter, .. } =
        parse_query_options_in(&mut args, &context, &mut input_work, &mut filter_work).unwrap();
    let pointer = filter.get("doc.tag").unwrap().as_str().unwrap().as_ptr();
    let filter_bytes = layout(&filter);
    assert_eq!(filter_work.reserved_bytes(), filter_bytes);
    assert_eq!(input_work.reserved_bytes(), 0);
    let (result, mut work) = db
        .why_not_debug_with_metadata_boost_in(
            "notes",
            &chunk_id,
            Some("stable".into()),
            None,
            filter,
            None,
            false,
            None,
            10,
            None,
            Some(context.clone()),
        )
        .unwrap();
    let expression = result
        .result
        .get("filter_result")
        .unwrap()
        .get("expression")
        .unwrap();
    assert_eq!(
        expression
            .get("doc.tag")
            .unwrap()
            .as_str()
            .unwrap()
            .as_ptr(),
        pointer
    );
    let copied = work.json_bytes();
    let expected_json = layout(&result.result);
    work.merge(filter_work);
    assert_eq!(work.json_bytes(), copied + filter_bytes);
    assert_eq!(work.json_bytes(), expected_json);
    drop((args, input_work, db));
    assert_eq!(
        context.available_workspace_bytes(),
        JsonProcessingPolicyV1::WORK_BUDGET_BYTES - work.json_bytes() - work.workspace_bytes()
    );
    let value = result.into_json_retained(&mut work).unwrap();
    let expression = value
        .get("filter_result")
        .unwrap()
        .get("expression")
        .unwrap();
    assert_eq!(
        expression
            .get("doc.tag")
            .unwrap()
            .as_str()
            .unwrap()
            .as_ptr(),
        pointer
    );
    assert_eq!(work.json_bytes(), layout(&value));
    drop(value);
    drop(work);
    assert_eq!(
        context.available_workspace_bytes(),
        JsonProcessingPolicyV1::WORK_BUDGET_BYTES
    );
    let fired = Rc::new(std::cell::Cell::new(0));
    let hook_fired = Rc::clone(&fired);
    let reset = install(
        QueryDtoStage::InputRelease,
        Box::new(move |context, work| {
            hook_fired.set(hook_fired.get() + 1);
            assert_eq!(work.json_bytes(), expected_json);
            assert_eq!(
                context.available_workspace_bytes(),
                JsonProcessingPolicyV1::WORK_BUDGET_BYTES - expected_json - work.workspace_bytes()
            );
        }),
    );
    let command = run_command(
        "why-not",
        Args::new(vec![
            path.display().to_string(),
            "--collection".into(),
            "notes".into(),
            "--chunk-id".into(),
            chunk_id,
            "--query".into(),
            "stable".into(),
            "--filter".into(),
            format!(r#"{{"doc.tag":"{}"}}"#, "v".repeat(4097)),
        ]),
    )
    .unwrap();
    assert_eq!(fired.get(), 1);
    assert_retained(&command);
    let output = success_output("why-not", command, 0).unwrap();
    assert_eq!(
        parse(&output)
            .unwrap()
            .get("result")
            .unwrap()
            .get("filter_result")
            .unwrap()
            .get("expression")
            .unwrap()
            .get("doc.tag")
            .unwrap()
            .as_str()
            .unwrap()
            .len(),
        4097
    );
    drop(reset);
    assert_eq!(fs::read(&path).unwrap(), original);
    fs::remove_file(path).unwrap();
}

#[test]
fn filter_guard_survives_native_boost_parse_failure_until_payload_cleanup() {
    let context = JsonOperationContext::standalone();
    let mut input_work = context.reservation();
    let mut filter_work = context.reservation();
    let boost = format!(
        r#"{{"schema_version":"0.1","combine":"sum","max_boost":0.3,"rules":[{{"when":{{"doc.tag":"{}"}},"boost":0.1}}]}}"#,
        "v".repeat(8193)
    );
    let mut args = Args::new(vec![
        "--filter".into(),
        r#"{"$and":[{}]}"#.into(),
        "--metadata-boost".into(),
        boost,
    ]);
    let error = crate::test_alloc::fail_byte_allocation_once(8193, || {
        parse_query_options_in(&mut args, &context, &mut input_work, &mut filter_work)
    })
    .err()
    .unwrap();
    assert_eq!(error.code(), ErrorCode::IoError);
    assert!(filter_work.reserved_bytes() > 0);
    drop((args, filter_work, input_work));
    assert_eq!(
        context.available_workspace_bytes(),
        JsonProcessingPolicyV1::WORK_BUDGET_BYTES
    );
}
