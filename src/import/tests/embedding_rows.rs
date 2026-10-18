use super::*;

#[test]
fn embedding_row_credit_survives_delivery_and_native_conversion_failure_refunds() {
    let context = JsonOperationContext::standalone();
    let available = context.available_workspace_bytes();
    let bytes = b"{\"chunk_external_id\":\"c\",\"model_name\":\"m\",\"dim\":7,\"vector\":[1,1,1,1,1,1,1]}\n";
    let mut retained = None;
    visit_embeddings_jsonl_in(
        Path::new("embeddings.jsonl"),
        &bytes[..],
        &context,
        u64::MAX,
        |_, input, work| {
            assert_eq!(
                work.reserved_bytes(),
                input.chunk_external_id.capacity()
                    + input.model_name.capacity()
                    + input.vector.capacity() * std::mem::size_of::<f32>()
            );
            retained = Some((input, work));
            Ok(())
        },
    )
    .unwrap();
    let (input, work) = retained.take().unwrap();
    assert_eq!(
        context.available_workspace_bytes(),
        available - work.reserved_bytes()
    );
    drop(input);
    drop(work);
    assert_eq!(context.available_workspace_bytes(), available);
    let (_, parsed_allocations) = crate::test_alloc::allocations(|| {
        let mut rows =
            JsonLines::new(&bytes[..], JsonSchema::EmbeddingRow, &context, u64::MAX).unwrap();
        drop(rows.next().unwrap().unwrap());
    });
    let error = crate::test_alloc::fail_nth_allocation_once(parsed_allocations + 1, || {
        visit_embeddings_jsonl_in(
            Path::new("embeddings.jsonl"),
            &bytes[..],
            &context,
            u64::MAX,
            |_, _, _| panic!("native-failed row delivered"),
        )
    })
    .unwrap_err();
    assert_eq!(error.code(), ErrorCode::IoError);
    assert_eq!(detail_value(&error, "path"), Some("embeddings.jsonl"));
    assert_eq!(detail_value(&error, "line"), Some("1"));
    assert_eq!(context.available_workspace_bytes(), available);
}

#[test]
fn embedding_row_string_errors_and_combined_error_order_remain_compatible() {
    let context = JsonOperationContext::standalone();
    for (vector, expected) in [
        ("[1]", ErrorCode::MissingRequiredField),
        ("[\"bad\"]", ErrorCode::InvalidVectorLength),
    ] {
        let input = format!(
            "{{\"chunk_external_id\":\"c\",\"model_name\":7,\"dim\":1,\"vector\":{vector}}}\n"
        );
        let error = visit_embeddings_jsonl_in(
            Path::new("embeddings.jsonl"),
            input.as_bytes(),
            &context,
            u64::MAX,
            |_, _, _| panic!("invalid row delivered"),
        )
        .unwrap_err();
        assert_eq!(error.code(), expected);
        assert_eq!(detail_value(&error, "line"), Some("1"));
    }
}
