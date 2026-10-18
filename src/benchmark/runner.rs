use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

use super::{
    validate_benchmark_result_contract, BenchmarkDataset, BenchmarkQuery, BenchmarkQueryMode,
    QueryEmbeddingSource,
};
use crate::catalog::Catalog;
use crate::db::{Database, DatabaseOpenOptions};
use crate::error::{Error, ErrorCode, Result};
use crate::hash::Sha256Reader;
use crate::import::{visit_chunks_jsonl_in, visit_documents_jsonl_in, visit_embeddings_jsonl_in};
use crate::index::hybrid::fuse;
use crate::index::text::{Bm25Hits, TextCandidates};
use crate::index::vector::{normalize_score, VectorCandidates, VectorHits};
use crate::json::{
    btree_workspace, copy_precharged_string, CatalogReservation, JsonBuilder, JsonInputKind,
    JsonOperationContext, JsonOutputSchema, JsonReservation, JsonValue, ReservedString,
};
use crate::metadata::FilterExpr;
use crate::model::{
    token_count_hint_in, Chunk, DistanceMetric, Document, OnConflict, SearchWeights,
};
use crate::profile::{DatabaseConfig, Durability, ResourceProfile};
use crate::query::{CandidateLimits, SearchResults};
use crate::storage::FORMAT_VERSION;
use crate::temporary::{create_temporary_file_in, TempFileGuard};

mod acceptance;
mod execution;
mod memory;
mod output;
#[cfg(test)]
mod source_probe;
mod sqlite;
#[cfg(test)]
use source_probe::Source as BaselineSource;
#[cfg(not(test))]
type BaselineSource = File;
use memory::process_memory_peak_mb;

pub(crate) struct BenchmarkOptions {
    pub(crate) path: String,
    pub(crate) dataset_path: String,
    pub(crate) collection: String,
    pub(crate) profile: ResourceProfile,
    pub(crate) durability: Durability,
    pub(crate) result_file: Option<String>,
    pub(crate) require_baseline: bool,
}

pub(crate) struct BenchmarkRun {
    pub(crate) value: JsonValue,
    pub(crate) context: JsonOperationContext,
    pub(crate) work: JsonReservation,
}

struct QueryVerification<'a, 'run> {
    expected: &'a acceptance::Acceptance,
    mode: usize,
    journal: Option<&'a mut execution::ExecutionJournal<'run>>,
}

pub(crate) fn run_benchmark(options: BenchmarkOptions) -> Result<BenchmarkRun> {
    // One continuous clock includes input verification, import, reference preparation,
    // every planned query, and verification. It is never reconstructed from phase sums.
    let import_started = Instant::now();
    run_benchmark_started(options, import_started).map_err(|error| {
        error.with_detail(
            "benchmark_elapsed_ms",
            import_started.elapsed().as_secs_f64() * 1000.0,
        )
    })
}

fn run_benchmark_started(
    options: BenchmarkOptions,
    import_started: Instant,
) -> Result<BenchmarkRun> {
    let BenchmarkOptions {
        path,
        dataset_path,
        collection,
        profile,
        durability,
        result_file,
        require_baseline,
    } = options;
    let dataset_context = JsonOperationContext::standalone();
    let (dataset, _dataset_work, input_paths) =
        BenchmarkDataset::load_with_sources_in(&dataset_path, &dataset_context)?;
    let modes = benchmark_query_modes(&dataset);
    let output = result_file
        .as_ref()
        .map(|result_file| {
            output::BenchmarkOutput::new(Path::new(result_file), Path::new(&path), input_paths)
        })
        .transpose()?;
    let mut journal = execution::ExecutionJournal::new(
        &dataset,
        &modes,
        execution::ExecutionConfig {
            database: &path,
            result_file: result_file.as_deref(),
            collection: &collection,
            profile,
            durability,
        },
        &dataset_context,
        import_started,
    )?;
    let result = (|| {
        let db = if PathBuf::from(&path).exists() {
            Database::open_with_options(
                &path,
                DatabaseOpenOptions {
                    resource_profile: Some(profile),
                    durability: Some(durability),
                    ..DatabaseOpenOptions::default()
                },
            )?
        } else {
            Database::create(
                &path,
                DatabaseConfig {
                    durability,
                    resource_profile: profile,
                    ..DatabaseConfig::default()
                },
            )?
        };
        let import_manifest_file =
            write_benchmark_import_manifest(&dataset, &collection, &dataset_context)?;
        let import_manifest_label = "<generated-import-manifest>".to_string();
        let measured_import = db
            .import_manifest_measured_in(
                import_manifest_file.path(),
                OnConflict::Replace,
                crate::db::ImportIndexBuildMode::InlineFullCheckpoint,
                dataset_context.clone(),
            )
            .map_err(|error| dataset.locate_input_error(error))?;
        drop(import_manifest_file);
        journal.import_completed(&measured_import.batches)?;
        let import_summary = measured_import.summary;
        let acceptance = acceptance::Acceptance::prepare(
            &db,
            &collection,
            &dataset,
            profile,
            &modes,
            &dataset_context,
        )
        .map_err(|error| dataset.locate_input_error(error))?;
        journal.expectations(&acceptance)?;
        drop(db);

        let primary_mode = modes
            .iter()
            .copied()
            .find(|mode| *mode == BenchmarkQueryMode::Hybrid)
            .unwrap_or_else(|| modes[0]);
        let mut runs = Vec::new();
        let mut output_work = dataset_context.reservation();
        let mut primary_metrics = None;
        let mut index_size_bytes = 0;
        for (mode_index, mode) in modes.iter().copied().enumerate() {
            let bench_db = Database::open_with_options(
                &path,
                DatabaseOpenOptions {
                    resource_profile: Some(profile),
                    durability: Some(durability),
                    ..DatabaseOpenOptions::default()
                },
            )?;
            if index_size_bytes == 0 {
                index_size_bytes = bench_db.index_stats()?.index_size_bytes;
            }
            let (cold_run, _cold_metrics, cold_work) = run_benchmark_queries(
                &bench_db,
                &collection,
                &dataset,
                profile,
                mode,
                "cold",
                &dataset_context,
                Some(QueryVerification {
                    expected: &acceptance,
                    mode: mode_index,
                    journal: Some(&mut journal),
                }),
            )?;
            output_work.merge(cold_work);
            let (warm_run, warm_metrics, warm_work) = run_benchmark_queries(
                &bench_db,
                &collection,
                &dataset,
                profile,
                mode,
                "warm",
                &dataset_context,
                Some(QueryVerification {
                    expected: &acceptance,
                    mode: mode_index,
                    journal: Some(&mut journal),
                }),
            )?;
            output_work.merge(warm_work);
            output_work.reserve_vec(&mut runs, 2, JsonInputKind::BenchmarkArtifact)?;
            runs.push(cold_run);
            runs.push(warm_run);
            if mode == primary_mode {
                primary_metrics = Some(warm_metrics);
            }
        }
        let import_time_ms = import_started.elapsed().as_secs_f64() * 1000.0;
        drop(acceptance);
        let warm_metrics = primary_metrics.unwrap_or_default();
        let (baseline_outcome, baseline_work) =
            benchmark_baseline_outcome(&dataset, profile, &dataset_context);
        output_work.merge(baseline_work);
        if require_baseline && baseline_outcome.status != "completed" {
            return Err(Error::new(
                ErrorCode::InvalidJsonlSchema,
                "benchmark baseline is required but did not complete",
            )
            .with_detail("field", "baseline_status")
            .with_detail("baseline_status", &baseline_outcome.status)
            .with_detail(
                "baseline_status_reason",
                baseline_outcome.reason.as_deref().unwrap_or(""),
            ));
        }
        let builder = JsonBuilder::budgeted(&mut output_work, JsonInputKind::BenchmarkArtifact);
        let baseline_warnings = benchmark_baseline_warnings_in(&baseline_outcome, &builder)?;
        let import_context = builder.object([
        ("tokenizer", import_summary.tokenizer.build_json(&builder)?),
        (
            "batch_size",
            // Largest actual committed chunk batch, not the profile cap or manifest total.
            JsonValue::Number(measured_import.batches.max_batch_chunks as f64),
        ),
        (
            "index_build_mode",
            builder.string("inline_full_checkpoint")?,
        ),
        ("import_time_ms", JsonValue::Number(import_time_ms)),
        (
            "import_command",
            builder.owned_string(format!(
                "coffer import <benchmark-db> --manifest <generated-import-manifest> --on-conflict replace --durability {}",
                durability.as_str()
            ))?,
        ),
        ("manifest_path", builder.owned_string(import_manifest_label)?),
        ("database_path", builder.string(&path)?),
        ("on_conflict", builder.string("replace")?),
        ("summary", import_summary.build_json(&builder)?),
        ("durability", builder.string(durability.as_str())?),
    ])?;
        let mut metrics = benchmark_metrics_json_in(&warm_metrics, &builder)?;
        builder.insert_if_absent(metrics.as_object_mut().unwrap(), "import_time_ms", || {
            Ok(JsonValue::Number(import_time_ms))
        })?;
        builder.insert_if_absent(metrics.as_object_mut().unwrap(), "index_size_mb", || {
            Ok(JsonValue::Number(index_size_bytes as f64 / 1_048_576.0))
        })?;
        let result = builder.object([
            ("schema_version", builder.string("0.1")?),
            (
                "cofferdb_version",
                builder.string(env!("CARGO_PKG_VERSION"))?,
            ),
            ("git_commit", benchmark_git_commit_json_in(&builder)?),
            (
                "file_format_version",
                JsonValue::Number(FORMAT_VERSION as f64),
            ),
            ("hardware", benchmark_hardware_json_in(&builder)?),
            ("status", builder.string("completed_cofferdb_runs")?),
            ("path", builder.owned_string(path)?),
            ("result_file", builder.optional_str(result_file.as_deref())?),
            ("dataset_path", builder.owned_string(dataset_path)?),
            ("dataset_id", builder.string(&dataset.manifest.dataset_id)?),
            ("collection", builder.owned_string(collection)?),
            (
                "dataset",
                dataset.manifest.build_result_dataset_json(&builder)?,
            ),
            ("profile", builder.string(profile.as_json_str())?),
            ("profile_params", profile_params_json_in(profile, &builder)?),
            ("engine_params", benchmark_engine_params_json_in(&builder)?),
            (
                "candidate_counts",
                builder.object([
                    (
                        "bm25",
                        JsonValue::Number(profile.params().bm25_candidates as f64),
                    ),
                    (
                        "vector",
                        JsonValue::Number(profile.params().vector_candidates as f64),
                    ),
                    (
                        "fusion",
                        JsonValue::Number(profile.params().fusion_candidates as f64),
                    ),
                ])?,
            ),
            ("durability", builder.string(durability.as_str())?),
            (
                "index_build_mode",
                builder.string("inline_full_checkpoint")?,
            ),
            ("import_context", import_context),
            ("metrics", metrics),
            ("warnings", baseline_warnings),
            ("runs", JsonValue::Array(runs)),
            (
                "baseline_configs",
                sqlite_baseline_configs_in(
                    &dataset,
                    baseline_outcome.status == "completed",
                    &builder,
                )?,
            ),
            (
                "baseline_results",
                JsonValue::Array(baseline_outcome.results),
            ),
            (
                "baseline_status",
                builder.owned_string(baseline_outcome.status)?,
            ),
            (
                "baseline_status_reason",
                builder.optional_string(baseline_outcome.reason)?,
            ),
        ])?;
        validate_benchmark_result_contract(&result)?;
        let schema = JsonOutputSchema::Value(JsonInputKind::BenchmarkArtifact);
        result.reconcile_output_retained_in(schema, &mut output_work)?;
        Ok(BenchmarkRun {
            value: result,
            context: dataset_context,
            work: output_work,
        })
    })();
    match result {
        Ok(run) => journal.publish_run(run, output),
        Err(error) => Err(journal.fail(error)),
    }
}

#[derive(Clone, Debug, Default)]
struct BenchmarkRunMetrics {
    recall_at_5: f64,
    recall_at_10: f64,
    mrr: f64,
    p95_ms: f64,
    memory_peak_mb: Option<f64>,
    vector_scan_count: usize,
    distance_ops: usize,
    fusion_candidate_count: usize,
}

#[derive(Clone, Copy, Debug, Default)]
struct BenchmarkExecutionStats {
    vector_scan_count: usize,
    distance_ops: usize,
    fusion_candidate_count: usize,
}

impl BenchmarkExecutionStats {
    #[cfg(test)]
    fn to_json(self) -> JsonValue {
        self.build_json(&JsonBuilder::host())
            .expect("host DTO construction is infallible")
    }

    fn build_json(self, builder: &JsonBuilder<'_>) -> Result<JsonValue> {
        builder.object([
            (
                "vector_scan_count",
                JsonValue::Number(self.vector_scan_count as f64),
            ),
            ("distance_ops", JsonValue::Number(self.distance_ops as f64)),
            (
                "fusion_candidate_count",
                JsonValue::Number(self.fusion_candidate_count as f64),
            ),
        ])
    }
}

fn benchmark_query_modes(dataset: &BenchmarkDataset) -> Vec<BenchmarkQueryMode> {
    let all_have_text = dataset
        .queries
        .iter()
        .all(|query| !query.text.as_deref().unwrap_or("").trim().is_empty());
    let all_have_embedding = dataset
        .queries
        .iter()
        .all(|query| query.embedding.is_some());
    let mut modes = Vec::new();
    if all_have_text {
        modes.push(BenchmarkQueryMode::TextBm25);
    }
    if all_have_embedding {
        modes.push(BenchmarkQueryMode::VectorExact);
    }
    if all_have_text && all_have_embedding {
        modes.push(BenchmarkQueryMode::Hybrid);
    }
    if modes.is_empty() {
        modes.push(BenchmarkQueryMode::TextBm25);
    }
    modes
}

fn write_benchmark_import_manifest(
    dataset: &BenchmarkDataset,
    collection: &str,
    context: &JsonOperationContext,
) -> Result<TempFileGuard> {
    let mut work = context.reservation();
    let builder = JsonBuilder::budgeted(&mut work, JsonInputKind::Manifest);
    let mut manifest = builder.object([
        (
            "tokenizer",
            dataset.manifest.tokenizer.build_json(&builder)?,
        ),
        ("schema_version", builder.string("0.1")?),
        ("dataset_id", builder.string(&dataset.manifest.dataset_id)?),
        ("collection", builder.string(collection)?),
        (
            "documents",
            builder.string(&canonical_path_string(&dataset.documents_path)?)?,
        ),
        (
            "chunks",
            builder.string(&canonical_path_string(&dataset.chunks_path)?)?,
        ),
        (
            "embedding_model",
            builder.optional_str(dataset.manifest.embedding_model.as_deref())?,
        ),
        (
            "embedding_dim",
            JsonValue::Number(dataset.manifest.embedding_dim as f64),
        ),
        ("distance_metric", builder.string("cosine")?),
        (
            "hashes",
            builder.object([
                (
                    "documents_sha256",
                    builder.string(&dataset.manifest.documents_sha256)?,
                ),
                (
                    "chunks_sha256",
                    builder.string(&dataset.manifest.chunks_sha256)?,
                ),
            ])?,
        ),
    ])?;
    if let Some(embeddings_path) = &dataset.embeddings_path {
        builder.insert_if_absent(manifest.as_object_mut().unwrap(), "embeddings", || {
            builder.string(&canonical_path_string(embeddings_path)?)
        })?;
        if let Some(obj) = manifest.as_object_mut() {
            let hashes = obj
                .get_mut("hashes")
                .and_then(JsonValue::as_object_mut)
                .expect("benchmark import manifest hashes is an object");
            if let Some(hash) = &dataset.manifest.embeddings_sha256 {
                builder.insert_if_absent(hashes, "embeddings_sha256", || builder.string(hash))?;
            }
        }
    }
    let mut output_work = context.reservation();
    output_work.grow(8192, JsonInputKind::Manifest)?;
    let (file, guard) = create_temporary_file("benchmark-import", "json")?;
    let mut file = std::io::BufWriter::with_capacity(8192, file);
    manifest.write_value_retained_in(
        &mut file,
        JsonInputKind::Manifest,
        context,
        &mut work,
        true,
    )?;
    file.write_all(b"\n")?;
    file.flush()?;
    Ok(guard)
}

fn canonical_path_string(path: &Path) -> Result<String> {
    Ok(fs::canonicalize(path)?.display().to_string())
}

#[allow(clippy::too_many_arguments)]
fn run_benchmark_queries(
    db: &Database,
    collection: &str,
    dataset: &BenchmarkDataset,
    profile: ResourceProfile,
    mode: BenchmarkQueryMode,
    cache_state: &str,
    context: &JsonOperationContext,
    mut acceptance: Option<QueryVerification<'_, '_>>,
) -> Result<(JsonValue, BenchmarkRunMetrics, JsonReservation)> {
    let collection = db.collection(collection)?;
    let mut work = context.reservation();
    let mut scratch = context.reservation();
    let mut per_query = Vec::new();
    let mut latencies = Vec::new();
    let mut recall_at_5_sum = 0.0;
    let mut recall_at_10_sum = 0.0;
    let mut reciprocal_rank_sum = 0.0;
    let mut vector_scan_count = 0;
    let mut distance_ops = 0;
    let mut fusion_candidate_count = 0;

    for (query_index, query) in dataset.queries.iter().enumerate() {
        let (text, embedding) = match mode {
            BenchmarkQueryMode::TextBm25 => {
                let text = query.text.as_ref().ok_or_else(|| {
                    Error::new(
                        ErrorCode::InvalidJsonlSchema,
                        "text benchmark mode requires query text",
                    )
                    .with_detail("query_id", &query.query_id)
                })?;
                (Some(text.as_str()), None)
            }
            BenchmarkQueryMode::VectorExact => {
                let embedding = query.embedding.as_ref().ok_or_else(|| {
                    Error::new(
                        ErrorCode::InvalidJsonlSchema,
                        "vector benchmark mode requires query embedding",
                    )
                    .with_detail("query_id", &query.query_id)
                })?;
                (None, Some(embedding))
            }
            BenchmarkQueryMode::Hybrid => {
                let text = query.text.as_ref().ok_or_else(|| {
                    Error::new(
                        ErrorCode::InvalidJsonlSchema,
                        "hybrid benchmark mode requires query text",
                    )
                    .with_detail("query_id", &query.query_id)
                })?;
                let embedding = query.embedding.as_ref().ok_or_else(|| {
                    Error::new(
                        ErrorCode::InvalidJsonlSchema,
                        "hybrid benchmark mode requires query embedding",
                    )
                    .with_detail("query_id", &query.query_id)
                })?;
                (Some(text.as_str()), Some(embedding))
            }
        };
        if let Some(verification) = acceptance.as_mut() {
            if let Some(journal) = &mut verification.journal {
                journal.begin(verification.mode, cache_state, query_index)?;
            }
        }
        let trace = acceptance
            .as_ref()
            .and_then(|verification| verification.journal.as_ref())
            .map(|_| crate::search_cost::SearchCostTrace::new());
        let started = Instant::now();
        let search = db.search_in_collection_traced(
            collection.collection_id(),
            text,
            embedding,
            &query.filter,
            None,
            false,
            None,
            query.top_k,
            Some(profile),
            Some(context.clone()),
            trace.as_ref(),
        );
        let latency_ms = started.elapsed().as_secs_f64() * 1000.0;
        if let Some(trace) = &trace {
            trace.finish();
        }
        if let Some(verification) = acceptance.as_mut() {
            if let Some(journal) = &mut verification.journal {
                if let Some(trace) = trace {
                    if let Err(record_error) =
                        journal.search_finished(trace, latency_ms, search.is_ok())
                    {
                        return Err(match search {
                            Err(error) => error
                                .with_detail("execution_record_error", record_error.to_string()),
                            Ok(_) => record_error,
                        });
                    }
                }
            }
        }
        let (mut results, mut result_work) = search?;
        if let Some(verification) = acceptance.as_ref() {
            verification
                .expected
                .verify(verification.mode, query_index, &results)
                .map_err(|error| {
                    error
                        .with_detail("query_id", &query.query_id)
                        .with_detail("query_mode", mode.as_str())
                        .with_detail("cache_state", cache_state)
                })?;
        }
        if let Some(verification) = acceptance.as_mut() {
            if let Some(journal) = &mut verification.journal {
                journal.complete_query(latency_ms, &results)?;
            }
        }
        let ranked_chunk_ids = || {
            results
                .results
                .iter()
                .map(|result| result.chunk_external_id.as_str())
        };
        let hit = ranked_chunk_ids().any(|chunk_id| query.relevant_chunk_ids.contains(chunk_id));
        let recall_at_5 = recall_at_k(ranked_chunk_ids(), query, 5);
        let recall_at_10 = recall_at_k(ranked_chunk_ids(), query, 10);
        let reciprocal_rank = reciprocal_rank(ranked_chunk_ids(), query);
        let stats = benchmark_execution_stats(&results);

        scratch.reserve_vec(&mut latencies, 1, JsonInputKind::BenchmarkArtifact)?;
        latencies.push(latency_ms);
        recall_at_5_sum += recall_at_5;
        recall_at_10_sum += recall_at_10;
        reciprocal_rank_sum += reciprocal_rank;
        vector_scan_count += stats.vector_scan_count;
        distance_ops += stats.distance_ops;
        fusion_candidate_count += stats.fusion_candidate_count;

        work.reserve_vec(&mut per_query, 1, JsonInputKind::BenchmarkArtifact)?;
        let builder = JsonBuilder::budgeted(&mut work, JsonInputKind::BenchmarkArtifact);
        let scores = builder.array(
            results
                .results
                .iter()
                .map(|result| Ok(JsonValue::Number(result.final_score))),
        )?;
        let ranked_chunk_ids = {
            let (_, strings) = result_work.parts_mut();
            let builder = JsonBuilder::budgeted_owned_strings(
                &mut work,
                strings,
                JsonInputKind::BenchmarkArtifact,
            );
            builder.array(
                std::mem::take(&mut results.results)
                    .into_iter()
                    .map(|result| builder.owned_string(result.chunk_external_id)),
            )?
        };
        results.discard_rows(&mut result_work);
        let query_context = results.query_context;
        drop((results.schema_version, results.collection, results.query));
        result_work.retain_workspace(query_context.workspace_bytes());
        let (_, strings) = result_work.parts_mut();
        let builder = JsonBuilder::budgeted_owned_strings(
            &mut work,
            strings,
            JsonInputKind::BenchmarkArtifact,
        );
        per_query.push(builder.object([
            ("query_id", builder.string(&query.query_id)?),
            ("latency_ms", JsonValue::Number(latency_ms)),
            ("ranked_chunk_ids", ranked_chunk_ids),
            ("score_kind", builder.string(mode.score_kind())?),
            ("score_scale", builder.string("engine_local")?),
            ("score_comparable", JsonValue::Bool(false)),
            ("scores", scores),
            ("hit", JsonValue::Bool(hit)),
            ("execution_stats", stats.build_json(&builder)?),
            ("query_context", query_context.build_json(&builder)?),
        ])?);
    }

    let query_count = dataset.queries.len() as f64;
    let metrics = BenchmarkRunMetrics {
        recall_at_5: recall_at_5_sum / query_count,
        recall_at_10: recall_at_10_sum / query_count,
        mrr: reciprocal_rank_sum / query_count,
        p95_ms: percentile_95_in_place(&mut latencies),
        memory_peak_mb: process_memory_peak_mb(),
        vector_scan_count,
        distance_ops,
        fusion_candidate_count,
    };
    let builder = JsonBuilder::budgeted(&mut work, JsonInputKind::BenchmarkArtifact);
    let value = builder.object([
        ("engine", builder.string("cofferdb")?),
        ("query_mode", builder.string(mode.as_str())?),
        ("cache_state", builder.string(cache_state)?),
        ("metrics", benchmark_metrics_json_in(&metrics, &builder)?),
        ("per_query", JsonValue::Array(per_query)),
    ])?;
    Ok((value, metrics, work))
}

#[cfg(test)]
fn benchmark_metrics_json(metrics: &BenchmarkRunMetrics) -> JsonValue {
    benchmark_metrics_json_in(metrics, &JsonBuilder::host())
        .expect("host DTO construction is infallible")
}

fn benchmark_metrics_json_in(
    metrics: &BenchmarkRunMetrics,
    builder: &JsonBuilder<'_>,
) -> Result<JsonValue> {
    builder.object([
        ("recall_at_5", JsonValue::Number(metrics.recall_at_5)),
        ("recall_at_10", JsonValue::Number(metrics.recall_at_10)),
        ("mrr", JsonValue::Number(metrics.mrr)),
        ("p95_ms", JsonValue::Number(metrics.p95_ms)),
        (
            "memory_peak_mb",
            metrics
                .memory_peak_mb
                .map(JsonValue::Number)
                .unwrap_or(JsonValue::Null),
        ),
        (
            "vector_scan_count",
            JsonValue::Number(metrics.vector_scan_count as f64),
        ),
        (
            "distance_ops",
            JsonValue::Number(metrics.distance_ops as f64),
        ),
        (
            "fusion_candidate_count",
            JsonValue::Number(metrics.fusion_candidate_count as f64),
        ),
    ])
}

fn recall_at_k<'a>(
    ranked_chunk_ids: impl IntoIterator<Item = &'a str>,
    query: &BenchmarkQuery,
    k: usize,
) -> f64 {
    if query.relevant_chunk_ids.is_empty() {
        return 0.0;
    }
    let hits = ranked_chunk_ids
        .into_iter()
        .take(k)
        .filter(|chunk_id| query.relevant_chunk_ids.contains(*chunk_id))
        .count();
    hits as f64 / query.relevant_chunk_ids.len() as f64
}

fn reciprocal_rank<'a>(
    ranked_chunk_ids: impl IntoIterator<Item = &'a str>,
    query: &BenchmarkQuery,
) -> f64 {
    ranked_chunk_ids
        .into_iter()
        .position(|chunk_id| query.relevant_chunk_ids.contains(chunk_id))
        .map(|index| 1.0 / (index + 1) as f64)
        .unwrap_or(0.0)
}

fn percentile_95_in_place(values: &mut [f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.sort_by(f64::total_cmp);
    let index = ((values.len() as f64 * 0.95).ceil() as usize).saturating_sub(1);
    values[index.min(values.len() - 1)]
}

fn benchmark_execution_stats(results: &SearchResults) -> BenchmarkExecutionStats {
    let stats = results.query_context.execution_stats;
    BenchmarkExecutionStats {
        vector_scan_count: stats.vector_scan_count,
        distance_ops: stats.distance_ops,
        fusion_candidate_count: stats.fusion_candidate_count,
    }
}

#[cfg(test)]
fn benchmark_hardware_json() -> JsonValue {
    benchmark_hardware_json_in(&JsonBuilder::host()).expect("host DTO construction is infallible")
}

fn benchmark_hardware_json_in(builder: &JsonBuilder<'_>) -> Result<JsonValue> {
    let cpu_model = hardware_cpu_model();
    let cpu_cores = hardware_cpu_cores();
    let (os_name, os_version) = hardware_os_name_version();
    builder.object([
        (
            "cpu",
            builder.owned_string(format!("{cpu_model}; {cpu_cores} cores"))?,
        ),
        (
            "memory_gb",
            hardware_memory_gb()
                .map(JsonValue::Number)
                .unwrap_or(JsonValue::Null),
        ),
        ("storage", builder.string("local-storage")?),
        (
            "os",
            builder.owned_string(format!("{os_name} {os_version}"))?,
        ),
    ])
}

fn benchmark_git_commit_json_in(builder: &JsonBuilder<'_>) -> Result<JsonValue> {
    // The caller's working directory and environment do not identify this binary's source.
    let commit = option_env!("COFFERDB_GIT_COMMIT")
        .filter(|value| !value.trim().is_empty())
        .unwrap_or("unknown");
    builder.string(commit)
}

fn sqlite_baseline_configs_in(
    dataset: &BenchmarkDataset,
    baseline_completed: bool,
    builder: &JsonBuilder<'_>,
) -> Result<JsonValue> {
    let requires_query_embedding =
        dataset.manifest.query_embedding_source != QueryEmbeddingSource::None;
    let name = sqlite_baseline_name(dataset);
    let filter_strategy = if requires_query_embedding {
        "metadata_filter_before_vector_scoring"
    } else {
        "metadata_filter_after_candidate_fetch"
    };
    let max_top_k = dataset
        .queries
        .iter()
        .map(|query| query.top_k)
        .max()
        .unwrap_or(0);
    builder.array(std::iter::once_with(|| {
        builder.object([
            ("name", builder.string(name)?),
            ("version", builder.optional_string(sqlite3_version())?),
            ("run_mode", builder.string("local_cli")?),
            ("mode", builder.string("local_cli")?),
            (
                "requires_query_embedding",
                JsonValue::Bool(requires_query_embedding),
            ),
            ("top_k", JsonValue::Number(max_top_k as f64)),
            (
                "cache_states",
                builder.array(
                    ["cold", "warm"]
                        .into_iter()
                        .map(|value| builder.string(value)),
                )?,
            ),
            (
                "config_source",
                builder.string("benchmarks/baselines/sqlite/README.md")?,
            ),
            (
                "checked_at",
                if baseline_completed {
                    builder.owned_string(benchmark_checked_at_date())?
                } else {
                    JsonValue::Null
                },
            ),
            (
                "config",
                builder.object([
                    ("engine", builder.string("sqlite")?),
                    (
                        "sqlite_cli",
                        builder.string(sqlite_cli_label(&sqlite_cli())?)?,
                    ),
                    ("fts", builder.string("fts5")?),
                    ("fts_table", builder.string("chunks_fts")?),
                    ("fts_tokenizer", builder.string("default")?),
                    ("fts_score", builder.string("bm25(chunks_fts)")?),
                    (
                        "cache_policy",
                        builder.string(
                            "cold=reopened_handle_no_os_cache_eviction;warm=repeated_query",
                        )?,
                    ),
                    (
                        "latency_scope",
                        builder.string(
                            "sqlite_cli_process_startup_included;cofferdb_process_startup_excluded",
                        )?,
                    ),
                    (
                        "vector_extension",
                        if requires_query_embedding {
                            builder.string("sqlite-vec")?
                        } else {
                            JsonValue::Null
                        },
                    ),
                    (
                        "vector_mode",
                        if requires_query_embedding {
                            builder.string("exact_scan")?
                        } else {
                            JsonValue::Null
                        },
                    ),
                    (
                        "vector_extension_version",
                        if requires_query_embedding {
                            builder.optional_string(
                                sqlite3_script(Path::new(":memory:"), "SELECT vec_version();")
                                    .ok()
                                    .map(|value| value.trim().to_owned()),
                            )?
                        } else {
                            JsonValue::Null
                        },
                    ),
                    (
                        "fusion",
                        if requires_query_embedding {
                            builder.string("weighted_sum")?
                        } else {
                            builder.string("text_only_bm25")?
                        },
                    ),
                    ("filter", builder.string(filter_strategy)?),
                    (
                        "query_count",
                        JsonValue::Number(dataset.queries.len() as f64),
                    ),
                    (
                        "status",
                        builder.string(if baseline_completed {
                            "completed"
                        } else {
                            "not_run"
                        })?,
                    ),
                ])?,
            ),
        ])
    }))
}

fn sqlite_baseline_name(dataset: &BenchmarkDataset) -> &'static str {
    if dataset.manifest.query_embedding_source != QueryEmbeddingSource::None {
        "sqlite-fts5-sqlite-vec"
    } else {
        "sqlite-fts5-text-only"
    }
}

fn benchmark_checked_at_date() -> String {
    crate::time::now_rfc3339()
        .chars()
        .take(10)
        .collect::<String>()
}

fn benchmark_baseline_warnings_in(
    outcome: &BenchmarkBaselineOutcome,
    builder: &JsonBuilder<'_>,
) -> Result<JsonValue> {
    if outcome.status == "completed" {
        return Ok(JsonValue::Array(Vec::new()));
    }
    builder.array(std::iter::once_with(|| {
        builder.object([
            ("code", builder.string("benchmark_baseline_not_run")?),
            ("message", builder.string("benchmark baseline did not run")?),
            ("reason", builder.optional_str(outcome.reason.as_deref())?),
        ])
    }))
}

fn sqlite3_version() -> Option<String> {
    sqlite3_script(Path::new(":memory:"), "SELECT sqlite_version();")
        .ok()
        .map(|version| version.trim().to_owned())
}

#[derive(Clone, Debug)]
struct BenchmarkBaselineOutcome {
    status: String,
    reason: Option<String>,
    results: Vec<JsonValue>,
}

impl BenchmarkBaselineOutcome {
    fn completed(results: Vec<JsonValue>) -> Self {
        Self {
            status: "completed".into(),
            reason: None,
            results,
        }
    }

    fn not_run(reason: impl Into<String>) -> Self {
        Self {
            status: "not_run".into(),
            reason: Some(reason.into()),
            results: Vec::new(),
        }
    }
}

struct SqliteBaselineData {
    catalog: Catalog,
    chunk_ids_by_external_id: BTreeMap<String, String>,
    json_context: JsonOperationContext,
    _rows_work: CatalogReservation,
}

fn benchmark_baseline_outcome(
    dataset: &BenchmarkDataset,
    profile: ResourceProfile,
    context: &JsonOperationContext,
) -> (BenchmarkBaselineOutcome, JsonReservation) {
    if !sqlite3_fts5_available() {
        return (
            BenchmarkBaselineOutcome::not_run("sqlite3 with FTS5 is unavailable"),
            context.reservation(),
        );
    }
    if dataset.manifest.query_embedding_source != QueryEmbeddingSource::None
        && !sqlite3_sqlite_vec_available()
    {
        return (BenchmarkBaselineOutcome::not_run("sqlite-vec is unavailable; set COFFERDB_SQLITE_VEC_EXTENSION to a trusted local extension"), context.reservation());
    }
    match run_sqlite_baseline(dataset, profile, context)
        .map_err(|error| dataset.locate_input_error(error))
    {
        Ok((results, work)) => (BenchmarkBaselineOutcome::completed(results), work),
        Err(err) => (
            BenchmarkBaselineOutcome::not_run(format!("sqlite baseline failed: {err}")),
            context.reservation(),
        ),
    }
}

fn run_sqlite_baseline(
    dataset: &BenchmarkDataset,
    profile: ResourceProfile,
    context: &JsonOperationContext,
) -> Result<(Vec<JsonValue>, JsonReservation)> {
    let baseline_data = read_sqlite_baseline_data(dataset, context)?;
    let (file, db_file) = create_temporary_file("sqlite-baseline", "sqlite")?;
    drop(file);
    build_sqlite_fts5_database(db_file.path(), &baseline_data.catalog)?;
    if dataset.manifest.query_embedding_source != QueryEmbeddingSource::None {
        build_sqlite_vector_database(db_file.path(), dataset, context)?;
    }
    let modes = benchmark_query_modes(dataset);
    let mut work = context.reservation();
    let mut results = Vec::new();
    work.reserve_vec(
        &mut results,
        modes.len() * 2,
        JsonInputKind::BenchmarkArtifact,
    )?;
    for mode in modes {
        let cold = run_sqlite_baseline_queries(
            db_file.path(),
            dataset,
            &baseline_data,
            profile,
            mode,
            "cold",
            &mut work,
        )?;
        let warm = run_sqlite_baseline_queries(
            db_file.path(),
            dataset,
            &baseline_data,
            profile,
            mode,
            "warm",
            &mut work,
        )?;
        results.push(cold);
        results.push(warm);
    }
    Ok((results, work))
}

fn create_temporary_file(label: &str, extension: &str) -> Result<(File, TempFileGuard)> {
    create_temporary_file_in(&std::env::temp_dir(), label, extension)
}

fn read_sqlite_baseline_data(
    dataset: &BenchmarkDataset,
    context: &JsonOperationContext,
) -> Result<SqliteBaselineData> {
    let mut work = context.catalog_reservation();
    let documents = read_verified_baseline_input(
        &dataset.documents_path,
        "documents",
        &dataset.manifest.documents_sha256,
        |source| {
            read_sqlite_baseline_documents(&dataset.documents_path, source, context, &mut work)
        },
    )?;
    let (mut catalog, chunk_ids_by_external_id) = read_verified_baseline_input(
        &dataset.chunks_path,
        "chunks",
        &dataset.manifest.chunks_sha256,
        |source| {
            read_sqlite_baseline_chunks(
                &dataset.chunks_path,
                source,
                &documents,
                context,
                &mut work,
            )
        },
    )?;
    if dataset.manifest.query_embedding_source == QueryEmbeddingSource::None {
        visit_sqlite_baseline_embeddings(dataset, context, |_| Ok(()))?;
    }
    install_baseline_documents(&mut catalog, documents, &mut work)?;
    Ok(SqliteBaselineData {
        catalog,
        chunk_ids_by_external_id,
        json_context: context.clone(),
        _rows_work: work,
    })
}

fn read_verified_baseline_input<T>(
    path: &Path,
    kind: &str,
    expected: &str,
    read: impl FnOnce(&mut Sha256Reader<BaselineSource>) -> Result<T>,
) -> Result<T> {
    let source = File::open(path)?;
    #[cfg(test)]
    let source = source_probe::Source::new(source, kind);
    let mut source = Sha256Reader::new(source);
    let value = read(&mut source)?;
    let actual = source.finish();
    if actual != expected {
        return Err(
            Error::new(ErrorCode::ChecksumMismatch, format!("{kind} hash mismatch"))
                .with_detail("field", format!("hashes.{kind}_sha256"))
                .with_detail("artifact_kind", kind)
                .with_detail("path", path.display())
                .with_detail("expected_sha256", expected)
                .with_detail("actual_sha256", actual),
        );
    }
    #[cfg(test)]
    source_probe::verified(kind);
    Ok(value)
}

fn visit_sqlite_baseline_embeddings(
    dataset: &BenchmarkDataset,
    context: &JsonOperationContext,
    mut consume: impl FnMut(crate::EmbeddingInput) -> Result<()>,
) -> Result<()> {
    let Some(path) = &dataset.embeddings_path else {
        return Ok(());
    };
    let expected = dataset
        .manifest
        .embeddings_sha256
        .as_deref()
        .ok_or_else(|| {
            Error::new(
                ErrorCode::MissingRequiredField,
                "benchmark embeddings require a hash",
            )
            .with_detail("field", "hashes.embeddings_sha256")
        })?;
    read_verified_baseline_input(path, "embeddings", expected, |source| {
        visit_embeddings_jsonl_in(path, source, context, u64::MAX, |_, embedding, work| {
            let (_work, embedding) = (work, embedding);
            consume(embedding)
        })
    })
}

fn read_sqlite_baseline_documents(
    path: &Path,
    source: impl Read,
    context: &JsonOperationContext,
    retained: &mut CatalogReservation,
) -> Result<BTreeMap<String, Box<Document>>> {
    let before = retained.reserved_bytes();
    let result = (|| {
        let mut documents = BTreeMap::new();
        visit_documents_jsonl_in(path, source, context, u64::MAX, |line_number, row, work| {
            let (mut work, row) = (work, row);
            if documents.contains_key(&row.external_id) {
                return Err(Error::new(
                    ErrorCode::DuplicateExternalId,
                    "duplicate document external_id in benchmark documents",
                )
                .with_detail("external_id", &row.external_id));
            }
            let (document_id, id_work) = ReservedString::build(context, "catalog_build", |out| {
                write!(out, "doc_{line_number}")
            })?
            .into_parts();
            work.merge_workspace(id_work);
            let additional = btree_workspace::<String, Box<Document>>(documents.len() + 1)
                - btree_workspace::<String, Box<Document>>(documents.len())
                + std::mem::size_of::<Document>()
                + row.external_id.len()
                + "col_1".len();
            work.transfer_to_catalog(retained, work.json_bytes(), work.workspace_bytes())?;
            retained.grow(additional)?;
            documents.insert(
                copy_precharged_string(&row.external_id)?,
                Box::new(Document {
                    document_id,
                    collection_id: copy_precharged_string("col_1")?,
                    external_id: row.external_id,
                    title: row.title,
                    source_uri: row.source_uri,
                    metadata: row.metadata,
                    created_at: String::new(),
                    updated_at: String::new(),
                }),
            );
            Ok(())
        })?;
        if documents.is_empty() {
            return Err(Error::new(
                ErrorCode::MissingRequiredField,
                "benchmark documents must not be empty",
            )
            .with_detail("path", path.display()));
        }
        Ok(documents)
    })();
    if result.is_err() {
        // The failed file's partial map is gone; earlier source files stay owned.
        retained.retain(before);
    }
    result
}

fn read_sqlite_baseline_chunks(
    path: &Path,
    source: impl Read,
    documents: &BTreeMap<String, Box<Document>>,
    context: &JsonOperationContext,
    retained: &mut CatalogReservation,
) -> Result<(Catalog, BTreeMap<String, String>)> {
    let before = retained.reserved_bytes();
    let result = (|| {
        let mut catalog = Catalog::default();
        let mut chunk_ids_by_external_id = BTreeMap::new();
        visit_chunks_jsonl_in(path, source, context, u64::MAX, |line_number, row, work| {
            let (mut work, row) = (work, row);
            let document = documents.get(&row.document_external_id).ok_or_else(|| {
                Error::new(
                    ErrorCode::DocumentNotFound,
                    "benchmark chunk references unknown document_external_id",
                )
                .with_detail("document_external_id", &row.document_external_id)
            })?;
            drop(row.document_external_id);
            work.retain_workspace(
                row.chunk.chunk_external_id.capacity() + row.chunk.text.capacity(),
            );
            let crate::ChunkInput {
                chunk_external_id,
                text,
                ordinal,
                metadata,
                source_span,
            } = row.chunk;
            if chunk_ids_by_external_id.contains_key(&chunk_external_id) {
                return Err(Error::new(
                    ErrorCode::DuplicateExternalId,
                    "duplicate chunk_external_id in benchmark chunks",
                )
                .with_detail("chunk_external_id", &chunk_external_id));
            }
            let token_count = token_count_hint_in(&text, context)?;
            let (chunk_id, id_work) = ReservedString::build(context, "catalog_build", |out| {
                write!(out, "chk_{line_number}")
            })?
            .into_parts();
            work.merge_workspace(id_work);
            let (text_hash, hash_work) =
                ReservedString::build_exact(64, context, "catalog_build", |out| {
                    crate::hash::write_digest_hex(out, crate::hash::sha256(text.as_bytes()))
                })?
                .into_parts();
            work.merge_workspace(hash_work);
            let entries = catalog.chunks.len();
            let additional = btree_workspace::<String, Box<Chunk>>(entries + 1)
                - btree_workspace::<String, Box<Chunk>>(entries)
                + btree_workspace::<String, String>(entries + 1)
                - btree_workspace::<String, String>(entries)
                + std::mem::size_of::<Chunk>()
                + 2 * chunk_id.len()
                + document.document_id.len()
                + "col_1".len()
                + chunk_external_id.len();
            work.transfer_to_catalog(retained, work.json_bytes(), work.workspace_bytes())?;
            retained.grow(additional)?;
            let chunk = Chunk {
                chunk_id: copy_precharged_string(&chunk_id)?,
                document_id: copy_precharged_string(&document.document_id)?,
                collection_id: copy_precharged_string("col_1")?,
                chunk_external_id: copy_precharged_string(&chunk_external_id)?,
                text_hash,
                token_count,
                text,
                ordinal,
                metadata,
                embedding_id: None,
                source_span,
                created_at: String::new(),
            };
            catalog
                .chunks
                .insert(copy_precharged_string(&chunk_id)?, Box::new(chunk));
            chunk_ids_by_external_id.insert(chunk_external_id, chunk_id);
            Ok(())
        })?;
        if catalog.chunks.is_empty() {
            return Err(Error::new(
                ErrorCode::MissingRequiredField,
                "benchmark chunks must not be empty",
            )
            .with_detail("path", path.display()));
        }
        Ok((catalog, chunk_ids_by_external_id))
    })();
    if result.is_err() {
        retained.retain(before);
    }
    result
}

fn install_baseline_documents(
    catalog: &mut Catalog,
    documents: BTreeMap<String, Box<Document>>,
    retained: &mut CatalogReservation,
) -> Result<()> {
    let old_keys = documents.keys().map(String::capacity).sum::<usize>();
    let new_keys = documents
        .values()
        .map(|doc| doc.document_id.len())
        .sum::<usize>();
    let tree = btree_workspace::<String, Box<Document>>(documents.len());
    // Both trees and their distinct keys coexist while the boxes are moved.
    retained.grow(tree + new_keys)?;
    for (_, document) in documents {
        catalog
            .documents
            .insert(copy_precharged_string(&document.document_id)?, document);
    }
    retained.retain(retained.reserved_bytes() - tree - old_keys);
    Ok(())
}

fn build_sqlite_fts5_database(path: &Path, catalog: &Catalog) -> Result<()> {
    sqlite::execute(path, |output| {
        output.write_all(
            b"PRAGMA journal_mode=OFF;\n\
         PRAGMA synchronous=OFF;\n\
         CREATE VIRTUAL TABLE chunks_fts USING fts5(chunk_id UNINDEXED, text);\n\
         BEGIN;\n",
        )?;
        for (index, chunk) in catalog.chunks.values().enumerate() {
            write!(
                output,
                "INSERT INTO chunks_fts(rowid, chunk_id, text) VALUES({}, ",
                index + 1
            )?;
            sqlite::write_string(output, &chunk.chunk_external_id)?;
            output.write_all(b", ")?;
            sqlite::write_string(output, &chunk.text)?;
            output.write_all(b");\n")?;
        }
        output.write_all(b"COMMIT;\n")?;
        Ok(())
    })
}

fn run_sqlite_baseline_queries(
    db_path: &Path,
    dataset: &BenchmarkDataset,
    baseline_data: &SqliteBaselineData,
    profile: ResourceProfile,
    mode: BenchmarkQueryMode,
    cache_state: &str,
    work: &mut JsonReservation,
) -> Result<JsonValue> {
    let mut scratch = baseline_data.json_context.reservation();
    let mut per_query = Vec::new();
    let mut latencies = Vec::new();
    let mut recall_at_5_sum = 0.0;
    let mut recall_at_10_sum = 0.0;
    let mut reciprocal_rank_sum = 0.0;
    let mut vector_scan_count = 0;
    let mut distance_ops = 0;
    let mut fusion_candidate_count = 0;

    for query in &dataset.queries {
        let started = Instant::now();
        let (ranked_chunk_ids, scores, stats, query_context) = sqlite_baseline_query(
            db_path,
            query,
            baseline_data,
            profile,
            mode,
            &JsonBuilder::budgeted(work, JsonInputKind::BenchmarkArtifact),
        )?;
        let latency_ms = started.elapsed().as_secs_f64() * 1000.0;
        let hit = ranked_chunk_ids
            .iter()
            .any(|chunk_id| query.relevant_chunk_ids.contains(chunk_id));
        recall_at_5_sum += recall_at_k(ranked_chunk_ids.iter().map(String::as_str), query, 5);
        recall_at_10_sum += recall_at_k(ranked_chunk_ids.iter().map(String::as_str), query, 10);
        reciprocal_rank_sum += reciprocal_rank(ranked_chunk_ids.iter().map(String::as_str), query);
        scratch.reserve_vec(&mut latencies, 1, JsonInputKind::BenchmarkArtifact)?;
        latencies.push(latency_ms);
        vector_scan_count += stats.vector_scan_count;
        distance_ops += stats.distance_ops;
        fusion_candidate_count += stats.fusion_candidate_count;
        work.reserve_vec(&mut per_query, 1, JsonInputKind::BenchmarkArtifact)?;
        let builder = JsonBuilder::budgeted(work, JsonInputKind::BenchmarkArtifact);
        per_query.push(
            builder.object([
                ("query_id", builder.string(&query.query_id)?),
                ("latency_ms", JsonValue::Number(latency_ms)),
                (
                    "ranked_chunk_ids",
                    builder.array(
                        ranked_chunk_ids
                            .into_iter()
                            .map(|id| builder.owned_string(id)),
                    )?,
                ),
                ("score_kind", builder.string(mode.sqlite_score_kind())?),
                ("score_scale", builder.string("engine_local")?),
                ("score_comparable", JsonValue::Bool(false)),
                (
                    "scores",
                    builder.array(scores.into_iter().map(|value| Ok(JsonValue::Number(value))))?,
                ),
                ("hit", JsonValue::Bool(hit)),
                ("execution_stats", stats.build_json(&builder)?),
                ("query_context", query_context),
            ])?,
        );
    }

    let query_count = dataset.queries.len() as f64;
    let metrics = BenchmarkRunMetrics {
        recall_at_5: recall_at_5_sum / query_count,
        recall_at_10: recall_at_10_sum / query_count,
        mrr: reciprocal_rank_sum / query_count,
        p95_ms: percentile_95_in_place(&mut latencies),
        // SQLite runs in child processes; the parent's peak does not measure this baseline.
        memory_peak_mb: None,
        vector_scan_count,
        distance_ops,
        fusion_candidate_count,
    };
    let builder = JsonBuilder::budgeted(work, JsonInputKind::BenchmarkArtifact);
    builder.object([
        ("engine", builder.string("sqlite")?),
        ("name", builder.string(sqlite_baseline_name(dataset))?),
        ("query_mode", builder.string(mode.as_str())?),
        ("cache_state", builder.string(cache_state)?),
        ("metrics", benchmark_metrics_json_in(&metrics, &builder)?),
        ("per_query", JsonValue::Array(per_query)),
    ])
}

fn sqlite_baseline_query(
    db_path: &Path,
    query: &BenchmarkQuery,
    baseline_data: &SqliteBaselineData,
    profile: ResourceProfile,
    mode: BenchmarkQueryMode,
    builder: &JsonBuilder<'_>,
) -> Result<(Vec<String>, Vec<f64>, BenchmarkExecutionStats, JsonValue)> {
    let params = profile.params();
    let bm25_limit = if mode.has_text() {
        params.bm25_candidates.max(query.top_k)
    } else {
        0
    };
    let vector_limit = if mode.has_vector() {
        params.vector_candidates.max(query.top_k)
    } else {
        0
    };
    let fusion_limit = if mode.has_vector() {
        params.fusion_candidates.max(query.top_k)
    } else {
        0
    };
    let (ranked_chunk_ids, scores, stats) = match mode {
        BenchmarkQueryMode::TextBm25 => {
            let hits = sqlite_fts5_hits(db_path, query, baseline_data, bm25_limit)?;
            let ranked_chunk_ids = hits
                .iter()
                .take(query.top_k)
                .map(|hit| hit.chunk_id.clone())
                .collect::<Vec<_>>();
            let scores = hits
                .iter()
                .take(query.top_k)
                .map(|hit| hit.score)
                .collect::<Vec<_>>();
            (ranked_chunk_ids, scores, BenchmarkExecutionStats::default())
        }
        BenchmarkQueryMode::VectorExact => {
            let (hits, stats) = sqlite_vec_exact_hits(db_path, query, baseline_data, vector_limit)?;
            let ranked_chunk_ids = hits
                .iter()
                .take(query.top_k)
                .map(|hit| hit.chunk_id.clone())
                .collect::<Vec<_>>();
            let scores = hits
                .iter()
                .take(query.top_k)
                .map(|hit| hit.normalized_score)
                .collect::<Vec<_>>();
            (ranked_chunk_ids, scores, stats)
        }
        BenchmarkQueryMode::Hybrid => {
            let bm25_hits = sqlite_fts5_hits(db_path, query, baseline_data, bm25_limit)?;
            let (vector_hits, mut stats) =
                sqlite_vec_exact_hits(db_path, query, baseline_data, vector_limit)?;
            let weights = SearchWeights::default_for(true, true);
            let fused = fuse(
                &bm25_hits,
                &vector_hits,
                weights,
                fusion_limit,
                &baseline_data.json_context,
                |_| Default::default(),
            )?;
            let fused = fused.as_slice();
            stats.fusion_candidate_count = fused.len();
            let ranked_chunk_ids = fused
                .iter()
                .take(query.top_k)
                .map(|hit| hit.chunk_id.to_owned())
                .collect::<Vec<_>>();
            let scores = fused
                .iter()
                .take(query.top_k)
                .map(|hit| hit.final_score)
                .collect::<Vec<_>>();
            (ranked_chunk_ids, scores, stats)
        }
    };
    let query_context = sqlite_baseline_query_context(
        query,
        profile,
        mode,
        CandidateLimits {
            bm25: bm25_limit,
            vector: vector_limit,
            fusion: fusion_limit,
        },
        stats,
        builder,
    )?;
    Ok((ranked_chunk_ids, scores, stats, query_context))
}

fn sqlite_baseline_query_context(
    query: &BenchmarkQuery,
    profile: ResourceProfile,
    mode: BenchmarkQueryMode,
    limits: CandidateLimits,
    execution_stats: BenchmarkExecutionStats,
    builder: &JsonBuilder<'_>,
) -> Result<JsonValue> {
    let weights = SearchWeights::default_for(mode.has_text(), mode.has_vector());
    let filter_present = query
        .filter
        .as_object()
        .map(|obj| !obj.is_empty())
        .unwrap_or(false);
    builder.object([
        ("text_present", JsonValue::Bool(mode.has_text())),
        ("tokenizer", JsonValue::Null),
        ("embedding_present", JsonValue::Bool(mode.has_vector())),
        ("filter_present", JsonValue::Bool(filter_present)),
        (
            "weights",
            builder.object([
                ("present", JsonValue::Bool(false)),
                ("source", builder.string("default_no_boost")?),
                (
                    "requested",
                    benchmark_search_weights_json_in(weights, builder)?,
                ),
                (
                    "effective",
                    benchmark_search_weights_json_in(weights, builder)?,
                ),
                ("renormalization_reason", JsonValue::Null),
            ])?,
        ),
        (
            "requested_weights",
            benchmark_search_weights_json_in(weights, builder)?,
        ),
        (
            "effective_weights",
            benchmark_search_weights_json_in(weights, builder)?,
        ),
        ("weights_source", builder.string("default_no_boost")?),
        ("renormalization_reason", JsonValue::Null),
        ("metadata_boost_present", JsonValue::Bool(false)),
        ("metadata_boost_enabled", JsonValue::Bool(false)),
        ("boost_config_hash", JsonValue::Null),
        ("boost_rule_count", JsonValue::Number(0.0)),
        ("top_k", JsonValue::Number(query.top_k as f64)),
        ("profile", builder.string(profile.as_json_str())?),
        (
            "candidate_limits",
            builder.object([
                ("bm25", JsonValue::Number(limits.bm25 as f64)),
                ("vector", JsonValue::Number(limits.vector as f64)),
                ("fusion", JsonValue::Number(limits.fusion as f64)),
            ])?,
        ),
        ("execution_stats", execution_stats.build_json(builder)?),
        ("index_source", builder.string(sqlite_index_source(mode))?),
        ("index_rebuild_reason", JsonValue::Null),
        (
            "transient_rebuild",
            builder.object([
                ("started", JsonValue::Bool(false)),
                ("completed", JsonValue::Bool(false)),
                ("cancelled", JsonValue::Bool(false)),
                ("budget_mb", JsonValue::Number(0.0)),
                ("chunks_scanned", JsonValue::Null),
                ("bytes_read", JsonValue::Null),
                ("budget_exceeded", JsonValue::Bool(false)),
            ])?,
        ),
    ])
}

fn benchmark_search_weights_json_in(
    weights: SearchWeights,
    builder: &JsonBuilder<'_>,
) -> Result<JsonValue> {
    builder.object([
        ("bm25", JsonValue::Number(weights.bm25)),
        ("vector", JsonValue::Number(weights.vector)),
        ("metadata", JsonValue::Number(weights.metadata)),
    ])
}

fn sqlite_index_source(mode: BenchmarkQueryMode) -> &'static str {
    match mode {
        BenchmarkQueryMode::TextBm25 => "sqlite_fts5",
        BenchmarkQueryMode::VectorExact => "sqlite_vec",
        BenchmarkQueryMode::Hybrid => "sqlite_fts5_sqlite_vec",
    }
}

fn sqlite_fts5_hits(
    db_path: &Path,
    query: &BenchmarkQuery,
    baseline_data: &SqliteBaselineData,
    candidate_limit: usize,
) -> Result<Bm25Hits> {
    let Some(text) = query.text.as_deref() else {
        return Ok(Bm25Hits::empty(&baseline_data.json_context));
    };
    let Some(match_query) = sqlite_fts5_match_query_in(text, &baseline_data.json_context)? else {
        return Ok(Bm25Hits::empty(&baseline_data.json_context));
    };
    let (filter_expr, _filter_work) = FilterExpr::parse_validated_in(
        &query.filter,
        &baseline_data.json_context,
        JsonInputKind::Filter,
    )?;
    let mut hits = TextCandidates::new(candidate_limit, &baseline_data.json_context);
    sqlite::query(
        db_path,
        |output| {
            output.write_all(b"SELECT json_object('chunk_id',chunk_id,'score',bm25(chunks_fts)) FROM chunks_fts WHERE chunks_fts MATCH ")?;
            sqlite::write_string(output, match_query.as_ref())?;
            writeln!(
                output,
                " ORDER BY bm25(chunks_fts), chunk_id LIMIT {candidate_limit};"
            )?;
            Ok(())
        },
        |input| {
            sqlite::visit_scores(input, &baseline_data.json_context, |chunk_id, raw_score| {
                if sqlite_baseline_candidate_matches_filter(&chunk_id, &filter_expr, baseline_data)
                {
                    hits.push(
                        &chunk_id,
                        sqlite_fts5_rank_score(raw_score),
                        std::iter::empty(),
                    )?;
                }
                Ok(())
            })
        },
    )?;
    Ok(hits.finish())
}

fn sqlite_fts5_rank_score(raw_score: f64) -> f64 {
    if raw_score <= 0.0 {
        -raw_score
    } else {
        1.0 / (1.0 + raw_score)
    }
}

fn build_sqlite_vector_database(
    path: &Path,
    dataset: &BenchmarkDataset,
    context: &JsonOperationContext,
) -> Result<()> {
    sqlite::execute(path, |output| {
        write_sqlite_vectors(output, dataset, context)
    })
}

fn write_sqlite_vectors(
    output: &mut impl Write,
    dataset: &BenchmarkDataset,
    context: &JsonOperationContext,
) -> Result<()> {
    output.write_all(
        b"BEGIN; CREATE TABLE chunks_vec(chunk_id TEXT PRIMARY KEY, dim INTEGER NOT NULL, vector BLOB NOT NULL);\n",
    )?;
    visit_sqlite_baseline_embeddings(dataset, context, |embedding| {
        output.write_all(b"INSERT OR REPLACE INTO chunks_vec VALUES(")?;
        sqlite::write_string(output, &embedding.chunk_external_id)?;
        write!(output, ",{},vec_f32('", embedding.vector.len())?;
        crate::json::write_vector_json(output, &embedding.vector)?;
        output.write_all(b"'));\n")?;
        Ok(())
    })?;
    // The producing stream, including its trailing bytes, is verified before commit.
    output.write_all(b"COMMIT;\n")?;
    Ok(())
}

fn sqlite_vec_exact_hits(
    db_path: &Path,
    query: &BenchmarkQuery,
    baseline_data: &SqliteBaselineData,
    candidate_limit: usize,
) -> Result<(VectorHits, BenchmarkExecutionStats)> {
    let Some(embedding) = query.embedding.as_ref() else {
        return Ok((
            VectorHits::empty(&baseline_data.json_context),
            BenchmarkExecutionStats::default(),
        ));
    };
    let (filter_expr, _filter_work) = FilterExpr::parse_validated_in(
        &query.filter,
        &baseline_data.json_context,
        JsonInputKind::Filter,
    )?;
    let mut hits = VectorCandidates::new(candidate_limit, &baseline_data.json_context);
    let ((), vector_scan_count) = sqlite::query(
        db_path,
        |output| {
            output.write_all(b"SELECT json_object('chunk_id',chunk_id,'score',COALESCE(vec_distance_cosine(vector,vec_f32('")?;
            crate::json::write_vector_json(output, &embedding.vector)?;
            write!(
                output,
                "')),1.0)) FROM chunks_vec WHERE dim={}",
                embedding.vector.len()
            )?;
            if filter_expr.is_some() {
                output.write_all(b" AND chunk_id IN (")?;
                let mut count = 0;
                for chunk_external_id in baseline_data.chunk_ids_by_external_id.keys() {
                    if !sqlite_baseline_candidate_matches_filter(
                        chunk_external_id,
                        &filter_expr,
                        baseline_data,
                    ) {
                        continue;
                    }
                    if count != 0 {
                        output.write_all(b",")?;
                    }
                    sqlite::write_string(output, chunk_external_id)?;
                    count += 1;
                }
                if count == 0 {
                    output.write_all(b"NULL")?;
                }
                output.write_all(b")")?;
            }
            output.write_all(b";\n")?;
            Ok(())
        },
        |input| {
            let mut count = 0;
            sqlite::visit_scores(input, &baseline_data.json_context, |chunk_id, distance| {
                count += 1;
                let raw = (1.0 - distance).clamp(-1.0, 1.0);
                hits.push(
                    &chunk_id,
                    raw,
                    normalize_score(raw, DistanceMetric::Cosine, false),
                )
            })?;
            Ok(count)
        },
    )?;
    let hits = hits.finish();
    let stats = BenchmarkExecutionStats {
        vector_scan_count,
        distance_ops: vector_scan_count * embedding.dim,
        fusion_candidate_count: hits.len(),
    };
    Ok((hits, stats))
}

fn sqlite_baseline_candidate_matches_filter(
    chunk_external_id: &str,
    filter_expr: &Option<FilterExpr>,
    baseline_data: &SqliteBaselineData,
) -> bool {
    let Some(expr) = filter_expr else {
        return true;
    };
    let Some(chunk_id) = baseline_data
        .chunk_ids_by_external_id
        .get(chunk_external_id)
    else {
        return false;
    };
    let Some(chunk) = baseline_data.catalog.chunks.get(chunk_id) else {
        return false;
    };
    expr.matches_for_chunk(&baseline_data.catalog, chunk)
}

fn sqlite_fts5_match_query_in(
    text: &str,
    context: &JsonOperationContext,
) -> Result<Option<crate::json::ReservedString>> {
    let mut bytes = 0usize;
    crate::tokenizer::visit_tokens_in(text, context, |term| {
        let separator = usize::from(bytes != 0) * 4;
        bytes = bytes
            .checked_add(separator)
            .and_then(|size| size.checked_add(2))
            .and_then(|size| size.checked_add(term.len()))
            .and_then(|size| size.checked_add(term.bytes().filter(|byte| *byte == b'"').count()))
            .ok_or_else(|| {
                Error::new(
                    ErrorCode::FileFormatError,
                    "SQLite match query length overflow",
                )
            })?;
        Ok(())
    })?;
    if bytes == 0 {
        return Ok(None);
    }
    let mut query =
        crate::json::ReservedString::with_capacity(bytes, context, "operation_scratch")?;
    render_sqlite_fts5_match_query_in(text, &mut query, context)?;
    if query.as_ref().len() != bytes {
        return Err(std::io::Error::other("SQLite match query sizing differs").into());
    }
    Ok(Some(query))
}

fn render_sqlite_fts5_match_query_in(
    text: &str,
    query: &mut crate::json::ReservedString,
    context: &JsonOperationContext,
) -> Result<()> {
    let mut first = true;
    crate::tokenizer::visit_tokens_in(text, context, |term| {
        query.append(|output| {
            if !first {
                output.write_str(" OR ")?;
            }
            output.write_char('"')?;
            for ch in term.chars() {
                output.write_char(ch)?;
                if ch == '"' {
                    output.write_char('"')?;
                }
            }
            output.write_char('"')
        })?;
        first = false;
        Ok(())
    })
}

fn sqlite3_fts5_available() -> bool {
    Command::new(sqlite_cli())
        .arg("-batch")
        .arg(":memory:")
        .arg("CREATE VIRTUAL TABLE t USING fts5(x);")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .map(|status| status.success())
        .unwrap_or(false)
}

fn sqlite3_sqlite_vec_available() -> bool {
    sqlite3_script(Path::new(":memory:"), "SELECT vec_version();").is_ok()
}

fn sqlite3_command(db_path: &Path) -> Command {
    let mut command = Command::new(sqlite_cli());
    command
        .arg("-bail")
        .arg("-batch")
        .arg("-noheader")
        .arg("-separator")
        .arg("\t")
        .arg(db_path);
    command
}

fn sqlite3_script(db_path: &Path, sql: &str) -> Result<String> {
    sqlite::script(db_path, sql)
}

fn sqlite_cli() -> std::ffi::OsString {
    std::env::var_os("COFFERDB_SQLITE3").unwrap_or_else(|| "sqlite3".into())
}

fn sqlite_cli_label(command: &std::ffi::OsStr) -> Result<&str> {
    command
        .to_str()
        .filter(|value| !value.trim().is_empty() && !value.contains('\0'))
        .ok_or_else(|| {
            Error::new(
                ErrorCode::CliUsage,
                "SQLite executable must be nonempty UTF-8 without NUL bytes",
            )
        })
}

#[cfg(test)]
fn sql_quote(value: &str) -> String {
    format!("'{}'", value.replace('\'', "''"))
}

fn hardware_cpu_model() -> String {
    command_stdout("sysctl", &["-n", "machdep.cpu.brand_string"])
        .or_else(linux_cpu_model)
        .unwrap_or_else(|| "unknown CPU".into())
}

fn hardware_cpu_cores() -> usize {
    std::thread::available_parallelism()
        .map(|value| value.get())
        .unwrap_or(1)
}

fn hardware_memory_gb() -> Option<f64> {
    command_stdout("sysctl", &["-n", "hw.memsize"])
        .and_then(|value| value.parse::<u64>().ok())
        .map(bytes_to_gb)
        .or_else(linux_memory_gb)
}

fn hardware_os_name_version() -> (String, String) {
    if let Some(name) = command_stdout("sw_vers", &["-productName"]) {
        if let Some(version) = command_stdout("sw_vers", &["-productVersion"]) {
            return (name, version);
        }
    }
    if let Some(name) = linux_os_release_field("NAME") {
        let version = linux_os_release_field("VERSION_ID")
            .or_else(|| linux_os_release_field("VERSION"))
            .unwrap_or_else(|| std::env::consts::ARCH.into());
        return (name, version);
    }
    (std::env::consts::OS.into(), std::env::consts::ARCH.into())
}

fn command_stdout(program: &str, args: &[&str]) -> Option<String> {
    let output = Command::new(program).args(args).output().ok()?;
    if !output.status.success() {
        return None;
    }
    let value = String::from_utf8(output.stdout).ok()?;
    let value = value.trim();
    (!value.is_empty()).then(|| value.to_owned())
}

fn linux_cpu_model() -> Option<String> {
    fs::read_to_string("/proc/cpuinfo")
        .ok()?
        .lines()
        .find_map(|line| line.strip_prefix("model name"))
        .and_then(|value| {
            value
                .split_once(':')
                .map(|(_, model)| model.trim().to_owned())
        })
        .filter(|value| !value.is_empty())
}

fn linux_memory_gb() -> Option<f64> {
    fs::read_to_string("/proc/meminfo")
        .ok()?
        .lines()
        .find_map(|line| line.strip_prefix("MemTotal:"))
        .and_then(|value| value.split_whitespace().next())
        .and_then(|kb| kb.parse::<u64>().ok())
        .map(|kb| bytes_to_gb(kb * 1024))
}

fn linux_os_release_field(field: &str) -> Option<String> {
    fs::read_to_string("/etc/os-release")
        .ok()?
        .lines()
        .find_map(|line| {
            let (key, value) = line.split_once('=')?;
            (key == field).then(|| value.trim_matches('"').to_owned())
        })
        .filter(|value| !value.is_empty())
}

fn bytes_to_gb(bytes: u64) -> f64 {
    ((bytes as f64 / 1_073_741_824.0) * 10.0).round() / 10.0
}

fn profile_params_json_in(
    profile: ResourceProfile,
    builder: &JsonBuilder<'_>,
) -> Result<JsonValue> {
    builder.object([
        (
            "bm25_candidates",
            JsonValue::Number(profile.params().bm25_candidates as f64),
        ),
        (
            "vector_candidates",
            JsonValue::Number(profile.params().vector_candidates as f64),
        ),
        (
            "fusion_candidates",
            JsonValue::Number(profile.params().fusion_candidates as f64),
        ),
        (
            "cache_budget_mb",
            JsonValue::Number(profile.params().cache_budget_mb as f64),
        ),
    ])
}

#[cfg(test)]
fn benchmark_engine_params_json() -> JsonValue {
    benchmark_engine_params_json_in(&JsonBuilder::host())
        .expect("host DTO construction is infallible")
}

fn benchmark_engine_params_json_in(builder: &JsonBuilder<'_>) -> Result<JsonValue> {
    builder.object([
        (
            "text_index",
            builder.object([
                ("index_kind", builder.string("text_bm25")?),
                (
                    "tokenizer",
                    crate::TokenizerConfig::default().build_json(builder)?,
                ),
                (
                    "bm25",
                    builder.object([
                        ("k1", JsonValue::Number(crate::index::text::BM25_K1)),
                        ("b", JsonValue::Number(crate::index::text::BM25_B)),
                    ])?,
                ),
            ])?,
        ),
        (
            "vector_index",
            builder.object([
                ("index_kind", builder.string("vector_exact_f32")?),
                ("distance_metric", builder.string("cosine")?),
                ("vectors_normalized", JsonValue::Bool(false)),
                (
                    "normalization_method",
                    builder.string("cosine_to_unit_interval")?,
                ),
            ])?,
        ),
        (
            "fusion",
            builder.object([
                ("algorithm", builder.string("weighted_sum")?),
                ("score_scale", builder.string("engine_local")?),
            ])?,
        ),
    ])
}

#[cfg(test)]
mod tests;
