use super::*;
use std::io::BufWriter;

const RULES_VERSION: &str = "cofferdb-builtin-source-acceptance-v4";

pub(super) struct ExecutionConfig<'a> {
    pub database: &'a str,
    pub result_file: Option<&'a str>,
    pub collection: &'a str,
    pub profile: ResourceProfile,
    pub durability: Durability,
}

pub(super) struct ExecutionJournal<'a> {
    file: File,
    path: PathBuf,
    dataset: &'a BenchmarkDataset,
    modes: &'a [BenchmarkQueryMode],
    context: JsonOperationContext,
    started: Instant,
    next: usize,
    active: Option<Instant>,
    active_cost: Option<crate::search_cost::SearchCostTrace>,
    active_search_ms: Option<f64>,
    active_search_successful: Option<bool>,
    planned: usize,
    write_failed: bool,
    latencies: Vec<f64>,
    latency_work: JsonReservation,
}

impl<'a> ExecutionJournal<'a> {
    pub(super) fn new(
        dataset: &'a BenchmarkDataset,
        modes: &'a [BenchmarkQueryMode],
        config: ExecutionConfig<'_>,
        context: &JsonOperationContext,
        started: Instant,
    ) -> Result<Self> {
        let parent = Path::new(config.result_file.unwrap_or(config.database))
            .parent()
            .filter(|path| !path.as_os_str().is_empty())
            .unwrap_or_else(|| Path::new("."))
            .canonicalize()?;
        let planned = dataset
            .queries
            .len()
            .checked_mul(modes.len())
            .and_then(|n| n.checked_mul(2))
            .ok_or_else(|| {
                Error::new(
                    ErrorCode::InvalidJsonlSchema,
                    "benchmark sample count overflow",
                )
            })?;
        let (file, guard) = create_temporary_file_in(&parent, "benchmark-execution", "jsonl")?;
        let path = guard.path().to_path_buf();
        // Publish a retained sidecar under its exclusive name before execution, so
        // interrupted runs survive without overwriting inputs or prior attempts.
        guard.persist(&path)?;
        let mut journal = Self {
            file,
            path,
            dataset,
            modes,
            context: context.clone(),
            started,
            next: 0,
            active: None,
            active_cost: None,
            active_search_ms: None,
            active_search_successful: None,
            planned,
            write_failed: false,
            latencies: Vec::new(),
            latency_work: context.reservation(),
        };
        let result = (|| {
            let mut work = context.reservation();
            let builder = JsonBuilder::budgeted(&mut work, JsonInputKind::BenchmarkArtifact);
            let header = builder.object([
            ("event", builder.string("protocol")?),
            ("rules_version", builder.string(RULES_VERSION)?),
                ("git_commit", benchmark_git_commit_json_in(&builder)?),
                ("hardware", benchmark_hardware_json_in(&builder)?),
            ("dataset", dataset.manifest.build_result_dataset_json(&builder)?),
            ("database_path", builder.string(config.database)?),
            ("result_file", builder.optional_str(config.result_file)?),
            ("collection", builder.string(config.collection)?),
            ("profile", builder.string(config.profile.as_json_str())?),
            ("durability", builder.string(config.durability.as_str())?),
            ("index_build_mode", builder.string("inline_full_checkpoint")?),
            ("planned_samples", JsonValue::Number(planned as f64)),
            ("order", builder.string("text_bm25,vector_exact,hybrid when available; cold then warm; artifact query order")?),
            ("repetitions_per_group", JsonValue::Number(1.0)),
            ("warmup_rounds", JsonValue::Number(0.0)),
            ("warmup_reason", builder.string("first pass is a measured cold-handle group; the measured warm group repeats it")?),
            ("cold_reset", builder.string("reopen handle; no OS page-cache eviction")?),
            ("concurrency", JsonValue::Number(1.0)),
            ("query_timeout_ms", JsonValue::Null),
            ("run_timeout_ms", JsonValue::Null),
            ("timeout_policy", builder.string("no internal deadline; externally interrupted journals without a terminal event are incomplete, never successful")?),
            ("completion_rule", builder.string("CLI exit zero plus a durable final run_completed event linked to the published result; any failure or missing terminal event disqualifies the run")?),
            ("retry_policy", builder.string("abort on first error; no retries; unexecuted planned samples remain explicit")?),
            ("latency_scope", builder.string("continuous search call including preparation, recall, score completion, normalization/fusion and result materialization; excludes reference generation, verification and journal IO; failed samples use frozen search time when a search finished, otherwise started-event elapsed")?),
            ("search_cost_version", builder.string(crate::search_cost::VERSION)?),
            ("search_cost_scope", builder.string("costs are cumulative snapshots per sequence/attempt: use search_completed/search_failed once; completed/failed repeat the same snapshot solely to bind acceptance status and must not be added again; fixed per-search diagnostic counters; failed/reclaimed attempts remain counted; stage times are inclusive diagnostics, not a substitute for or sum defining end-to-end latency; preparation includes optional cache loads across collections; no engine candidate staging reads; source-reference/import/journal IO excluded")?),
            ("search_cost_work_rules", builder.string("text visits are consumer document/posting visits, not all validator CPU or binary-search comparisons; vector checks count inspected vectors before eligibility and distances count scored components; reuse counts each recalled branch score consumed directly in final candidate construction, including failed fusion attempts; logical bytes count typed payload or index-extent requests, application bytes count successful Read/read_at returns including partial failures, retries and checksum drains; physical disk IO unknown; unset candidate counts are null, never zero")?),
                ("percentile_method", builder.string("nearest rank ceil(p*n)-1 for P50/P95, unrounded successful query latencies, separately per mode/cache group; incomplete-group percentiles are diagnostic only")?),
            ("import_time_scope", builder.string("continuous first input read through all cold/warm queries and source verification; includes reference generation and journal IO in that interval")?),
            ("import_phase_scope", builder.string("commit includes commit preparation, old-index prefetch, critical append/durability and Catalog application; checkpoint includes postcommit index preparation, resident construction and suffix persistence/retry; cycles count logical calls, ready cycles count query-usable persistent suffixes after those calls; disjoint diagnostics, not the end-to-end total; omitted when import fails")?),
            ("reference", builder.string("hash-verified raw rows; full-collection streaming BM25 statistics; only the filtered recall union receives supplemental scores before normalization/fusion limits; shared vector metric and fusion primitives; imported IDs only for ties; qrels are quality labels, not a 100-percent-recall assertion")?),
            ("score_tolerance", JsonValue::Number(1e-12)),
            ("score_tolerance_rule", builder.string("absolute error <= tolerance * max(1, abs(expected)); exact ranked external IDs and result count")?),
            ("journal_durability", builder.string("flush each event; sync plan and terminal events")?),
        ])?;
            journal.write(header, work)?;
            for index in 0..planned {
                journal.sample("planned", index, None, None, None)?;
            }
            journal.file.sync_data().map_err(Error::from)
        })();
        result.map_err(|error| journal.locate(error))?;
        Ok(journal)
    }

    pub(super) fn import_completed(
        &mut self,
        batches: &crate::db::ImportBatchMetrics,
    ) -> Result<()> {
        let mut work = self.context.reservation();
        let builder = JsonBuilder::budgeted(&mut work, JsonInputKind::BenchmarkArtifact);
        let value = builder.object([
            ("event", builder.string("import_completed")?),
            (
                "committed_batches",
                JsonValue::Number(batches.committed_batches as f64),
            ),
            (
                "checkpoint_cycles",
                JsonValue::Number(batches.checkpoint_cycles as f64),
            ),
            (
                "checkpoint_ready_cycles",
                JsonValue::Number(batches.checkpoint_ready_cycles as f64),
            ),
            (
                "final_checkpoint_ready",
                JsonValue::Bool(batches.final_checkpoint_ready),
            ),
            (
                "max_batch_chunks",
                JsonValue::Number(batches.max_batch_chunks as f64),
            ),
            (
                "commit_time_ms",
                JsonValue::Number(batches.commit_time.as_secs_f64() * 1000.0),
            ),
            (
                "checkpoint_time_ms",
                JsonValue::Number(batches.checkpoint_time.as_secs_f64() * 1000.0),
            ),
            (
                "elapsed_ms",
                JsonValue::Number(self.started.elapsed().as_secs_f64() * 1000.0),
            ),
        ])?;
        self.write(value, work)?;
        self.file.sync_data()?;
        Ok(())
    }

    pub(super) fn expectations(&mut self, acceptance: &acceptance::Acceptance) -> Result<()> {
        for (mode, _) in self.modes.iter().enumerate() {
            for query in 0..self.dataset.queries.len() {
                let mut work = self.context.reservation();
                let builder = JsonBuilder::budgeted(&mut work, JsonInputKind::BenchmarkArtifact);
                let value = builder.object([
                    ("event", builder.string("expected")?),
                    ("query_mode", builder.string(self.modes[mode].as_str())?),
                    (
                        "query_id",
                        builder.string(&self.dataset.queries[query].query_id)?,
                    ),
                    (
                        "ranked_chunk_ids",
                        builder.array(
                            acceptance
                                .expected_hits(mode, query)
                                .map(|(id, _)| builder.string(id)),
                        )?,
                    ),
                    (
                        "scores",
                        builder.array(
                            acceptance
                                .expected_hits(mode, query)
                                .map(|(_, score)| Ok(JsonValue::Number(score))),
                        )?,
                    ),
                ])?;
                self.write(value, work)?;
            }
        }
        self.file.sync_data()?;
        Ok(())
    }

    pub(super) fn begin(&mut self, mode: usize, cache: &str, query: usize) -> Result<()> {
        let cache = match cache {
            "cold" => 0,
            "warm" => 1,
            _ => return Err(invalid_order()),
        };
        let expected = (mode * 2 + cache) * self.dataset.queries.len() + query;
        if self.active.is_some() || self.next != expected || self.next >= self.planned {
            return Err(invalid_order());
        }
        self.active_cost = None;
        self.active_search_ms = None;
        self.active_search_successful = None;
        self.sample("started", self.next, None, None, None)?;
        self.active = Some(Instant::now());
        Ok(())
    }

    pub(super) fn search_finished(
        &mut self,
        trace: crate::search_cost::SearchCostTrace,
        elapsed_ms: f64,
        successful: bool,
    ) -> Result<()> {
        if self.active.is_none()
            || self.active_cost.is_some()
            || !elapsed_ms.is_finite()
            || elapsed_ms < 0.0
        {
            return Err(invalid_order());
        }
        self.active_search_ms = Some(elapsed_ms);
        let valid = trace.valid();
        self.active_search_successful = Some(successful && valid);
        self.active_cost = Some(trace);
        self.sample(
            if successful {
                "search_completed"
            } else {
                "search_failed"
            },
            self.next,
            Some(elapsed_ms),
            None,
            None,
        )?;
        if !valid {
            return Err(Error::new(
                ErrorCode::InvalidSearchParams,
                "benchmark search cost counters overflowed; the measurement is invalid",
            ));
        }
        Ok(())
    }

    pub(super) fn complete_query(
        &mut self,
        latency_ms: f64,
        results: &SearchResults,
    ) -> Result<()> {
        if self.active.is_none()
            || self.active_cost.is_none()
            || self.active_search_ms != Some(latency_ms)
            || self.active_search_successful != Some(true)
        {
            return Err(invalid_order());
        }
        self.latency_work
            .reserve_vec(&mut self.latencies, 1, JsonInputKind::BenchmarkArtifact)?;
        self.sample(
            "completed",
            self.next,
            Some(latency_ms),
            Some(&results.query_context.index_source),
            None,
        )?;
        self.latencies.push(latency_ms);
        self.next += 1;
        self.active = None;
        self.active_cost = None;
        self.active_search_ms = None;
        self.active_search_successful = None;
        if self.next % self.dataset.queries.len() == 0 {
            let summary = self.group_summary(self.next / self.dataset.queries.len() - 1, 0, 0);
            self.latencies = Vec::new();
            self.latency_work = self.context.reservation();
            summary?;
        }
        Ok(())
    }

    pub(super) fn publish_run(
        &mut self,
        mut run: BenchmarkRun,
        output: Option<output::BenchmarkOutput>,
    ) -> Result<BenchmarkRun> {
        // A journal failure must not replace a previous result. Publication is
        // last; if it fails, the final journal event disqualifies this attempt.
        let result = self.complete_run(&mut run).and_then(|()| {
            if let Some(output) = output {
                output.write(&run.value, &run.context, &mut run.work)?;
            }
            Ok(())
        });
        match result {
            Ok(()) => Ok(run),
            Err(error) => {
                drop(run);
                Err(self.fail(error))
            }
        }
    }

    fn complete_run(&mut self, run: &mut BenchmarkRun) -> Result<()> {
        if self.active.is_some() || self.next != self.planned {
            return Err(invalid_order());
        }
        let mut digest = crate::hash::Sha256Writer::new(std::io::sink());
        run.value.write_pretty_retained_in(
            &mut digest,
            JsonOutputSchema::Value(JsonInputKind::BenchmarkArtifact),
            &run.context,
            &mut run.work,
        )?;
        let digest = digest.finish()?;
        let mut work = self.context.reservation();
        let builder = JsonBuilder::budgeted(&mut work, JsonInputKind::BenchmarkArtifact);
        let value = builder.object([
            ("event", builder.string("run_completed")?),
            ("result_pretty_json_sha256", builder.string(&digest)?),
            (
                "elapsed_ms",
                JsonValue::Number(self.started.elapsed().as_secs_f64() * 1000.0),
            ),
            (
                "import_time_ms",
                JsonValue::Number(
                    run.value
                        .get("metrics")
                        .unwrap()
                        .get("import_time_ms")
                        .unwrap()
                        .as_f64()
                        .unwrap(),
                ),
            ),
            ("completed_samples", JsonValue::Number(self.next as f64)),
        ])?;
        self.write(value, work)?;
        self.file.sync_data()?;
        Ok(())
    }

    pub(super) fn fail(&mut self, error: Error) -> Error {
        let record = (|| {
            let unfinished_group =
                (self.next < self.planned).then_some(self.next / self.dataset.queries.len().max(1));
            let failed = usize::from(self.active.is_some());
            if let Some(started) = self.active {
                self.sample(
                    "failed",
                    self.next,
                    Some(
                        self.active_search_ms
                            .unwrap_or_else(|| started.elapsed().as_secs_f64() * 1000.0),
                    ),
                    None,
                    Some(&error),
                )?;
                self.next += 1;
                self.active = None;
                self.active_cost = None;
                self.active_search_ms = None;
                self.active_search_successful = None;
            }
            for index in self.next..self.planned {
                self.sample("not_run", index, None, None, Some(&error))?;
            }
            if let Some(group) = unfinished_group {
                self.group_summary(
                    group,
                    failed,
                    self.dataset.queries.len() - self.latencies.len() - failed,
                )?;
                self.latencies = Vec::new();
                self.latency_work = self.context.reservation();
                for remaining in group + 1..self.modes.len() * 2 {
                    self.group_summary(remaining, 0, self.dataset.queries.len())?;
                }
            }
            let mut work = self.context.reservation();
            let builder = JsonBuilder::budgeted(&mut work, JsonInputKind::BenchmarkArtifact);
            let value = builder.object([
                ("event", builder.string("run_failed")?),
                (
                    "elapsed_ms",
                    JsonValue::Number(self.started.elapsed().as_secs_f64() * 1000.0),
                ),
                ("error_code", builder.string(error.code_str())?),
                ("reason", builder.string(error.message())?),
                (
                    "error_detail",
                    builder.array(error.detail().iter().map(|(key, value)| {
                        builder.object([
                            ("field", builder.string(key)?),
                            ("value", builder.string(value)?),
                        ])
                    }))?,
                ),
            ])?;
            self.write(value, work)?;
            self.file.sync_data()?;
            Ok::<_, Error>(())
        })();
        let error = self.locate(error);
        match record {
            Ok(()) => error,
            Err(record_error) => {
                error.with_detail("execution_record_error", record_error.to_string())
            }
        }
    }

    fn locate(&self, error: Error) -> Error {
        // The sidecar is beside the supplied result (or database). Never expose
        // its internally canonicalized directory or a private staging location.
        error
            .with_detail(
                "benchmark_execution_record",
                self.path.file_name().unwrap().to_string_lossy(),
            )
            .with_detail("benchmark_rules_version", RULES_VERSION)
    }

    fn group_summary(&mut self, group: usize, failed: usize, not_run: usize) -> Result<()> {
        self.latencies.sort_by(f64::total_cmp);
        let count = self.latencies.len();
        let percentile = |p: f64| {
            if count == 0 {
                JsonValue::Null
            } else {
                JsonValue::Number(self.latencies[(p * count as f64).ceil() as usize - 1])
            }
        };
        let mut work = self.context.reservation();
        let builder = JsonBuilder::budgeted(&mut work, JsonInputKind::BenchmarkArtifact);
        let value = builder.object([
            ("event", builder.string("group_summary")?),
            (
                "query_mode",
                builder.string(self.modes[group / 2].as_str())?,
            ),
            (
                "cache_state",
                builder.string(if group % 2 == 0 { "cold" } else { "warm" })?,
            ),
            (
                "planned",
                JsonValue::Number(self.dataset.queries.len() as f64),
            ),
            ("completed", JsonValue::Number(count as f64)),
            ("failed", JsonValue::Number(failed as f64)),
            ("not_run", JsonValue::Number(not_run as f64)),
            ("timed_out", JsonValue::Number(0.0)),
            ("p50_ms", percentile(0.5)),
            ("p95_ms", percentile(0.95)),
            (
                "statistics_scope",
                builder.string(if failed == 0 && not_run == 0 {
                    "complete_group"
                } else {
                    "diagnostic_successful_samples_only"
                })?,
            ),
        ])?;
        self.write(value, work)
    }

    fn sample(
        &mut self,
        event: &str,
        index: usize,
        latency: Option<f64>,
        index_source: Option<&str>,
        error: Option<&Error>,
    ) -> Result<()> {
        let query_count = self.dataset.queries.len();
        let query = &self.dataset.queries[index % query_count];
        let group = index / query_count;
        let mut work = self.context.reservation();
        let builder = JsonBuilder::budgeted(&mut work, JsonInputKind::BenchmarkArtifact);
        let value = builder.object([
            ("event", builder.string(event)?),
            ("sequence", JsonValue::Number(index as f64)),
            ("query_id", builder.string(&query.query_id)?),
            (
                "query_mode",
                builder.string(self.modes[group / 2].as_str())?,
            ),
            (
                "cache_state",
                builder.string(if group % 2 == 0 { "cold" } else { "warm" })?,
            ),
            ("round", JsonValue::Number(1.0)),
            ("attempt", JsonValue::Number(1.0)),
            (
                "elapsed_ms",
                latency.map(JsonValue::Number).unwrap_or(JsonValue::Null),
            ),
            ("index_source", builder.optional_str(index_source)?),
            (
                "search_cost",
                if matches!(
                    event,
                    "search_completed" | "search_failed" | "completed" | "failed"
                ) {
                    self.active_cost
                        .as_ref()
                        .map(|trace| trace.json(&builder))
                        .transpose()?
                        .unwrap_or(JsonValue::Null)
                } else {
                    JsonValue::Null
                },
            ),
            (
                "error_code",
                builder.optional_str(error.map(Error::code_str))?,
            ),
            ("reason", builder.optional_str(error.map(Error::message))?),
        ])?;
        self.write(value, work)
    }

    fn write(&mut self, value: JsonValue, mut retained: JsonReservation) -> Result<()> {
        if self.write_failed {
            return Err(Error::new(
                ErrorCode::IoError,
                "execution journal contains an incomplete event; further writes are disabled",
            ));
        }
        let mut buffer_work = self.context.reservation();
        buffer_work.grow(8192, JsonInputKind::BenchmarkArtifact)?;
        let mut output = BufWriter::with_capacity(8192, &mut self.file);
        let result = value
            .write_value_retained_in(
                &mut output,
                JsonInputKind::BenchmarkArtifact,
                &self.context,
                &mut retained,
                false,
            )
            .and_then(|()| {
                output.write_all(b"\n")?;
                output.flush().map_err(Error::from)
            });
        // A partial event remains incomplete; never retry buffered bytes during Drop.
        let _ = output.into_parts();
        self.write_failed = result.is_err();
        result
    }
}

fn invalid_order() -> Error {
    Error::new(
        ErrorCode::RecoveryError,
        "benchmark execution differs from its fixed sample plan",
    )
}

#[cfg(test)]
mod tests {
    use super::super::tests::text_only_benchmark_dataset;
    use super::*;

    #[test]
    fn journal_and_publication_failures_do_not_replace_previous_results() {
        let dir = text_only_benchmark_dataset("journal-publication-order");
        let context = JsonOperationContext::standalone();
        let (dataset, _work) = BenchmarkDataset::load_and_validate_in(&dir, &context).unwrap();
        let modes = [BenchmarkQueryMode::TextBm25];
        let database = dir.join("data.coffer");
        for stage in 0..3 {
            let result_path = dir.join(format!("result-{stage}.json"));
            fs::write(&result_path, b"prior result").unwrap();
            let output = output::BenchmarkOutput::new(&result_path, &database, Vec::new()).unwrap();
            let mut journal = ExecutionJournal::new(
                &dataset,
                &modes,
                ExecutionConfig {
                    database: database.to_str().unwrap(),
                    result_file: result_path.to_str(),
                    collection: "notes",
                    profile: ResourceProfile::Balanced,
                    durability: Durability::Fast,
                },
                &context,
                Instant::now(),
            )
            .unwrap();
            // Isolate terminal publication from query execution, tested separately.
            journal.next = journal.planned;
            if stage == 0 {
                journal.file = File::open(&journal.path).unwrap();
            } else if stage == 1 {
                fs::rename(&result_path, dir.join("prior-result.json")).unwrap();
                fs::create_dir(&result_path).unwrap();
            }
            let mut work = context.reservation();
            let builder = JsonBuilder::budgeted(&mut work, JsonInputKind::BenchmarkArtifact);
            let value = builder
                .object([(
                    "metrics",
                    builder
                        .object([("import_time_ms", JsonValue::Number(1.0))])
                        .unwrap(),
                )])
                .unwrap();
            let result = journal.publish_run(
                BenchmarkRun {
                    value,
                    context: context.clone(),
                    work,
                },
                Some(output),
            );
            let records = fs::read_to_string(&journal.path).unwrap();
            match stage {
                0 => {
                    assert_eq!(result.err().unwrap().code(), ErrorCode::IoError);
                    assert_eq!(fs::read(&result_path).unwrap(), b"prior result");
                    assert!(!records.contains("\"run_completed\""));
                }
                1 => {
                    assert_eq!(result.err().unwrap().code(), ErrorCode::IoError);
                    assert_eq!(
                        fs::read(dir.join("prior-result.json")).unwrap(),
                        b"prior result"
                    );
                    assert!(records.lines().last().unwrap().contains("\"run_failed\""));
                }
                _ => {
                    let run = result.unwrap();
                    let mut expected = Vec::new();
                    let mut run = run;
                    run.value
                        .write_pretty_retained_in(
                            &mut expected,
                            JsonOutputSchema::Value(JsonInputKind::BenchmarkArtifact),
                            &run.context,
                            &mut run.work,
                        )
                        .unwrap();
                    assert_eq!(fs::read(&result_path).unwrap(), expected);
                    assert!(records
                        .lines()
                        .last()
                        .unwrap()
                        .contains("\"run_completed\""));
                }
            }
        }
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn journal_write_failure_is_sticky_and_does_not_corrupt_prior_events() {
        let dir = text_only_benchmark_dataset("journal-io-failure");
        let context = JsonOperationContext::standalone();
        let (dataset, _work) = BenchmarkDataset::load_and_validate_in(&dir, &context).unwrap();
        let modes = [BenchmarkQueryMode::TextBm25];
        let database = dir.join("data.coffer");
        let mut journal = ExecutionJournal::new(
            &dataset,
            &modes,
            ExecutionConfig {
                database: database.to_str().unwrap(),
                result_file: None,
                collection: "notes",
                profile: ResourceProfile::Balanced,
                durability: Durability::Fast,
            },
            &context,
            Instant::now(),
        )
        .unwrap();
        let before = fs::read(&journal.path).unwrap();
        journal.file = File::open(&journal.path).unwrap();
        let write_error = journal.begin(0, "cold", 0).unwrap_err();
        assert_eq!(write_error.code(), ErrorCode::IoError);
        assert!(journal.write_failed);
        journal.file = fs::OpenOptions::new()
            .append(true)
            .open(&journal.path)
            .unwrap();
        let error = journal.fail(Error::new(ErrorCode::ChecksumMismatch, "original failure"));
        assert_eq!(error.code(), ErrorCode::ChecksumMismatch);
        assert!(error
            .detail()
            .iter()
            .any(|(key, value)| key == "execution_record_error"
                && value.contains("further writes are disabled")));
        assert_eq!(fs::read(&journal.path).unwrap(), before);
        drop(journal);
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn failed_initial_record_retains_location_and_shared_budget_can_be_reused() {
        let dir = text_only_benchmark_dataset("journal-budget-failure");
        let context = JsonOperationContext::standalone();
        let (dataset, work) = BenchmarkDataset::load_and_validate_in(&dir, &context).unwrap();
        let modes = [BenchmarkQueryMode::TextBm25];
        let database = dir.join("data.coffer");
        let new_journal = || {
            ExecutionJournal::new(
                &dataset,
                &modes,
                ExecutionConfig {
                    database: database.to_str().unwrap(),
                    result_file: None,
                    collection: "notes",
                    profile: ResourceProfile::Balanced,
                    durability: Durability::Fast,
                },
                &context,
                Instant::now(),
            )
        };
        let mut busy = context.reservation();
        busy.grow(
            crate::json::JsonProcessingPolicyV1::WORK_BUDGET_BYTES - work.reserved_bytes(),
            JsonInputKind::BenchmarkArtifact,
        )
        .unwrap();
        let error = new_journal().err().unwrap();
        assert_eq!(error.code(), ErrorCode::InvalidJsonlSchema);
        let path = error
            .detail()
            .iter()
            .find(|(key, _)| key == "benchmark_execution_record")
            .unwrap()
            .1
            .clone();
        assert_eq!(Path::new(&path).components().count(), 1);
        assert!(dir.join(&path).is_file());
        assert_eq!(fs::metadata(dir.join(&path)).unwrap().len(), 0);
        drop(busy);
        let mut journal = new_journal().unwrap();
        assert_ne!(journal.path, dir.join(path));
        assert!(journal.begin(0, "warm", 0).is_err());
        journal.begin(0, "cold", 0).unwrap();
        let interrupted = journal.path.clone();
        drop(journal);
        let records = fs::read_to_string(interrupted).unwrap();
        assert!(records.contains("\"started\""));
        assert!(!records.contains("\"run_completed\""));
        drop((dataset, work));
        context
            .reservation()
            .grow(
                crate::json::JsonProcessingPolicyV1::WORK_BUDGET_BYTES,
                JsonInputKind::BenchmarkArtifact,
            )
            .unwrap();
        fs::remove_dir_all(dir).unwrap();
    }
}
