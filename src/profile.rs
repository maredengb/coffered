/// Resource presets with fixed default candidate windows and cache budgets.
/// String inputs accept only `expanded`, `balanced`, and `constrained`.
/// These names do not promise retrieval quality, latency, or process RSS.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum ResourceProfile {
    Expanded,
    #[default]
    Balanced,
    Constrained,
}

impl ResourceProfile {
    pub fn as_json_str(self) -> &'static str {
        match self {
            ResourceProfile::Expanded => "expanded",
            ResourceProfile::Balanced => "balanced",
            ResourceProfile::Constrained => "constrained",
        }
    }

    pub fn as_cli_str(self) -> &'static str {
        match self {
            ResourceProfile::Expanded => "expanded",
            ResourceProfile::Balanced => "balanced",
            ResourceProfile::Constrained => "constrained",
        }
    }

    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "expanded" => Some(ResourceProfile::Expanded),
            "balanced" => Some(ResourceProfile::Balanced),
            "constrained" => Some(ResourceProfile::Constrained),
            _ => None,
        }
    }

    pub fn params(self) -> ProfileParams {
        match self {
            ResourceProfile::Expanded => ProfileParams {
                bm25_candidates: 500,
                vector_candidates: 500,
                fusion_candidates: 500,
                cache_budget_mb: 512,
                background_compaction: false,
                import_batch_chunks: 2_000,
            },
            ResourceProfile::Balanced => ProfileParams {
                bm25_candidates: 200,
                vector_candidates: 200,
                fusion_candidates: 200,
                cache_budget_mb: 256,
                background_compaction: false,
                import_batch_chunks: 1_000,
            },
            ResourceProfile::Constrained => ProfileParams {
                bm25_candidates: 50,
                vector_candidates: 50,
                fusion_candidates: 50,
                cache_budget_mb: 64,
                background_compaction: false,
                import_batch_chunks: 250,
            },
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct ProfileParams {
    pub bm25_candidates: usize,
    pub vector_candidates: usize,
    pub fusion_candidates: usize,
    pub cache_budget_mb: usize,
    pub background_compaction: bool,
    pub import_batch_chunks: usize,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum Durability {
    #[default]
    Commit,
    Fast,
}

impl Durability {
    pub fn as_str(self) -> &'static str {
        match self {
            Durability::Commit => "commit",
            Durability::Fast => "fast",
        }
    }

    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "commit" => Some(Durability::Commit),
            "fast" => Some(Durability::Fast),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum IoMode {
    #[default]
    Auto,
    Read,
    MmapDisabled,
}

#[derive(Clone, Debug)]
pub struct DatabaseConfig {
    pub resource_profile: ResourceProfile,
    pub cache_budget_mb: Option<usize>,
    pub background_compaction: bool,
    pub durability: Durability,
    pub io_mode: IoMode,
}

impl Default for DatabaseConfig {
    fn default() -> Self {
        Self {
            resource_profile: ResourceProfile::Balanced,
            cache_budget_mb: None,
            background_compaction: false,
            durability: Durability::Commit,
            io_mode: IoMode::Auto,
        }
    }
}

impl DatabaseConfig {
    pub(crate) fn validate_json_policy(&self) -> crate::Result<()> {
        use crate::json::{object_from_pairs, JsonInputKind, JsonValue};
        object_from_pairs([
            (
                "resource_profile",
                JsonValue::String(self.resource_profile.as_json_str().into()),
            ),
            (
                "cache_budget_mb",
                self.cache_budget_mb
                    .map(|value| JsonValue::Number(value as f64))
                    .unwrap_or(JsonValue::Null),
            ),
            (
                "background_compaction",
                JsonValue::Bool(self.background_compaction),
            ),
            (
                "durability",
                JsonValue::String(self.durability.as_str().into()),
            ),
            (
                "io_mode",
                JsonValue::String(
                    match self.io_mode {
                        IoMode::Auto => "auto",
                        IoMode::Read => "read",
                        IoMode::MmapDisabled => "mmap_disabled",
                    }
                    .into(),
                ),
            ),
        ])
        .validate_kind(JsonInputKind::RuntimeConfig)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn supported_presets_keep_their_parameter_contract_and_balanced_default() {
        assert_eq!(ResourceProfile::default(), ResourceProfile::Balanced);
        assert_eq!(
            DatabaseConfig::default().resource_profile,
            ResourceProfile::Balanced
        );
        for (profile, name, candidates, cache, batch) in [
            (ResourceProfile::Expanded, "expanded", 500, 512, 2_000),
            (ResourceProfile::Balanced, "balanced", 200, 256, 1_000),
            (ResourceProfile::Constrained, "constrained", 50, 64, 250),
        ] {
            assert_eq!(ResourceProfile::parse(name), Some(profile));
            assert_eq!(profile.as_cli_str(), name);
            assert_eq!(profile.as_json_str(), name);
            let params = profile.params();
            assert_eq!(params.bm25_candidates, candidates);
            assert_eq!(params.vector_candidates, candidates);
            assert_eq!(params.fusion_candidates, candidates);
            assert_eq!(params.cache_budget_mb, cache);
            assert_eq!(params.import_batch_chunks, batch);
            assert!(!params.background_compaction);
        }
    }

    #[test]
    fn parse_rejects_old_names_case_variants_whitespace_and_help() {
        for value in [
            "quality-first",
            "quality_first",
            "resource-first",
            "resource_first",
            "performance",
            "eco",
            "QualityFirst",
            "ResourceFirst",
            "Expanded",
            "Balanced",
            "Constrained",
            "EXPANDED",
            "BALANCED",
            "CONSTRAINED",
            " expanded",
            "expanded ",
            " balanced",
            "constrained ",
            "constrained\n",
            "expanded-budget",
            "help",
            "",
            "turbo",
        ] {
            assert_eq!(ResourceProfile::parse(value), None, "{value:?}");
        }
    }
}
