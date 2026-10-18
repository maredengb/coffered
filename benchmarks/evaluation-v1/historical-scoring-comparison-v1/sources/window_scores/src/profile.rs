#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum ResourceProfile {
    QualityFirst,
    #[default]
    Balanced,
    ResourceFirst,
}

impl ResourceProfile {
    pub fn as_json_str(self) -> &'static str {
        match self {
            ResourceProfile::QualityFirst => "quality_first",
            ResourceProfile::Balanced => "balanced",
            ResourceProfile::ResourceFirst => "resource_first",
        }
    }

    pub fn as_cli_str(self) -> &'static str {
        match self {
            ResourceProfile::QualityFirst => "quality-first",
            ResourceProfile::Balanced => "balanced",
            ResourceProfile::ResourceFirst => "resource-first",
        }
    }

    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "quality-first" | "quality_first" | "performance" => {
                Some(ResourceProfile::QualityFirst)
            }
            "balanced" => Some(ResourceProfile::Balanced),
            "resource-first" | "resource_first" | "eco" => Some(ResourceProfile::ResourceFirst),
            _ => None,
        }
    }

    pub fn params(self) -> ProfileParams {
        match self {
            ResourceProfile::QualityFirst => ProfileParams {
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
            ResourceProfile::ResourceFirst => ProfileParams {
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
