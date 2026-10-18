#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum BenchmarkQueryMode {
    TextBm25,
    VectorExact,
    Hybrid,
}

impl BenchmarkQueryMode {
    pub(super) fn parse(value: &str) -> Option<Self> {
        match value {
            "text_bm25" => Some(Self::TextBm25),
            "vector_exact" => Some(Self::VectorExact),
            "hybrid" => Some(Self::Hybrid),
            _ => None,
        }
    }

    pub(super) fn as_str(self) -> &'static str {
        match self {
            Self::TextBm25 => "text_bm25",
            Self::VectorExact => "vector_exact",
            Self::Hybrid => "hybrid",
        }
    }

    pub(super) fn score_kind(self) -> &'static str {
        match self {
            Self::TextBm25 => "cofferdb_bm25_final_score",
            Self::VectorExact => "cofferdb_vector_final_score",
            Self::Hybrid => "cofferdb_hybrid_final_score",
        }
    }

    pub(super) fn sqlite_score_kind(self) -> &'static str {
        match self {
            Self::TextBm25 => "sqlite_fts5_bm25",
            Self::VectorExact => "sqlite_vec_cosine",
            Self::Hybrid => "sqlite_fts5_vec_weighted_sum",
        }
    }

    pub(super) fn has_text(self) -> bool {
        matches!(self, Self::TextBm25 | Self::Hybrid)
    }

    pub(super) fn has_vector(self) -> bool {
        matches!(self, Self::VectorExact | Self::Hybrid)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stable_query_modes_keep_wire_names_branches_and_engine_local_score_labels() {
        for (name, mode, text, vector, coffer_score, sqlite_score) in [
            (
                "text_bm25",
                BenchmarkQueryMode::TextBm25,
                true,
                false,
                "cofferdb_bm25_final_score",
                "sqlite_fts5_bm25",
            ),
            (
                "vector_exact",
                BenchmarkQueryMode::VectorExact,
                false,
                true,
                "cofferdb_vector_final_score",
                "sqlite_vec_cosine",
            ),
            (
                "hybrid",
                BenchmarkQueryMode::Hybrid,
                true,
                true,
                "cofferdb_hybrid_final_score",
                "sqlite_fts5_vec_weighted_sum",
            ),
        ] {
            assert_eq!(BenchmarkQueryMode::parse(name), Some(mode));
            assert_eq!(mode.as_str(), name);
            assert_eq!(mode.has_text(), text);
            assert_eq!(mode.has_vector(), vector);
            assert_eq!(mode.score_kind(), coffer_score);
            assert_eq!(mode.sqlite_score_kind(), sqlite_score);
            for seed in 0..100 {
                for invalid in [
                    format!("{name}_{seed}"),
                    format!("{seed}_{name}"),
                    format!("{name}{}", " ".repeat(seed + 1)),
                    format!("{}{name}", " ".repeat(seed + 1)),
                    format!("{name}\0{seed}"),
                ] {
                    assert_eq!(BenchmarkQueryMode::parse(&invalid), None);
                }
            }
        }
        for invalid in ["", "TEXT_BM25", "VectorExact", "bm25", "vector", "Hybrid"] {
            assert_eq!(BenchmarkQueryMode::parse(invalid), None);
        }
    }
}
