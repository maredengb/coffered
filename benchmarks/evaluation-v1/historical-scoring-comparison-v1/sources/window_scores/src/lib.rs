mod benchmark;
mod catalog;
#[doc(hidden)]
pub mod cli;
mod db;
mod error;
mod hash;
mod import;
mod index;
mod json;
mod lock;
mod metadata;
mod model;
mod profile;
mod query;
mod storage;
mod temporary;
mod time;
mod tokenizer;

#[cfg(test)]
mod test_alloc;

pub use tokenizer::TokenizerConfig;

pub use db::{
    CheckOptions, CheckReport, CompactOptions, CompactSummary, Database, DatabaseOpenOptions,
};
pub use error::{Error, ErrorCode, Result};
pub use import::{ExportSummary, ImportManifest, ImportSummary};
pub use json::{parse as parse_json, JsonObject, JsonValue};
pub use metadata::MetadataBoostConfig;
pub use model::{
    Chunk, ChunkInput, CollectionConfig, CollectionHandle, CollectionInfo, CollectionStats,
    DistanceMetric, Document, DocumentInput, Embedding, EmbeddingInput, IndexStats, InsertSummary,
    OnConflict, QueryEmbeddingInput, SearchWeights, SourceSpan, VectorIndexConfig,
};
pub use profile::{DatabaseConfig, Durability, IoMode, ResourceProfile};
pub use query::{SearchBuilder, SearchResult, SearchResults, WhyNotDiagnosis};
