import fs from 'node:fs';
import path from 'node:path';
import { createHash } from 'node:crypto';

const output = process.argv[2];
if (!output) throw new Error('Usage: node tools/prepare-benchmark.mjs <new-output-directory>');
fs.mkdirSync(output, { recursive: false });
const topics = ['apple harvest schedule', 'orchard irrigation pipes', 'greenhouse tomato seedlings',
  'beekeeping winter shelter', 'grain delivery timetable', 'berry pruning tools', 'farm weather observations',
  'market stall reservations', 'compost soil preparation', 'tractor maintenance checklist'];
const documents = topics.map((text, i) => ({ external_id: `doc-${i}`, title: text, metadata: {} }));
const chunks = topics.map((text, i) => ({ document_external_id: `doc-${i}`,
  chunk_external_id: `chunk-${i}`, ordinal: 0, text, metadata: { group: i % 3 } }));
const vector = i => [0, 1, 2].map(j => j === i % 3 ? 1 : 0);
const embeddings = topics.map((_, i) => ({ chunk_external_id: `chunk-${i}`,
  model_name: 'synthetic-demo-v1', dim: 3, vector: vector(i) }));
const queries = [0, 1, 2].map(i => ({ query_id: `q-${i}`, text: topics[i], filter: {}, top_k: 5,
  embedding: vector(i) }));
const qrels = [0, 1, 2].map(i => ({ query_id: `q-${i}`, chunk_external_id: `chunk-${i}`, relevance: 1 }));
const hashes = {};
for (const [name, rows] of Object.entries({ documents, chunks, embeddings, queries, qrels })) {
  const bytes = rows.map(row => JSON.stringify(row)).join('\n') + '\n';
  fs.writeFileSync(path.join(output, `${name}.jsonl`), bytes, { flag: 'wx' });
  hashes[`${name}_sha256`] = createHash('sha256').update(bytes).digest('hex');
}
const manifest = { schema_version: '0.1', dataset_id: 'orchard-functional-10', version: '1',
  documents: 'documents.jsonl', chunks: 'chunks.jsonl', embeddings: 'embeddings.jsonl',
  queries: 'queries.jsonl', qrels: 'qrels.jsonl', query_embedding_source: 'inline',
  embedding_model: 'synthetic-demo-v1', query_embedding_model: 'synthetic-demo-v1',
  query_embedding_dim: 3, chunking: { chunk_size: 256, chunk_overlap: 0 },
  tokenizer: { name: 'default_en', version: '0.1', profile: 'latin_code_default', language_hint: 'en',
    config: { case_fold: 'lower', split: 'unicode_punctuation', stop_words: 'default_en' } }, hashes };
fs.writeFileSync(path.join(output, 'dataset_manifest.json'), JSON.stringify(manifest, null, 2) + '\n', { flag: 'wx' });
console.log(path.resolve(output));
