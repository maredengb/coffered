use std::io::{BufRead, BufReader, Read};

use super::{
    parse_with_schema_in, JsonOperationContext, JsonReservation, JsonSchema, ProcessedJson,
};
use crate::{Error, ErrorCode, Result};

pub(crate) struct JsonLines<R> {
    reader: BufReader<R>,
    line: Vec<u8>,
    context: JsonOperationContext,
    schema: JsonSchema,
    line_number: usize,
    bytes_read: u64,
    byte_limit: u64,
    done: bool,
    _input_work: JsonReservation,
    line_work: JsonReservation,
}

impl<R: Read> JsonLines<R> {
    pub(crate) fn into_inner(self) -> R {
        self.reader.into_inner()
    }

    pub(crate) fn new(
        source: R,
        schema: JsonSchema,
        context: &JsonOperationContext,
        byte_limit: u64,
    ) -> Result<Self> {
        let mut input_work = context.reservation();
        input_work
            .grow(8192, schema.kind())
            .map_err(|error| error.with_detail("line", 1))?;
        Ok(Self {
            reader: BufReader::with_capacity(8192, source),
            context: context.clone(),
            schema,
            _input_work: input_work,
            line_work: context.reservation(),
            line: Vec::new(),
            line_number: 0,
            bytes_read: 0,
            byte_limit,
            done: false,
        })
    }

    fn read_next(&mut self) -> Result<Option<(usize, ProcessedJson)>> {
        loop {
            self.line_number = self.line_number.checked_add(1).ok_or_else(|| {
                Error::new(ErrorCode::InvalidJsonlSchema, "JSONL line count overflow")
            })?;
            self.line.clear();
            loop {
                let bytes = match self.reader.fill_buf() {
                    Err(error) if error.kind() == std::io::ErrorKind::Interrupted => continue,
                    result => result?,
                };
                if bytes.is_empty() {
                    break;
                }
                let count = bytes
                    .iter()
                    .position(|byte| *byte == b'\n')
                    .map_or(bytes.len(), |index| index + 1);
                self.bytes_read = self
                    .bytes_read
                    .checked_add(count as u64)
                    .filter(|bytes| *bytes <= self.byte_limit)
                    .ok_or_else(|| {
                        Error::new(
                            ErrorCode::InvalidJsonlSchema,
                            "JSONL input exceeds the byte budget",
                        )
                        .with_detail("reason", "import_budget_exceeded")
                        .with_detail("limit_bytes", self.byte_limit)
                    })?;
                let complete = bytes[count - 1] == b'\n';
                self.line_work
                    .reserve_vec(&mut self.line, count, self.schema.kind())?;
                self.line.extend_from_slice(&bytes[..count]);
                self.reader.consume(count);
                if complete {
                    break;
                }
            }
            if self.line.is_empty() {
                return Ok(None);
            }
            let line = std::str::from_utf8(&self.line).map_err(|_| {
                std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "stream did not contain valid UTF-8",
                )
            })?;
            if line.trim().is_empty() {
                continue;
            }
            let value = parse_with_schema_in(line, self.schema, &self.context)?;
            return Ok(Some((self.line_number, value)));
        }
    }
}

impl<R: Read> Iterator for JsonLines<R> {
    type Item = Result<(usize, ProcessedJson)>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }
        let result = self.read_next();
        self.done = matches!(&result, Ok(None) | Err(_));
        result
            .map_err(|error| error.with_detail("line", self.line_number))
            .transpose()
    }
}

impl<R: Read> std::iter::FusedIterator for JsonLines<R> {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::json::{JsonInputKind, JsonProcessingPolicyV1};

    struct RepeatedRows<'a> {
        bytes: &'a [u8],
        remaining: usize,
        offset: usize,
    }

    impl Read for RepeatedRows<'_> {
        fn read(&mut self, output: &mut [u8]) -> std::io::Result<usize> {
            if self.remaining == 0 {
                return Ok(0);
            }
            let count = output.len().min(self.bytes.len() - self.offset);
            output[..count].copy_from_slice(&self.bytes[self.offset..self.offset + count]);
            self.offset += count;
            if self.offset == self.bytes.len() {
                self.offset = 0;
                self.remaining -= 1;
            }
            Ok(count)
        }
    }

    #[test]
    fn rows_reuse_released_capacity_but_charge_retained_values() {
        let context = JsonOperationContext::standalone();
        let mut busy = context.reservation();
        busy.grow(
            JsonProcessingPolicyV1::WORK_BUDGET_BYTES - 2 * 1024 * 1024,
            JsonInputKind::JsonlRecord,
        )
        .unwrap();
        let bytes = format!(
            "{{\"external_id\":\"doc\",\"metadata\":{{\"value\":\"{}\"}}}}\n",
            "x".repeat(512 * 1024)
        )
        .into_bytes();
        let source = || RepeatedRows {
            bytes: &bytes,
            remaining: 100,
            offset: 0,
        };
        let rows = JsonLines::new(source(), JsonSchema::DocumentRow, &context, u64::MAX).unwrap();
        let mut count = 0;
        for row in rows {
            let (line, parsed) = row.unwrap();
            count += 1;
            assert_eq!(line, count);
            assert_eq!(
                parsed
                    .value
                    .get("metadata")
                    .unwrap()
                    .get("value")
                    .unwrap()
                    .as_str()
                    .unwrap()
                    .len(),
                512 * 1024
            );
        }
        assert_eq!(count, 100);
        let mut rows =
            JsonLines::new(source(), JsonSchema::DocumentRow, &context, u64::MAX).unwrap();
        let first = rows.next().unwrap().unwrap();
        let error = rows.next().unwrap().err().unwrap();
        assert!(error
            .detail()
            .iter()
            .any(|(key, value)| key == "json_limit_reason" && value == "work_budget_exceeded"));
        assert!(rows.next().is_none());
        drop((first, rows));
        busy.grow(2 * 1024 * 1024, JsonInputKind::JsonlRecord)
            .unwrap();
    }

    #[test]
    fn byte_limit_is_exact_and_errors_terminate_the_iterator() {
        for (bytes, limit, expected) in [
            (&b"{}\n{}\n"[..], 5, ErrorCode::InvalidJsonlSchema),
            (&b"{}\n\n{\"x\":\"\xff\"}"[..], u64::MAX, ErrorCode::IoError),
            (&b"{}\n\n{"[..], u64::MAX, ErrorCode::InvalidJsonlSchema),
        ] {
            let context = JsonOperationContext::standalone();
            let mut rows = JsonLines::new(bytes, JsonSchema::DocumentRow, &context, limit).unwrap();
            drop(rows.next().unwrap().unwrap());
            let error = rows.next().unwrap().err().unwrap();
            assert_eq!(error.code(), expected);
            let line = if limit == 5 { "2" } else { "3" };
            assert!(error
                .detail()
                .iter()
                .any(|(key, value)| key == "line" && value == line));
            assert!(rows.next().is_none());
            assert!(rows.next().is_none());
            drop(rows);
            context
                .reservation()
                .grow(
                    JsonProcessingPolicyV1::WORK_BUDGET_BYTES,
                    JsonInputKind::JsonlRecord,
                )
                .unwrap();
        }
        let context = JsonOperationContext::standalone();
        let rows = JsonLines::new(&b"{}\n{}\n"[..], JsonSchema::DocumentRow, &context, 6).unwrap();
        assert_eq!(rows.map(|row| row.unwrap().0).collect::<Vec<_>>(), [1, 2]);
    }

    #[test]
    fn line_buffer_failure_uses_the_enclosing_schema_kind() {
        for schema in [
            JsonSchema::DocumentRow,
            JsonSchema::ChunkRow,
            JsonSchema::EmbeddingRow,
            JsonSchema::QueryRow,
            JsonSchema::QueryEmbeddingObject,
            JsonSchema::BenchmarkRecord,
        ] {
            let context = JsonOperationContext::standalone();
            let mut busy = context.reservation();
            busy.grow(
                JsonProcessingPolicyV1::WORK_BUDGET_BYTES - 16384,
                schema.kind(),
            )
            .unwrap();
            let bytes = format!("\n{{\"text\":\"{}\"}}", "x".repeat(32768));
            let mut rows = JsonLines::new(bytes.as_bytes(), schema, &context, u64::MAX).unwrap();
            let error = rows.next().unwrap().err().unwrap();
            assert_eq!(error.code(), schema.kind().error_code());
            assert!(error
                .detail()
                .iter()
                .any(|(key, value)| key == "json_input_kind" && value == schema.kind().as_str()));
            assert!(error
                .detail()
                .iter()
                .any(|(key, value)| key == "line" && value == "2"));
            drop(rows);
            busy.grow(16384, schema.kind()).unwrap();
        }
    }

    #[test]
    fn fragmented_unicode_and_interrupted_reads_keep_line_numbers() {
        struct Fragmented<'a> {
            bytes: &'a [u8],
            calls: usize,
            width: usize,
        }
        impl Read for Fragmented<'_> {
            fn read(&mut self, output: &mut [u8]) -> std::io::Result<usize> {
                self.calls += 1;
                if self.calls % 2 == 1 {
                    return Err(std::io::ErrorKind::Interrupted.into());
                }
                let count = output.len().min(self.bytes.len()).min(self.width);
                output[..count].copy_from_slice(&self.bytes[..count]);
                self.bytes = &self.bytes[count..];
                Ok(count)
            }
        }
        let bytes = " \r\n{\"text\":\"\u{4e2d}\u{1f642}\"}\r\n\n{\"text\":\"last\"}";
        for width in 1..=100 {
            let context = JsonOperationContext::standalone();
            let rows = JsonLines::new(
                Fragmented {
                    bytes: bytes.as_bytes(),
                    calls: 0,
                    width,
                },
                JsonSchema::ChunkRow,
                &context,
                bytes.len() as u64,
            )
            .unwrap();
            let rows = rows
                .map(|row| {
                    let (line, parsed) = row.unwrap();
                    (
                        line,
                        parsed
                            .value
                            .get("text")
                            .unwrap()
                            .as_str()
                            .unwrap()
                            .to_owned(),
                    )
                })
                .collect::<Vec<_>>();
            assert_eq!(rows, [(2, "\u{4e2d}\u{1f642}".into()), (4, "last".into())]);
        }
    }
}
