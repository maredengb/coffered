use super::*;

pub(super) const LOCATION_BYTES: usize = 28;

pub(super) struct StagedRecord {
    pub(super) kind: BatchRecordKind,
    pub(super) source: PayloadRef,
}

impl StagedRecord {
    pub(super) fn encode(&self) -> Result<[u8; LOCATION_BYTES]> {
        let checksum = u32::try_from(self.source.checksum).map_err(|_| invalid_location())?;
        let mut bytes = [0; LOCATION_BYTES];
        bytes[..2].copy_from_slice(&self.kind.record_type.to_le_bytes());
        bytes[2..4].copy_from_slice(&self.kind.replaces.to_le_bytes());
        bytes[4..12].copy_from_slice(&self.source.offset.to_le_bytes());
        bytes[12..20].copy_from_slice(&self.source.len.to_le_bytes());
        bytes[20..24].copy_from_slice(&checksum.to_le_bytes());
        let checksum = checksum32(&bytes[..24]);
        bytes[24..].copy_from_slice(&checksum.to_le_bytes());
        Ok(bytes)
    }

    fn decode(bytes: &[u8; LOCATION_BYTES], source_bytes: u64) -> Result<Self> {
        if checksum32(&bytes[..24]) != read_u32(&bytes[24..]) {
            return Err(Error::new(
                ErrorCode::ChecksumMismatch,
                "prepared import location checksum mismatch",
            ));
        }
        let kind = BatchRecordKind {
            record_type: u16::from_le_bytes(bytes[..2].try_into().unwrap()),
            replaces: u16::from_le_bytes(bytes[2..4].try_into().unwrap()),
        };
        let source = PayloadRef {
            offset: read_u64(&bytes[4..12]),
            len: read_u64(&bytes[12..20]),
            checksum: u64::from(read_u32(&bytes[20..24])),
        };
        if !matches!(
            kind.record_type,
            COLLECTION_CREATE
                | COLLECTION_DROP
                | DOCUMENT_PUT
                | CHUNK_PUT
                | EMBEDDING_PUT
                | TOMBSTONE
        ) || !(kind.replaces == 0
            || (kind.record_type == TOMBSTONE && matches!(kind.replaces, DOCUMENT_PUT | CHUNK_PUT)))
            || source.len == 0
            || source
                .offset
                .checked_add(source.len)
                .filter(|end| *end <= source_bytes)
                .is_none()
        {
            return Err(invalid_location());
        }
        Ok(Self { kind, source })
    }
}

pub(super) struct LocationReader<'a> {
    input: BufReader<&'a mut File>,
    work: WorkspaceReservation,
    remaining: usize,
    source_bytes: u64,
}

impl<'a> LocationReader<'a> {
    pub(super) fn new(
        file: &'a mut File,
        first: usize,
        remaining: usize,
        source_bytes: u64,
        context: &JsonOperationContext,
    ) -> Result<Self> {
        let total = first.checked_add(remaining).ok_or_else(invalid_location)?;
        let expected_bytes = (total as u64)
            .checked_mul(LOCATION_BYTES as u64)
            .ok_or_else(invalid_location)?;
        let actual_bytes = file.metadata()?.len();
        if actual_bytes < expected_bytes {
            return Err(std::io::Error::from(std::io::ErrorKind::UnexpectedEof).into());
        }
        if actual_bytes != expected_bytes {
            return Err(invalid_location());
        }
        file.seek(SeekFrom::Start(first as u64 * LOCATION_BYTES as u64))?;
        let mut work = context.workspace_reservation();
        let preferred = remaining.saturating_mul(LOCATION_BYTES).min(8192);
        let capacity = work.grow_to(0, preferred, "operation_scratch")?;
        let mut input = BufReader::with_capacity(capacity, file);
        if input.capacity() > capacity {
            let file = input.into_inner();
            work.retain(0);
            input = BufReader::with_capacity(0, file);
        }
        Ok(Self {
            input,
            work,
            remaining,
            source_bytes,
        })
    }

    pub(super) fn is_buffered(&self) -> bool {
        self.work.reserved_bytes() > 0
    }

    pub(super) fn into_unbuffered(mut self) -> Result<Self> {
        // Discard read-ahead without skipping the locations already in the buffer.
        let position = self.input.stream_position()?;
        self.input.seek(SeekFrom::Start(position))?;
        let Self {
            input,
            mut work,
            remaining,
            source_bytes,
        } = self;
        let file = input.into_inner();
        work.retain(0);
        Ok(Self {
            input: BufReader::with_capacity(0, file),
            work,
            remaining,
            source_bytes,
        })
    }
}

impl Iterator for LocationReader<'_> {
    type Item = Result<StagedRecord>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 {
            return None;
        }
        let mut bytes = [0; LOCATION_BYTES];
        let result = self
            .input
            .read_exact(&mut bytes)
            .map_err(Error::from)
            .and_then(|_| StagedRecord::decode(&bytes, self.source_bytes));
        self.remaining = if result.is_ok() {
            self.remaining - 1
        } else {
            0
        };
        Some(result)
    }
}

fn invalid_location() -> Error {
    Error::new(
        ErrorCode::FileFormatError,
        "invalid prepared import location",
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn location_buffers_share_headroom_and_preserve_read_ahead_on_release() {
        use crate::json::JsonProcessingPolicyV1;
        for seed in 0..100usize {
            let (mut file, temporary) = crate::temporary::create_temporary_file_in(
                &std::env::temp_dir(),
                "location-work",
                "tmp",
            )
            .unwrap();
            let count = 317 + seed;
            for offset in 0..count {
                file.write_all(
                    &StagedRecord {
                        kind: BatchRecordKind {
                            record_type: DOCUMENT_PUT,
                            replaces: 0,
                        },
                        source: PayloadRef {
                            offset: offset as u64,
                            len: 1,
                            checksum: seed as u64,
                        },
                    }
                    .encode()
                    .unwrap(),
                )
                .unwrap();
            }
            let context = JsonOperationContext::standalone();
            let available = [0, 1, 27, 28, 29, 4096, 8191, 8192, 10000][seed % 9];
            let mut held = context.workspace_reservation();
            let occupied = JsonProcessingPolicyV1::WORK_BUDGET_BYTES - available;
            held.grow_to(occupied, occupied, "operation_scratch")
                .unwrap();
            let mut reader =
                LocationReader::new(&mut file, 3, count - 3, count as u64, &context).unwrap();
            let capacity = reader.input.capacity();
            assert_eq!(capacity, available.min(8192));
            assert_eq!(reader.work.reserved_bytes(), capacity);
            for offset in 3..count {
                if offset == 8 {
                    reader = reader.into_unbuffered().unwrap();
                    assert!(!reader.is_buffered());
                    assert_eq!(reader.input.capacity(), 0);
                    context
                        .workspace_reservation()
                        .grow_to(available, available, "operation_scratch")
                        .unwrap();
                }
                let actual = reader.next().unwrap().unwrap();
                assert_eq!(actual.source.offset, offset as u64);
                assert_eq!(actual.source.checksum, seed as u64);
            }
            assert!(reader.next().is_none());
            drop(reader);
            context
                .workspace_reservation()
                .grow_to(available, available, "operation_scratch")
                .unwrap();

            file.seek(SeekFrom::Start((7 * LOCATION_BYTES) as u64))
                .unwrap();
            file.write_all(&[0xff]).unwrap();
            let mut reader =
                LocationReader::new(&mut file, 7, count - 7, count as u64, &context).unwrap();
            assert_eq!(
                reader.next().unwrap().err().unwrap().code(),
                ErrorCode::ChecksumMismatch
            );
            assert!(reader.next().is_none());
            drop(reader);
            context
                .workspace_reservation()
                .grow_to(available, available, "operation_scratch")
                .unwrap();
            file.set_len((count * LOCATION_BYTES - 1) as u64).unwrap();
            let error = LocationReader::new(&mut file, 0, count, count as u64, &context)
                .err()
                .unwrap();
            assert_eq!(error.code(), ErrorCode::IoError);
            context
                .workspace_reservation()
                .grow_to(available, available, "operation_scratch")
                .unwrap();
            drop(file);
            drop(temporary);
        }
    }

    #[test]
    fn locations_roundtrip_and_reject_every_corrupted_byte_across_100_seeds() {
        for seed in 0..100_u64 {
            let record = StagedRecord {
                kind: BatchRecordKind {
                    record_type: TOMBSTONE,
                    replaces: [0, DOCUMENT_PUT, CHUNK_PUT][seed as usize % 3],
                },
                source: PayloadRef {
                    offset: seed * 37,
                    len: seed + 1,
                    checksum: seed * 1234567,
                },
            };
            let bytes = record.encode().unwrap();
            let end = record.source.offset + record.source.len;
            assert_eq!(
                StagedRecord::decode(&bytes, end).unwrap().encode().unwrap(),
                bytes
            );
            assert!(StagedRecord::decode(&bytes, end - 1).is_err());
            for position in 0..LOCATION_BYTES {
                let mut bad = bytes;
                bad[position] ^= 0x80;
                assert_eq!(
                    StagedRecord::decode(&bad, end).err().unwrap().code(),
                    ErrorCode::ChecksumMismatch
                );
            }
            for (offset, length, kind, replaces) in [
                (u64::MAX, 1, TOMBSTONE, 0),
                (0, 0, TOMBSTONE, 0),
                (0, 1, COMMIT, 0),
                (0, 1, DOCUMENT_PUT, CHUNK_PUT),
                (0, 1, TOMBSTONE, EMBEDDING_PUT),
            ] {
                let bad = StagedRecord {
                    kind: BatchRecordKind {
                        record_type: kind,
                        replaces,
                    },
                    source: PayloadRef {
                        offset,
                        len: length,
                        checksum: 0,
                    },
                }
                .encode()
                .unwrap();
                assert_eq!(
                    StagedRecord::decode(&bad, u64::MAX).err().unwrap().code(),
                    ErrorCode::FileFormatError
                );
            }
        }
    }
}
