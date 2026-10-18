use super::*;
use crate::json::{JsonInputKind, JsonOperationContext, JsonReservation, WorkspaceReservation};

// The summary cache is optional. When it cannot fit, retain only the verified
// batch range and replay it after the commit marker has been validated.
pub(super) struct PendingBatch {
    records: Vec<CommittedRecord>,
    work: RecordReservation,
    array: WorkspaceReservation,
    buffered: bool,
    count: usize,
    first_offset: u64,
    first_sequence: u64,
    last_sequence: u64,
}

impl PendingBatch {
    pub(super) fn new(context: &JsonOperationContext) -> Self {
        Self {
            records: Vec::new(),
            work: RecordReservation::new(context.reservation(), context.workspace_reservation()),
            array: context.workspace_reservation(),
            buffered: true,
            count: 0,
            first_offset: 0,
            first_sequence: 0,
            last_sequence: 0,
        }
    }

    pub(super) fn is_empty(&self) -> bool {
        self.count == 0
    }

    pub(super) fn first_sequence(&self) -> Option<u64> {
        (!self.is_empty()).then_some(self.first_sequence)
    }

    pub(super) fn matches(&self, commit: &CommitPayload) -> bool {
        self.count > 0
            && self.count == commit.record_count
            && self.first_sequence == commit.first_sequence_no
            && self.last_sequence == commit.last_sequence_no
    }

    pub(super) fn discard_buffer(&mut self) -> bool {
        if !self.buffered {
            return false;
        }
        self.records = Vec::new();
        self.work.json.retain(0);
        self.work.retain_workspace(0);
        self.array.retain(0);
        self.buffered = false;
        true
    }

    pub(super) fn push(
        &mut self,
        json: JsonReservation,
        strings: WorkspaceReservation,
        record: CommittedRecord,
    ) -> Result<()> {
        if self.is_empty() {
            self.first_offset = record.record_ref.offset;
            self.first_sequence = record.record_ref.sequence_no;
        }
        self.last_sequence = record.record_ref.sequence_no;
        self.count = self.count.checked_add(1).ok_or_else(|| {
            Error::new(ErrorCode::FileFormatError, "recovery batch count overflow")
        })?;
        if self.buffered && self.records.len() == self.records.capacity() {
            let element = std::mem::size_of::<CommittedRecord>();
            let minimum = self
                .records
                .len()
                .checked_add(1)
                .and_then(|count| count.checked_mul(element))
                .ok_or_else(|| std::io::Error::other("recovery batch capacity overflow"))?;
            let preferred = minimum.saturating_mul(2);
            match self.array.grow_to(minimum, preferred, "operation_scratch") {
                Ok(bytes) => {
                    let capacity = bytes / element;
                    if self
                        .records
                        .try_reserve_exact(capacity - self.records.len())
                        .is_err()
                        || self.records.capacity() > capacity
                    {
                        self.discard_buffer();
                    } else {
                        self.array.retain(self.records.capacity() * element);
                    }
                }
                Err(error) if error.code() == ErrorCode::ResourceExhausted => {
                    self.discard_buffer();
                }
                Err(error) => return Err(error),
            }
        }
        if self.buffered {
            self.records.push(record);
            self.work.merge(json);
            self.work.merge_workspace(strings);
        } else {
            drop(record);
        }
        Ok(())
    }

    pub(super) fn visit(
        &mut self,
        file: &mut File,
        batch_id: u64,
        commit_offset: u64,
        context: &JsonOperationContext,
        visit: &mut dyn FnMut(CommittedRecord, &mut RecordReservation) -> Result<()>,
    ) -> Result<()> {
        if self.buffered {
            for record in self.records.drain(..) {
                visit(record, &mut self.work)?;
            }
        } else {
            self.replay(file, batch_id, commit_offset, context, visit)?;
        }
        *self = Self::new(context);
        Ok(())
    }

    fn replay(
        &self,
        file: &mut File,
        batch_id: u64,
        commit_offset: u64,
        context: &JsonOperationContext,
        visit: &mut dyn FnMut(CommittedRecord, &mut RecordReservation) -> Result<()>,
    ) -> Result<()> {
        let resume = file.stream_position()?;
        let mut offset = self.first_offset;
        for index in 0..self.count {
            file.seek(SeekFrom::Start(offset))?;
            let mut header = [0; RECORD_HEADER_LEN];
            file.read_exact(&mut header)?;
            if checksum32(&header[..46]) != read_u32(&header[46..50]) {
                return Err(Error::new(
                    ErrorCode::ChecksumMismatch,
                    "critical record changed during recovery replay",
                ));
            }
            let record_type = read_u16(&header[4..6]);
            let payload_offset = offset.checked_add(RECORD_HEADER_LEN as u64);
            let len = read_u64(&header[34..42]);
            let end = payload_offset.and_then(|start| start.checked_add(len));
            let sequence_no = read_u64(&header[24..32]);
            if &header[..4] != b"CFRL"
                || read_u16(&header[6..8]) != 1
                || read_u16(&header[32..34]) != RECORD_HEADER_LEN as u16
                || len.checked_add(RECORD_HEADER_LEN as u64) != Some(read_u64(&header[8..16]))
                || !(COLLECTION_CREATE..=COMPACT_MARKER).contains(&record_type)
                || record_type == COMMIT
                || read_u64(&header[16..24]) != batch_id
                || Some(sequence_no) != self.first_sequence.checked_add(index as u64)
                || end.map_or(true, |end| end > commit_offset)
            {
                return Err(Error::new(
                    ErrorCode::FileFormatError,
                    "invalid critical record during recovery replay",
                ));
            }
            let checksum = u64::from(read_u32(&header[42..46]));
            let record_ref = RecordRef {
                offset,
                record_type: record_type_name(record_type),
                batch_id,
                sequence_no,
                payload: PayloadRef {
                    offset: payload_offset.unwrap(),
                    len,
                    checksum,
                },
            };
            let mut strings = context.workspace_reservation();
            let (payload, json) = payload::read_binary_payload_in(
                file,
                record_type,
                &record_ref.payload,
                false,
                JsonInputKind::StoredJson,
                context,
                Some(&mut strings),
            )
            .map_err(|error| {
                error
                    .with_detail("record_type", record_type_name(record_type))
                    .with_detail("record_offset", offset)
            })?;
            let mut work = RecordReservation::new(json, strings);
            visit(
                CommittedRecord {
                    payload,
                    record_ref,
                    payload_checksum: checksum,
                },
                &mut work,
            )?;
            offset = end.unwrap();
        }
        if offset != commit_offset {
            return Err(Error::new(
                ErrorCode::FileFormatError,
                "recovery batch range changed before replay completed",
            ));
        }
        file.seek(SeekFrom::Start(resume))?;
        Ok(())
    }
}
