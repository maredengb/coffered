use super::*;

const INDEX_ENTRY_BYTES: u64 = 16;
const PAGE_BYTES: usize = 2048;
const CACHE_PAGES: usize = 16;

fn query_workspace(key: u128) -> u128 {
    (MIN_WORKSPACE as u128)
        .max(2 * key + ENTRY_HEADER_BYTES as u128 + std::mem::size_of::<Entry>() as u128)
}

pub(super) fn preferred_workspace(key: u128) -> u128 {
    required_workspace(key).max(
        2 * key
            + ENTRY_HEADER_BYTES as u128
            + std::mem::size_of::<Entry>() as u128
            + CACHE_PAGES as u128 * (PAGE_BYTES + std::mem::size_of::<Page>()) as u128,
    )
}

pub(super) fn required_workspace(key: u128) -> u128 {
    query_workspace(key).max(2 * IO_BYTES as u128 + key + std::mem::size_of::<Entry>() as u128)
}

pub(crate) struct ExternalKeyLookup {
    files: Option<LookupFiles>,
    max_key: usize,
    buffer: Vec<u8>,
    last: Option<Entry>,
    pages: Vec<Page>,
    _work: WorkspaceReservation,
}

struct Page {
    index: bool,
    offset: u64,
    bytes: Vec<u8>,
}

struct LookupFiles {
    data: TemporaryFile,
    _data_temporary: TempFileGuard,
    index: TemporaryFile,
    _index_temporary: TempFileGuard,
    count: u64,
    data_bytes: u64,
}

impl ExternalKeyLookup {
    pub(super) fn new(
        run: Option<Run>,
        directory: &Path,
        disk: Arc<TemporaryDiskBudget>,
        max_key: usize,
        work: WorkspaceReservation,
    ) -> Result<Self> {
        let mut lookup = Self {
            files: None,
            max_key,
            buffer: Vec::new(),
            last: None,
            pages: Vec::new(),
            _work: work,
        };
        let Some(run) = run else { return Ok(lookup) };
        let count = run.count;
        let data_bytes = run.bytes;
        let mut cursor = RunCursor::new(run, max_key)?;
        let (file, mut temporary) =
            create_budgeted_sort_file(directory, "import-lookup", disk, lookup._work.context())?;
        temporary.reserve_disk(
            count
                .checked_mul(INDEX_ENTRY_BYTES)
                .ok_or_else(invalid_run)?,
        )?;
        let mut output = BufWriter::with_capacity(IO_BYTES, file);
        let mut offset = 0u64;
        while let Some(entry) = cursor.next()? {
            let header = entry.header()?;
            let length =
                u32::try_from(ENTRY_HEADER_BYTES + entry.key.len()).map_err(|_| invalid_run())?;
            let mut crc = Crc32c::default();
            crc.update(&header);
            crc.update(&entry.key);
            output.write_all(&offset.to_le_bytes())?;
            output.write_all(&length.to_le_bytes())?;
            output.write_all(&crc.finish().to_le_bytes())?;
            offset = offset
                .checked_add(u64::from(length))
                .ok_or_else(invalid_run)?;
        }
        output.flush()?;
        lookup.files = Some(LookupFiles {
            data: cursor.input.into_inner(),
            _data_temporary: cursor._temporary,
            index: output.into_inner().map_err(|error| error.into_error())?,
            _index_temporary: temporary,
            count,
            data_bytes,
        });
        let retained = lookup
            ._work
            .reserved_bytes()
            .min(preferred_workspace(max_key as u128) as usize);
        lookup._work.retain(retained);
        Ok(lookup)
    }

    pub(crate) fn get(&mut self, key: &[u8]) -> Result<Option<(u64, u64)>> {
        if let Some(last) = &self.last {
            if last.key == key {
                return Ok(Some((last.line, last.value)));
            }
        }
        let Some(files) = &mut self.files else {
            return Ok(None);
        };
        let cache_bytes = self._work.reserved_bytes()
            - (2 * self.max_key + ENTRY_HEADER_BYTES + std::mem::size_of::<Entry>());
        let mut low = 0;
        let mut high = files.count;
        while low < high {
            let middle = low + (high - low) / 2;
            let mut index = [0; INDEX_ENTRY_BYTES as usize];
            read_cached(
                &mut self.pages,
                &mut files.index,
                true,
                files.count * INDEX_ENTRY_BYTES,
                &mut index,
                middle * INDEX_ENTRY_BYTES,
                cache_bytes,
            )?;
            let offset = u64::from_le_bytes(index[..8].try_into().unwrap());
            let length = u32::from_le_bytes(index[8..12].try_into().unwrap()) as usize;
            let checksum = u32::from_le_bytes(index[12..].try_into().unwrap());
            if length < ENTRY_HEADER_BYTES
                || length - ENTRY_HEADER_BYTES > self.max_key
                || offset
                    .checked_add(length as u64)
                    .filter(|end| *end <= files.data_bytes)
                    .is_none()
            {
                return Err(invalid_run());
            }
            if length > self.buffer.capacity() {
                self.buffer = Vec::new();
                self.buffer
                    .try_reserve_exact(length)
                    .map_err(allocation_error)?;
            }
            if self.buffer.capacity() > self.max_key + ENTRY_HEADER_BYTES {
                return Err(
                    std::io::Error::other("lookup buffer exceeds admitted capacity").into(),
                );
            }
            self.buffer.resize(length, 0);
            read_cached(
                &mut self.pages,
                &mut files.data,
                false,
                files.data_bytes,
                &mut self.buffer,
                offset,
                cache_bytes,
            )?;
            let mut crc = Crc32c::default();
            crc.update(&self.buffer);
            if crc.finish() != checksum
                || u32::from_le_bytes(self.buffer[..4].try_into().unwrap()) as usize
                    != length - ENTRY_HEADER_BYTES
            {
                return Err(invalid_run());
            }
            let stored_key = &self.buffer[ENTRY_HEADER_BYTES..];
            match stored_key.cmp(key) {
                std::cmp::Ordering::Less => low = middle + 1,
                std::cmp::Ordering::Greater => high = middle,
                std::cmp::Ordering::Equal => {
                    let line = u64::from_le_bytes(self.buffer[4..12].try_into().unwrap());
                    let value = u64::from_le_bytes(self.buffer[12..20].try_into().unwrap());
                    let mut saved = self.last.take().map(|entry| entry.key).unwrap_or_default();
                    if saved.capacity() < stored_key.len() {
                        saved = Vec::new();
                    }
                    saved.clear();
                    if saved.try_reserve_exact(stored_key.len()).is_err()
                        || saved.capacity() > self.max_key
                    {
                        return Ok(Some((line, value)));
                    }
                    saved.extend_from_slice(stored_key);
                    self.last = Some(Entry {
                        key: saved,
                        line,
                        value,
                    });
                    return Ok(Some((line, value)));
                }
            }
        }
        Ok(None)
    }
}

fn read_cached(
    pages: &mut Vec<Page>,
    file: &mut File,
    index: bool,
    file_bytes: u64,
    mut output: &mut [u8],
    mut offset: u64,
    cache_bytes: usize,
) -> Result<()> {
    if offset
        .checked_add(output.len() as u64)
        .filter(|end| *end <= file_bytes)
        .is_none()
    {
        return Err(invalid_run());
    }
    let page_limit = (cache_bytes / (PAGE_BYTES + std::mem::size_of::<Page>())).min(CACHE_PAGES);
    if pages.capacity() > page_limit {
        *pages = Vec::new();
    }
    if page_limit == 0 {
        return read_at(file, output, offset);
    }
    while !output.is_empty() {
        let start = offset / PAGE_BYTES as u64 * PAGE_BYTES as u64;
        let found = pages
            .iter()
            .position(|page| page.index == index && page.offset == start);
        let mut page = if let Some(position) = found {
            pages.remove(position)
        } else if pages.len() == page_limit {
            pages.remove(0)
        } else {
            if pages.capacity() < page_limit
                && (pages.try_reserve_exact(page_limit - pages.len()).is_err()
                    || pages.capacity() > page_limit)
            {
                // Cache population is optional; direct reads preserve the
                // same checksums and lookup semantics without a cache entry.
                *pages = Vec::new();
                return read_at(file, output, offset);
            }
            Page {
                index,
                offset: start,
                bytes: Vec::new(),
            }
        };
        if found.is_none() {
            let length = (file_bytes - start).min(PAGE_BYTES as u64) as usize;
            if page.bytes.capacity() < length {
                page.bytes = Vec::new();
            }
            if page.bytes.capacity() < length
                && (page
                    .bytes
                    .try_reserve_exact(length - page.bytes.len())
                    .is_err()
                    || page.bytes.capacity() > PAGE_BYTES)
            {
                return read_at(file, output, offset);
            }
            page.bytes.resize(length, 0);
            read_at(file, &mut page.bytes, start)?;
            page.index = index;
            page.offset = start;
        }
        let within = (offset - start) as usize;
        let count = output.len().min(page.bytes.len() - within);
        output[..count].copy_from_slice(&page.bytes[within..within + count]);
        offset += count as u64;
        output = &mut output[count..];
        pages.push(page);
    }
    Ok(())
}

fn read_at(file: &mut File, bytes: &mut [u8], offset: u64) -> Result<()> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::FileExt;
        file.read_exact_at(bytes, offset).map_err(read_error)?;
    }
    #[cfg(not(unix))]
    {
        file.seek(std::io::SeekFrom::Start(offset))?;
        file.read_exact(bytes).map_err(read_error)?;
    }
    Ok(())
}

fn read_error(error: std::io::Error) -> Error {
    if error.kind() == std::io::ErrorKind::UnexpectedEof {
        invalid_run()
    } else {
        error.into()
    }
}

#[cfg(test)]
mod tests {
    use super::super::tests::{sorter, TestDirectory};
    use super::*;
    use std::collections::BTreeMap;
    use std::io::SeekFrom;

    #[test]
    fn optional_pages_do_not_raise_the_required_merge_workspace() {
        let directory = TestDirectory::new("lookup-optional-minimum");
        let disk = Arc::new(TemporaryDiskBudget::new(1024 * 1024));
        for extra in 0..100usize {
            let size = 19 * 1024 + extra;
            let mut small = vec![31; 3];
            let mut large = vec![31; size];
            large[size - 1] = 32;
            let mut second = large.clone();
            second[size - 1] = 33;
            let mut keys = sorter(&directory, disk.clone());
            keys.push_value(&[&large], 7, 11).unwrap();
            keys.flush_run().unwrap();
            keys.push_value(&[&second], 19, 23).unwrap();
            keys.flush_run().unwrap();
            keys.push_value(&[&small], 13, 17).unwrap();
            let mut lookup = keys.finish_lookup().unwrap();
            assert_eq!(lookup._work.reserved_bytes(), MIN_WORKSPACE);
            assert_eq!(lookup.get(&small).unwrap(), Some((13, 17)));
            assert_eq!(lookup.get(&large).unwrap(), Some((7, 11)));
            assert_eq!(lookup.get(&second).unwrap(), Some((19, 23)));
            let held = lookup.buffer.capacity()
                + lookup.last.as_ref().unwrap().key.capacity()
                + std::mem::size_of::<Entry>()
                + lookup.pages.capacity() * std::mem::size_of::<Page>()
                + lookup
                    .pages
                    .iter()
                    .map(|page| page.bytes.capacity())
                    .sum::<usize>();
            assert!(held <= lookup._work.reserved_bytes());
            let page_limit =
                (MIN_WORKSPACE - 2 * size - ENTRY_HEADER_BYTES - std::mem::size_of::<Entry>())
                    / (PAGE_BYTES + std::mem::size_of::<Page>());
            assert!(lookup.pages.capacity() <= page_limit);
            assert!(lookup.pages.len() < CACHE_PAGES);
            small.push(30);
            assert_eq!(lookup.get(&small).unwrap(), None);
            drop(lookup);
            directory.assert_clean(&disk);
        }
    }

    #[test]
    fn bounded_page_cache_handles_cross_page_reads_eviction_and_file_identity() {
        let directory = TestDirectory::new("lookup-pages");
        let disk = Arc::new(TemporaryDiskBudget::new(2 * 1024 * 1024));
        let mut sources = Vec::new();
        for index in 0..2 {
            let bytes = (0..PAGE_BYTES * 35 + 7)
                .map(|offset| (offset.wrapping_mul(31 + index * 6) ^ (offset >> 7)) as u8)
                .collect::<Vec<_>>();
            let (mut file, mut guard) =
                create_budgeted_temporary_file_in(&directory.0, "cache-test", "tmp", disk.clone())
                    .unwrap();
            guard.reserve_disk(bytes.len() as u64).unwrap();
            file.write_all(&bytes).unwrap();
            sources.push((file, guard, bytes));
        }
        let mut pages = Vec::new();
        for seed in 0..100 {
            for length in [
                0,
                1,
                PAGE_BYTES - 1,
                PAGE_BYTES,
                PAGE_BYTES + 1,
                PAGE_BYTES * 4,
            ] {
                for (index, (file, _, expected)) in sources.iter_mut().enumerate() {
                    let offset = seed * 997 % (expected.len() - length + 1);
                    let mut actual = vec![0; length];
                    read_cached(
                        &mut pages,
                        file,
                        index == 1,
                        expected.len() as u64,
                        &mut actual,
                        offset as u64,
                        64 * 1024,
                    )
                    .unwrap();
                    assert_eq!(actual, expected[offset..offset + length]);
                    assert!(pages.len() <= CACHE_PAGES);
                    assert!(pages.capacity() <= CACHE_PAGES);
                    assert!(pages.iter().all(|page| page.bytes.capacity() <= PAGE_BYTES));
                }
            }
        }
        let (file, _, expected) = &mut sources[0];
        for offset in [expected.len() as u64, u64::MAX] {
            assert_eq!(
                read_cached(
                    &mut pages,
                    file,
                    false,
                    expected.len() as u64,
                    &mut [0],
                    offset,
                    64 * 1024
                )
                .unwrap_err()
                .code(),
                ErrorCode::ChecksumMismatch
            );
        }
        drop(sources);
        directory.assert_clean(&disk);
    }

    #[test]
    fn shrinking_or_disabling_optional_cache_preserves_reads_and_integrity_checks() {
        let directory = TestDirectory::new("lookup-cache-fallback");
        let disk = Arc::new(TemporaryDiskBudget::new(128 * 1024));
        let (mut file, mut guard) =
            create_budgeted_temporary_file_in(&directory.0, "cache-fallback", "tmp", disk.clone())
                .unwrap();
        let bytes = (0..PAGE_BYTES * 20 + 9)
            .map(|n| (n * 31) as u8)
            .collect::<Vec<_>>();
        guard.reserve_disk(bytes.len() as u64).unwrap();
        file.write_all(&bytes).unwrap();
        let mut pages = Vec::new();
        for seed in 0..100 {
            for workspace in [65536, 0, 1, 8192, 65536] {
                let start = seed * 127 % (bytes.len() - PAGE_BYTES * 2);
                let mut output = vec![0; PAGE_BYTES * 2];
                read_cached(
                    &mut pages,
                    &mut file,
                    false,
                    bytes.len() as u64,
                    &mut output,
                    start as u64,
                    workspace,
                )
                .unwrap();
                assert_eq!(output, bytes[start..start + output.len()]);
                let held = pages.capacity() * std::mem::size_of::<Page>()
                    + pages
                        .iter()
                        .map(|page| page.bytes.capacity())
                        .sum::<usize>();
                assert!(held <= workspace);
            }
        }
        for workspace in [0, 65536] {
            let error = read_cached(
                &mut pages,
                &mut file,
                false,
                bytes.len() as u64,
                &mut [0],
                bytes.len() as u64,
                workspace,
            )
            .unwrap_err();
            assert_eq!(error.code(), ErrorCode::ChecksumMismatch);
        }
        drop((file, guard));
        directory.assert_clean(&disk);
    }

    #[test]
    fn disk_lookup_matches_100_independent_shuffled_maps_with_bounded_buffers() {
        let directory = TestDirectory::new("lookup-oracle");
        let disk = Arc::new(TemporaryDiskBudget::new(8 * 1024 * 1024));
        for seed in 0..100u64 {
            let mut state = seed + 1;
            let mut random = || {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                state
            };
            let mut expected = BTreeMap::new();
            for number in 0..257u64 {
                let mut key = number.to_be_bytes().to_vec();
                key.resize(128 + (random() % 256) as usize, random() as u8);
                expected.insert(key, (number * 3 + 1, random()));
            }
            expected.insert(Vec::new(), (u64::MAX, 0));
            expected.insert(vec![255; 8192], (1, u64::MAX));
            let mut input = expected.iter().collect::<Vec<_>>();
            for index in (1..input.len()).rev() {
                input.swap(index, random() as usize % (index + 1));
            }
            let mut keys = sorter(&directory, disk.clone());
            for (key, (line, value)) in &input {
                keys.push_value(&[key], *line, *value).unwrap();
            }
            let mut lookup = keys.finish_lookup().unwrap();
            let files = lookup.files.as_ref().unwrap();
            assert_eq!(
                disk.used(),
                files.data_bytes + files.count * INDEX_ENTRY_BYTES
            );
            for (key, value) in input.into_iter().rev() {
                assert_eq!(lookup.get(key).unwrap(), Some(*value), "seed={seed}");
                assert_eq!(lookup.get(key).unwrap(), Some(*value));
                assert!(lookup.buffer.capacity() <= lookup.max_key + ENTRY_HEADER_BYTES);
                assert!(lookup.last.as_ref().unwrap().key.capacity() <= lookup.max_key);
                assert!(lookup.pages.len() <= CACHE_PAGES);
                assert!(
                    lookup
                        .pages
                        .iter()
                        .map(|page| page.bytes.capacity())
                        .sum::<usize>()
                        <= CACHE_PAGES * PAGE_BYTES
                );
            }
            for missing in [b"missing".as_slice(), &[255; 2], &[0; 8193]] {
                assert_eq!(lookup.get(missing).unwrap(), None);
            }
            drop(lookup);
            directory.assert_clean(&disk);
        }
    }

    #[test]
    fn lookup_construction_enforces_exact_disk_quota_and_empty_lookup_needs_no_file() {
        let directory = TestDirectory::new("lookup-quota");
        for limit in [22, 23, 38, 39] {
            let disk = Arc::new(TemporaryDiskBudget::new(limit));
            let mut keys = sorter(&directory, disk.clone());
            keys.push_value(&[b"key"], 1, 5).unwrap();
            match keys.finish_lookup() {
                Ok(mut lookup) => {
                    assert_eq!(limit, 39);
                    assert_eq!(disk.used(), 39);
                    assert_eq!(lookup.get(b"key").unwrap(), Some((1, 5)));
                }
                Err(error) => {
                    assert!(limit < 39);
                    assert_eq!(error.code(), ErrorCode::ResourceExhausted);
                }
            }
            directory.assert_clean(&disk);
        }
        let disk = Arc::new(TemporaryDiskBudget::new(0));
        let mut empty = sorter(&directory, disk.clone()).finish_lookup().unwrap();
        assert_eq!(empty.get(b"anything").unwrap(), None);
        assert!(empty.files.is_none());
        directory.assert_clean(&disk);
    }

    #[test]
    fn lookup_index_offsets_lengths_and_record_values_are_checked_before_use() {
        let directory = TestDirectory::new("lookup-corruption");
        let disk = Arc::new(TemporaryDiskBudget::new(1024));
        for damage in 0..7 {
            let mut keys = sorter(&directory, disk.clone());
            for (line, key) in [b"a", b"b", b"c"].into_iter().enumerate() {
                keys.push_value(&[key], line as u64 + 1, line as u64 + 10)
                    .unwrap();
            }
            let mut lookup = keys.finish_lookup().unwrap();
            let files = lookup.files.as_mut().unwrap();
            match damage {
                0 => {
                    files.index.rewind().unwrap();
                    files.index.write_all(&u64::MAX.to_le_bytes()).unwrap();
                }
                1 => {
                    files.index.seek(SeekFrom::Start(8)).unwrap();
                    files.index.write_all(&u32::MAX.to_le_bytes()).unwrap();
                }
                2 => {
                    files.index.seek(SeekFrom::Start(12)).unwrap();
                    files.index.write_all(&[0; 4]).unwrap();
                }
                3 => files.index.set_len(0).unwrap(),
                4 => files.data.set_len(20).unwrap(),
                5 => {
                    files.data.seek(SeekFrom::Start(12)).unwrap();
                    files.data.write_all(&u64::MAX.to_le_bytes()).unwrap();
                }
                _ => {
                    files
                        .data
                        .seek(SeekFrom::Start(ENTRY_HEADER_BYTES as u64))
                        .unwrap();
                    files.data.write_all(b"z").unwrap();
                }
            }
            assert_eq!(
                lookup.get(b"a").unwrap_err().code(),
                ErrorCode::ChecksumMismatch,
                "damage={damage}"
            );
            assert!(lookup.buffer.capacity() <= lookup.max_key + ENTRY_HEADER_BYTES);
            drop(lookup);
            directory.assert_clean(&disk);
        }
    }
}
