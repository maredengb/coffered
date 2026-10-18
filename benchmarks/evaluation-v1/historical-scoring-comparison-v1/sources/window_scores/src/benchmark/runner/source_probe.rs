use std::cell::Cell;
use std::fs::File;
use std::io::{self, Read};

#[derive(Clone, Copy)]
pub(super) enum Fault {
    ReadAfter(usize),
    CorruptAt(usize),
}

#[derive(Clone, Copy, Default)]
pub(super) struct State {
    pub(super) opened: [usize; 3],
    pub(super) verified: [usize; 3],
    pub(super) bytes: [usize; 3],
    pub(super) injected: usize,
    fault: Option<(usize, usize, Fault)>,
}

thread_local! {
    static STATE: Cell<Option<State>> = const { Cell::new(None) };
}

fn kind_index(kind: &str) -> Option<usize> {
    match kind {
        "documents" => Some(0),
        "chunks" => Some(1),
        "embeddings" => Some(2),
        _ => None,
    }
}

pub(super) fn measure<T>(
    fault: Option<(usize, usize, Fault)>,
    action: impl FnOnce() -> T,
) -> (T, State) {
    struct Reset;
    impl Drop for Reset {
        fn drop(&mut self) {
            STATE.set(None);
        }
    }
    assert!(STATE.get().is_none(), "nested source measurement");
    STATE.set(Some(State {
        fault,
        ..State::default()
    }));
    let reset = Reset;
    let value = action();
    let state = STATE.get().unwrap();
    drop(reset);
    (value, state)
}

pub(super) fn verified(kind: &str) {
    if let (Some(mut state), Some(index)) = (STATE.get(), kind_index(kind)) {
        state.verified[index] += 1;
        STATE.set(Some(state));
    }
}

pub(super) struct Source {
    file: File,
    kind: Option<usize>,
    ordinal: usize,
    bytes: usize,
}

impl Source {
    pub(super) fn new(file: File, kind: &str) -> Self {
        let kind = kind_index(kind);
        let mut ordinal = 0;
        if let (Some(mut state), Some(index)) = (STATE.get(), kind) {
            state.opened[index] += 1;
            ordinal = state.opened[index];
            STATE.set(Some(state));
        }
        Self {
            file,
            kind,
            ordinal,
            bytes: 0,
        }
    }
}

impl Read for Source {
    fn read(&mut self, output: &mut [u8]) -> io::Result<usize> {
        if output.is_empty() {
            return Ok(0);
        }
        let fault = STATE
            .get()
            .and_then(|state| state.fault)
            .filter(|&(kind, ordinal, _)| Some(kind) == self.kind && ordinal == self.ordinal)
            .map(|(_, _, fault)| fault);
        let limit = match fault {
            Some(Fault::ReadAfter(offset)) if self.bytes >= offset => {
                let mut state = STATE.get().unwrap();
                state.injected += 1;
                STATE.set(Some(state));
                return Err(io::Error::other("source probe read failure"));
            }
            Some(Fault::ReadAfter(offset)) => output.len().min(offset - self.bytes),
            _ => output.len(),
        };
        let read = self.file.read(&mut output[..limit])?;
        if let Some(Fault::CorruptAt(offset)) = fault {
            if (self.bytes..self.bytes + read).contains(&offset) {
                assert!(output[offset - self.bytes].is_ascii_whitespace());
                output[offset - self.bytes] = b'\t';
                let mut state = STATE.get().unwrap();
                state.injected += 1;
                STATE.set(Some(state));
            }
        }
        self.bytes += read;
        if let (Some(mut state), Some(index)) = (STATE.get(), self.kind) {
            state.bytes[index] += read;
            STATE.set(Some(state));
        }
        Ok(read)
    }
}
