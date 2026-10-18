use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct KeyAllocations {
    pub(crate) allocated: usize,
    pub(crate) live: usize,
    pub(crate) peak: usize,
    pub(crate) overflow: bool,
}

#[derive(Clone, Copy)]
struct State {
    allocations: usize,
    allocation_bytes: u128,
    key_bytes: Option<usize>,
    fail_bytes: Option<usize>,
    fail_ordinal: Option<usize>,
    keys: KeyAllocations,
    pointers: [usize; 16],
}

thread_local! { static STATE: Cell<Option<State>> = const { Cell::new(None) }; }

fn allocated(pointer: *mut u8, bytes: usize, alignment: usize) {
    if pointer.is_null() {
        return;
    }
    let _ = STATE.try_with(|slot| {
        let Some(mut state) = slot.get() else { return };
        state.allocations += 1;
        state.allocation_bytes += bytes as u128;
        if state.key_bytes == Some(bytes) && alignment == 1 {
            state.keys.allocated += 1;
            if let Some(empty) = state.pointers.iter_mut().find(|address| **address == 0) {
                *empty = pointer as usize;
                state.keys.live += 1;
                state.keys.peak = state.keys.peak.max(state.keys.live);
            } else {
                state.keys.overflow = true;
            }
        }
        slot.set(Some(state));
    });
}

fn released(pointer: *mut u8) {
    let _ = STATE.try_with(|slot| {
        let Some(mut state) = slot.get().filter(|state| state.key_bytes.is_some()) else {
            return;
        };
        if let Some(address) = state
            .pointers
            .iter_mut()
            .find(|address| **address == pointer as usize)
        {
            *address = 0;
            state.keys.live -= 1;
            slot.set(Some(state));
        }
    });
}

struct AllocationCounter;

fn refused(bytes: usize, alignment: usize) -> bool {
    STATE
        .try_with(|slot| {
            let Some(mut state) = slot.get() else {
                return false;
            };
            let byte_match = alignment == 1 && state.fail_bytes == Some(bytes);
            let ordinal_match = state.fail_ordinal == Some(state.allocations + 1);
            if !byte_match && !ordinal_match {
                return false;
            }
            if byte_match {
                state.fail_bytes = None;
            }
            if ordinal_match {
                state.fail_ordinal = None;
            }
            slot.set(Some(state));
            true
        })
        .unwrap_or(false)
}

unsafe impl GlobalAlloc for AllocationCounter {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if refused(layout.size(), layout.align()) {
            return std::ptr::null_mut();
        }
        let pointer = unsafe { System.alloc(layout) };
        allocated(pointer, layout.size(), layout.align());
        pointer
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        if refused(layout.size(), layout.align()) {
            return std::ptr::null_mut();
        }
        let pointer = unsafe { System.alloc_zeroed(layout) };
        allocated(pointer, layout.size(), layout.align());
        pointer
    }

    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        if refused(size, layout.align()) {
            return std::ptr::null_mut();
        }
        let output = unsafe { System.realloc(pointer, layout, size) };
        if !output.is_null() {
            released(pointer);
            allocated(output, size, layout.align());
        }
        output
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        released(pointer);
        unsafe { System.dealloc(pointer, layout) };
    }
}

#[global_allocator]
static ALLOCATOR: AllocationCounter = AllocationCounter;

fn measure<T>(
    key_bytes: Option<usize>,
    fail_bytes: Option<usize>,
    fail_ordinal: Option<usize>,
    action: impl FnOnce() -> T,
) -> (T, State) {
    struct Reset;
    impl Drop for Reset {
        fn drop(&mut self) {
            STATE.set(None);
        }
    }
    assert!(STATE.get().is_none(), "nested allocation measurement");
    STATE.set(Some(State {
        allocations: 0,
        allocation_bytes: 0,
        key_bytes,
        fail_bytes,
        fail_ordinal,
        keys: KeyAllocations::default(),
        pointers: [0; 16],
    }));
    let reset = Reset;
    let result = action();
    let state = STATE.get().unwrap();
    drop(reset);
    (result, state)
}

pub(crate) fn allocations<T>(action: impl FnOnce() -> T) -> (T, usize) {
    let (result, state) = measure(None, None, None, action);
    (result, state.allocations)
}

pub(crate) fn allocation_bytes<T>(action: impl FnOnce() -> T) -> (T, u128) {
    let (result, state) = measure(None, None, None, action);
    (result, state.allocation_bytes)
}

// Tracks only matching u8 allocations created during this thread's measurement.
// Pre-existing buffers, other layouts and other threads cannot change the peak.
pub(crate) fn key_allocations<T>(bytes: usize, action: impl FnOnce() -> T) -> (T, KeyAllocations) {
    let (result, state) = measure(Some(bytes), None, None, action);
    (result, state.keys)
}

// Size and alignment identify the target byte allocation, not its Rust type.
// The target must be fallible: an infallible allocation aborts on a null result.
pub(crate) fn fail_byte_allocation_once<T>(bytes: usize, action: impl FnOnce() -> T) -> T {
    let (result, state) = measure(None, Some(bytes), None, action);
    assert!(
        state.fail_bytes.is_none(),
        "target allocation was not attempted"
    );
    result
}

// Setup must happen outside the window; the selected allocation must be fallible.
pub(crate) fn fail_nth_allocation_once<T>(ordinal: usize, action: impl FnOnce() -> T) -> T {
    assert!(ordinal != 0);
    let (result, state) = measure(None, None, Some(ordinal), action);
    assert!(
        state.fail_ordinal.is_none(),
        "target allocation was not attempted"
    );
    result
}

#[test]
fn ordinal_failure_matches_nonbyte_allocations_and_resets_after_unwinding() {
    fail_nth_allocation_once(2, || {
        let mut before = Vec::<u64>::new();
        before.try_reserve_exact(17).unwrap();
        before.resize(17, 7);
        std::hint::black_box(before.as_slice());
        let mut failed = Vec::<u64>::new();
        assert!(failed.try_reserve_exact(31).is_err());
        failed.try_reserve_exact(31).unwrap();
        assert_eq!(before.capacity(), 17);
        assert_eq!(failed.capacity(), 31);
    });
    let panic = std::panic::catch_unwind(|| {
        fail_nth_allocation_once(usize::MAX, || panic!("injected ordinal window unwind"));
    });
    assert!(panic.is_err());
    assert!(STATE.get().is_none());
    let mut after = Vec::<u64>::new();
    after.try_reserve_exact(31).unwrap();
}

#[test]
fn byte_allocation_failure_is_one_shot_and_resets_after_unwinding() {
    let bytes = 8193;
    fail_byte_allocation_once(bytes, || {
        let other = Layout::from_size_align(bytes, 8).unwrap();
        let target = Layout::from_size_align(bytes, 1).unwrap();
        unsafe {
            // Exercise the allocator interface directly: standard allocation
            // entry points permit the optimizer to elide an unused allocation.
            let other_pointer = ALLOCATOR.alloc(other);
            assert!(!other_pointer.is_null());
            assert!(ALLOCATOR.alloc(target).is_null());
            let target_pointer = ALLOCATOR.alloc_zeroed(target);
            assert!(!target_pointer.is_null());
            ALLOCATOR.dealloc(target_pointer, target);
            ALLOCATOR.dealloc(other_pointer, other);
        }
    });
    fail_byte_allocation_once(bytes, || {
        let target = Layout::from_size_align(bytes, 1).unwrap();
        unsafe {
            assert!(ALLOCATOR.alloc_zeroed(target).is_null());
            let pointer = ALLOCATOR.alloc(target);
            assert!(!pointer.is_null());
            ALLOCATOR.dealloc(pointer, target);
        }
    });
    let panic = std::panic::catch_unwind(|| {
        fail_byte_allocation_once(bytes, || panic!("injected window unwind"));
    });
    assert!(panic.is_err());
    assert!(STATE.get().is_none());
    let mut data = Vec::<u8>::new();
    data.try_reserve_exact(bytes).unwrap();
    assert_eq!(data.capacity(), bytes);
}

#[test]
fn key_counter_distinguishes_existing_buffers_reallocation_and_release() {
    for size in 8193..8293usize {
        let existing = vec![7u8; size];
        let (_, observed) = key_allocations(size, || {
            drop(existing);
            let first = std::hint::black_box(vec![17u8; size]);
            let mut second = std::hint::black_box(vec![31u8; size + 1]);
            second.truncate(size);
            second.shrink_to_fit();
            std::hint::black_box(&second);
            drop(first);
            second.reserve_exact(size);
            std::hint::black_box(&second);
        });
        assert_eq!(observed.allocated, 2);
        assert_eq!(observed.live, 0);
        assert_eq!(observed.peak, 2);
        assert!(!observed.overflow);
    }
}
