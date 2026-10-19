use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::process::Command;

use cofferdb::JsonValue;

struct TrackingAllocator;

thread_local! {
    static MODE: Cell<u8> = const { Cell::new(0) };
    static ALLOCATED: Cell<usize> = const { Cell::new(0) };
    static FREED: Cell<usize> = const { Cell::new(0) };
    static DROP_ALLOCATIONS: Cell<usize> = const { Cell::new(0) };
}

fn allocated(size: usize) {
    let _ = MODE.try_with(|mode| {
        if mode.get() != 0 {
            ALLOCATED.with(|bytes| bytes.set(bytes.get() + size));
        }
        if mode.get() == 2 {
            DROP_ALLOCATIONS.with(|count| count.set(count.get() + 1));
        }
    });
}

fn freed(size: usize) {
    let _ = MODE.try_with(|mode| {
        if mode.get() != 0 {
            FREED.with(|bytes| bytes.set(bytes.get() + size));
        }
    });
}

unsafe impl GlobalAlloc for TrackingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let pointer = unsafe { System.alloc(layout) };
        if !pointer.is_null() {
            allocated(layout.size());
        }
        pointer
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        freed(layout.size());
        unsafe { System.dealloc(pointer, layout) };
    }

    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        let resized = unsafe { System.realloc(pointer, layout, size) };
        if !resized.is_null() {
            freed(layout.size());
            allocated(size);
        }
        resized
    }
}

#[global_allocator]
static ALLOCATOR: TrackingAllocator = TrackingAllocator;

#[test]
fn deep_and_wide_drop_uses_no_allocations_and_releases_all_owned_memory() {
    const WORKER: &str = "COFFERDB_JSON_DROP_WORKER";
    if std::env::var_os(WORKER).is_none() {
        let output = Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "deep_and_wide_drop_uses_no_allocations_and_releases_all_owned_memory",
                "--nocapture",
            ])
            .env(WORKER, "1")
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{:?}: {}",
            output.status,
            String::from_utf8_lossy(&output.stderr)
        );
        return;
    }
    std::thread::Builder::new()
        .stack_size(64 * 1024)
        .spawn(|| {
            for shape in 0..4 {
                ALLOCATED.set(0);
                FREED.set(0);
                DROP_ALLOCATIONS.set(0);
                MODE.set(1);
                let mut value = JsonValue::String("leaf".into());
                for level in 0..30_000 {
                    value = if shape == 0 || (shape == 2 && level % 2 == 0) {
                        JsonValue::Array(vec![JsonValue::String("sibling".into()), value])
                    } else {
                        JsonValue::Object(
                            [
                                ("child".into(), value),
                                ("sibling".into(), JsonValue::String("sibling".into())),
                            ]
                            .into_iter()
                            .collect(),
                        )
                    };
                    if shape == 3 {
                        value = JsonValue::Array(vec![value, JsonValue::object()]);
                    }
                }
                let wide = JsonValue::Array(
                    (0..100_000)
                        .map(|_| JsonValue::String("wide".into()))
                        .collect(),
                );
                let value = JsonValue::Array(vec![value, wide]);
                MODE.set(2);
                drop(value);
                MODE.set(0);
                assert_eq!(DROP_ALLOCATIONS.get(), 0, "shape {shape}");
                assert_eq!(ALLOCATED.get(), FREED.get(), "shape {shape}");
            }
        })
        .unwrap()
        .join()
        .unwrap();
}
