/* Querying-thread userspace reads; this probe does not measure physical I/O. */
#include <sys/syscall.h>
#include <errno.h>
#include <limits.h>
#include <stdint.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/uio.h>
#include <unistd.h>

typedef struct { uint64_t calls, requested, returned, errors, saturated; } Probe;
static _Thread_local Probe stats;
static _Thread_local int active;
static void add(uint64_t *value, uint64_t amount) {
    if (UINT64_MAX - *value < amount) { *value = UINT64_MAX; stats.saturated = 1; }
    else *value += amount;
}
static void record(uint64_t requested, ssize_t result) {
    if (!active) return;
    add(&stats.calls, 1); add(&stats.requested, requested);
    if (result >= 0) add(&stats.returned, (uint64_t)result);
    else add(&stats.errors, 1);
}
void supplement_read_begin(void) { stats = (Probe){0}; active = 1; }
Probe supplement_read_end(void) { active = 0; return stats; }
static ssize_t wrapped_read(int fd, void *buf, size_t count) {
    ssize_t result = syscall(SYS_read, fd, buf, count); int saved = errno;
    record(count, result); errno = saved; return result;
}
static ssize_t wrapped_pread(int fd, void *buf, size_t count, off_t offset) {
    ssize_t result = syscall(SYS_pread, fd, buf, count, offset); int saved = errno;
    record(count, result); errno = saved; return result;
}
static ssize_t wrapped_readv(int fd, const struct iovec *iov, int count) {
    ssize_t result = syscall(SYS_readv, fd, iov, count); int saved = errno;
    uint64_t requested = 0;
    /* Failed calls may have an invalid iovec pointer; never inspect it then. */
    if (active && result >= 0) for (int i = 0; i < count; i++) add(&requested, iov[i].iov_len);
    record(requested, result); errno = saved; return result;
}
#define INTERPOSE(replacement, original) \
__attribute__((used)) static struct { const void *new_fn; const void *old_fn; } pair_##original \
__attribute__((section("__DATA,__interpose"))) = {(const void *)&replacement, (const void *)&original};
INTERPOSE(wrapped_read, read)
INTERPOSE(wrapped_pread, pread)
INTERPOSE(wrapped_readv, readv)
