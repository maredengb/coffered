#include <assert.h>
#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <sys/uio.h>
#include <unistd.h>
typedef struct { uint64_t calls, requested, returned, errors, saturated; } Probe;
extern void supplement_read_begin(void);
extern Probe supplement_read_end(void);
int main(void) {
    FILE *f = tmpfile(); assert(f); assert(fwrite("abcdef",1,6,f)==6); assert(fflush(f)==0);
    int fd=fileno(f); char b[8]={0}; assert(lseek(fd,0,SEEK_SET)==0);
    supplement_read_begin();
    assert(read(fd,b,2)==2 && memcmp(b,"ab",2)==0);
    assert(pread(fd,b,2,2)==2 && memcmp(b,"cd",2)==0);
    struct iovec v[2]={{b,1},{b+1,1}};
    assert(readv(fd,v,2)==2 && memcmp(b,"cd",2)==0);
    assert(read(-1,b,1)==-1 && errno==EBADF);
    Probe p=supplement_read_end();
    assert(p.calls==4 && p.requested==7 && p.returned==6 && p.errors==1 && p.saturated==0);
    assert(read(fd,b,1)==1); Probe q=supplement_read_end();assert(memcmp(&p,&q,sizeof p)==0);
    supplement_read_begin(); q=supplement_read_end();assert(q.calls==0);
    fclose(f);puts("PASS: actual read/pread/readv and error/disabled/reset controls; no engine query");return 0;
}
