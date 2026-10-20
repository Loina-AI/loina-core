#include "loina_internal.h"
#include <string.h>
#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#include <sys/statvfs.h>
#ifdef __linux__
#include <sys/sysinfo.h>
#endif
#ifdef __APPLE__
#include <sys/sysctl.h>
#include <mach/mach.h>
#endif
#endif

loina_status loina_device_probe(const char *directory, loina_device *d, loina_error *error) {
    if (!directory || !d) return loina_fail(error, LOINA_INVALID, "Missing device probe arguments");
    memset(d, 0, sizeof(*d));
#ifdef _WIN32
    MEMORYSTATUSEX m = {.dwLength = sizeof(m)};
    SYSTEM_INFO s;
    ULARGE_INTEGER available;
    GetSystemInfo(&s);
    d->threads = (int)s.dwNumberOfProcessors;
    if (GlobalMemoryStatusEx(&m)) { d->ram_bytes=m.ullTotalPhys; d->available_ram_bytes=m.ullAvailPhys; }
    if (GetDiskFreeSpaceExA(directory, &available, NULL, NULL)) d->storage_bytes=available.QuadPart;
#else
    d->threads=(int)sysconf(_SC_NPROCESSORS_ONLN);
#ifdef __linux__
    struct sysinfo s;
    if (!sysinfo(&s)) {
        d->ram_bytes=(uint64_t)s.totalram*s.mem_unit;
        d->available_ram_bytes=(uint64_t)s.freeram*s.mem_unit;
    }
    /* Cached pages are reclaimable; match Android's available-memory probe. */
    FILE *info=fopen("/proc/meminfo","r");
    if (info) {
        char line[128];unsigned long long kb;
        while (fgets(line,sizeof(line),info)) if (sscanf(line,"MemAvailable: %llu",&kb)==1) {
            if (kb<=d->ram_bytes/1024) d->available_ram_bytes=(uint64_t)kb*1024;
            break;
        }
        fclose(info);
    }
#elif defined(__APPLE__)
    size_t size=sizeof(d->ram_bytes);
    sysctlbyname("hw.memsize",&d->ram_bytes,&size,NULL,0);
    vm_statistics64_data_t vm;
    mach_msg_type_number_t count=HOST_VM_INFO64_COUNT;
    if (host_statistics64(mach_host_self(),HOST_VM_INFO64,(host_info64_t)&vm,&count)==KERN_SUCCESS)
        d->available_ram_bytes=(uint64_t)(vm.free_count+vm.inactive_count)*sysconf(_SC_PAGESIZE);
#else
    d->ram_bytes=(uint64_t)sysconf(_SC_PHYS_PAGES)*sysconf(_SC_PAGESIZE);
    d->available_ram_bytes=d->ram_bytes/2;
#endif
    struct statvfs v;
    if (!statvfs(directory,&v)) d->storage_bytes=(uint64_t)v.f_bavail*v.f_frsize;
#endif
    if (d->threads<1) d->threads=1;
    if (d->threads>8) d->threads=8;
    d->backends=loina_runtime_backends();
    return LOINA_OK;
}

bool loina_package_compatible(loina_engine *e, const loina_package *p) {
    if (!e || !loina_package_valid(p)) return false;
    const loina_runtime *r=loina_runtime_find(p->runtime);
    if (!r || r->abi!=p->runtime_abi) return false;
    const loina_device *d=&e->config.device;
    unsigned backends=r->backends() & d->backends & p->backends;
    if (!(backends & (LOINA_CPU|LOINA_VULKAN|LOINA_METAL|LOINA_CUDA|LOINA_OPENCL))) return false;
    return true;
}
const loina_package *loina_resolve(loina_engine *e, loina_feature feature, loina_error *error) {
    if (!e || feature<LOINA_TEXT || feature>LOINA_SPEECH) {
        loina_fail(error,LOINA_INVALID,"Invalid model selection request"); return NULL;
    }
    const loina_package *best=NULL;
    /* Use installed weights before offering a download. Allocation failures
     * belong to the runtime; RAM estimates must not substitute another model. */
    for (size_t i=0;i<loina_catalog_count();i++) {
        const loina_package *p=loina_catalog_get(i);
        if (p->feature!=feature || !loina_package_compatible(e,p)) continue;
        bool installed=loina_package_installed(e,p),have=best && loina_package_installed(e,best);
        if (!best || (installed && !have) || (installed==have && p->quality>best->quality)) best=p;
    }
    if (!best) loina_fail(error,LOINA_UNSUPPORTED,"No compatible package supports this device's runtimes");
    return best;
}
