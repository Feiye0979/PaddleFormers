#define _GNU_SOURCE
#include <nccl.h>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>

/*
 * NCCL Compatibility Shim
 *
 * Compile with: gcc -shared -fPIC -o libnccl.so.2 nccl_shim.c
 *               -I<cuda_include>
 *               ./libnccl_real.so          <- NCCL 2.21.5 backend
 *               -Wl,-rpath,$ORIGIN
 *               -Wl,-soname,libnccl.so.2
 *
 * This shim ONLY defines symbols missing in NCCL 2.21.5.
 * All standard NCCL functions (ncclGetUniqueId, ncclAllReduce, ...) are
 * provided transparently by libnccl_real.so through direct linkage.
 *
 * Missing-symbol stubs:
 *   Paddle:  ncclCommInitRank2, ncclCommInitRankConfigMemOpt,
 *            ncclCommGenMemOptConfig, ncclCommFreeMemOptConfig
 *   PyTorch: ncclGroupSimulateEnd, ncclCommInitRankScalable,
 *            ncclCommWindowRegister, ncclCommWindowDeregister
 *            (+ p-prefixed aliases for the same)
 */

/* -----------------------------------------------------------------------
 * ncclCommInitRankConfig: present in NCCL 2.21.5+ but define a safe
 * wrapper in case the build doesn't have it.
 * Since we link against libnccl_real.so it will be resolved there.
 * We still need to declare it for the stubs below that call it.
 * ----------------------------------------------------------------------- */

/* Paddle-required stubs (NCCL 2.22+ APIs) */

typedef struct ncclMemOptConfig ncclMemOptConfig_t;
struct ncclMemOptConfig { int _unused; };

ncclResult_t ncclCommInitRank2(
        ncclComm_t* comm, int nranks, ncclUniqueId commId,
        int rank, int param) {
    ncclConfig_t cfg = NCCL_CONFIG_INITIALIZER;
    return ncclCommInitRankConfig(comm, nranks, commId, rank, &cfg);
}

ncclResult_t ncclCommInitRankConfigMemOpt(
        ncclComm_t* comm, int nranks, ncclUniqueId commId, int rank,
        ncclConfig_t* config, ncclMemOptConfig_t* memopt_config) {
    return ncclCommInitRankConfig(comm, nranks, commId, rank, config);
}

ncclMemOptConfig_t* ncclCommGenMemOptConfig(
        const char* name, int a, int b, int c, int d, int e,
        const char* f, const char* g) {
    return NULL;
}

ncclResult_t ncclCommFreeMemOptConfig(ncclMemOptConfig_t* config) {
    return ncclSuccess;
}

/* PyTorch-required stubs (NCCL 2.27+ APIs) */

ncclResult_t ncclGroupSimulateEnd(void) { return ncclSuccess; }
ncclResult_t pncclGroupSimulateEnd(void) { return ncclSuccess; }

ncclResult_t ncclCommInitRankScalable(
        ncclComm_t* comm, int nranks, ncclUniqueId commId, int rank) {
    return ncclCommInitRank(comm, nranks, commId, rank);
}
ncclResult_t pncclCommInitRankScalable(
        ncclComm_t* comm, int nranks, ncclUniqueId commId, int rank) {
    return ncclCommInitRank(comm, nranks, commId, rank);
}

ncclResult_t ncclCommWindowRegister(
        ncclComm_t comm, void* ptr, size_t size, int* windowId) {
    if (windowId) *windowId = 0;
    return ncclSuccess;
}
ncclResult_t pncclCommWindowRegister(
        ncclComm_t comm, void* ptr, size_t size, int* windowId) {
    if (windowId) *windowId = 0;
    return ncclSuccess;
}
ncclResult_t ncclCommWindowDeregister(ncclComm_t comm, int windowId) {
    return ncclSuccess;
}
ncclResult_t pncclCommWindowDeregister(ncclComm_t comm, int windowId) {
    return ncclSuccess;
}
