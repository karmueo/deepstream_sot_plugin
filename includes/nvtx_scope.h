#ifndef SOT_NVTX_SCOPE_H
#define SOT_NVTX_SCOPE_H

#include <nvToolsExt.h>

class NvtxScope
{
public:
    explicit NvtxScope(const char *name) { nvtxRangePushA(name); }
    ~NvtxScope() { nvtxRangePop(); }

    NvtxScope(const NvtxScope &) = delete;
    NvtxScope &operator=(const NvtxScope &) = delete;
};

#endif // SOT_NVTX_SCOPE_H
