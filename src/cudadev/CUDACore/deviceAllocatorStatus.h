#ifndef HeterogeneousCore_CUDAUtilities_deviceAllocatorStatus_h
#define HeterogeneousCore_CUDAUtilities_deviceAllocatorStatus_h

#include <map>

namespace cms {
  namespace cuda {
    namespace allocator {
      struct TotalBytes {
        size_t free = 0;
        size_t live = 0;
        size_t liveRequested = 0;  // CMS: monitor also requested amount
      };
      /// Map type of device ordinals to the number of cached bytes cached by each device
      using GpuCachedBytes = std::map<int, TotalBytes>;
    }  // namespace allocator
  }  // namespace cuda
}  // namespace cms

#endif
