#ifndef ALPAKAMEMORYHELPER_H
#define ALPAKAMEMORYHELPER_H

#include "AlpakaCore/alpakaConfig.h"
#include "AlpakaCore/alpakaDevices.h"

using namespace alpaka_common;

namespace cms {
  namespace alpakatools {

    template <typename TData>
    auto allocHostBuf(const Extent& extent) {
      return alpaka::allocBuf<TData, Idx>(host, extent);
    }

    template <typename TData>
    auto createHostView(TData* data, const Extent& extent) {
      return alpaka::ViewPlainPtr<DevHost, TData, Dim1D, Idx>(data, host, extent);
    }

    template <typename TData>
    auto allocDeviceBuf(const Extent& extent) {
      return alpaka::allocBuf<TData, Idx>(ALPAKA_ACCELERATOR_NAMESPACE::device, extent);
    }

    template <typename TData>
    auto createDeviceView(const TData* data, const Extent& extent) {
      return alpaka::ViewPlainPtr<ALPAKA_ACCELERATOR_NAMESPACE::Device, const TData, Dim1D, Idx>(
          data, ALPAKA_ACCELERATOR_NAMESPACE::device, extent);
    }

    template <typename TData>
    auto createDeviceView(TData* data, const Extent& extent) {
      return alpaka::ViewPlainPtr<ALPAKA_ACCELERATOR_NAMESPACE::Device, TData, Dim1D, Idx>(
          data, ALPAKA_ACCELERATOR_NAMESPACE::device, extent);
    }

  }  // namespace alpakatools
}  // namespace cms

#endif  // ALPAKAMEMORYHELPER_H
