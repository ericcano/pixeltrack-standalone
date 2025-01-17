#include "CUDACore/copyAsync.h"
#include "CUDACore/cudaCheck.h"
#include "CUDACore/device_unique_ptr.h"
#include "CUDACore/host_unique_ptr.h"
#include "CUDADataFormats/TrackingRecHit2DHeterogeneous.h"
#include "CondFormats/SiPixelROCsStatusAndMapping.h"

template <>
TrackingRecHit2DHostSOAView TrackingRecHit2DCUDA::hitsToHostAsync(cudaStream_t stream) const {
  // copy xl, yl, xerr, yerr, xg, yg, zg,rg, charge, clusterSizeX, clusterSizeY.
  TrackingRecHit2DHostSOAView ret(nHits(), stream);
  cms::cuda::copyAsync(ret.hits_h, m_hitsSupportLayerStartStore, ret.hitsStore_.soaMetadata().byteSize(), stream);
  return ret;
}