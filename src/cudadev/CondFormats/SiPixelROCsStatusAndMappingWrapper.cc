// C++ includes
#include <algorithm>
#include <iomanip>
#include <iostream>
#include <vector>

// CUDA includes
#include <cuda_runtime.h>

// CMSSW includes
#include "CUDACore/cudaCheck.h"
#include "CUDACore/device_unique_ptr.h"
#include "CUDACore/host_unique_ptr.h"
#include "CUDADataFormats/gpuClusteringConstants.h"
#include "CondFormats/SiPixelROCsStatusAndMappingWrapper.h"

SiPixelROCsStatusAndMappingWrapper::SiPixelROCsStatusAndMappingWrapper(SiPixelROCsStatusAndMapping const& cablingMap,
                                                               std::vector<unsigned char> modToUnp)
    : modToUnpDefault(modToUnp.size()), hasQuality_(true) {
  cablingMapHost = cms::cuda::make_host_unique<SiPixelROCsStatusAndMapping>(cudaStreamDefault);
  std::memcpy(cablingMapHost.get(), &cablingMap, sizeof(SiPixelROCsStatusAndMapping));

  std::copy(modToUnp.begin(), modToUnp.end(), modToUnpDefault.begin());
}

const SiPixelROCsStatusAndMapping* SiPixelROCsStatusAndMappingWrapper::getGPUProductAsync(cudaStream_t cudaStream) const {
  const auto& data = gpuData_.dataForCurrentDeviceAsync(cudaStream, [this](GPUData& data, cudaStream_t stream) {
    // allocate
    data.cablingMapDevice = cms::cuda::make_device_unique<SiPixelROCsStatusAndMapping>(stream);

    // transfer
    cudaCheck(cudaMemcpyAsync(
        data.cablingMapDevice.get(), this->cablingMapHost.get(), sizeof(SiPixelROCsStatusAndMapping), cudaMemcpyDefault, stream));
  });
  return data.cablingMapDevice.get();
}

const unsigned char* SiPixelROCsStatusAndMappingWrapper::getModToUnpAllAsync(cudaStream_t cudaStream) const {
  const auto& data =
      modToUnp_.dataForCurrentDeviceAsync(cudaStream, [this](ModulesToUnpack& data, cudaStream_t stream) {
        data.modToUnpDefault = cms::cuda::make_device_unique<unsigned char []>(pixelgpudetails::MAX_SIZE_BYTE_BOOL, stream);
        cudaCheck(cudaMemcpyAsync(data.modToUnpDefault.get(),
                                  this->modToUnpDefault.data(),
                                  this->modToUnpDefault.size() * sizeof(unsigned char),
                                  cudaMemcpyDefault,
                                  stream));
      });
  return data.modToUnpDefault.get();
}