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
  // TODO: check if cudaStreamDefault is appropriate
  cablingMapHostBuffer = cms::cuda::make_host_unique<std::byte[]>(
          SiPixelROCsStatusAndMapping::computeDataSize(pixelgpudetails::MAX_SIZE), cudaStreamDefault);
  std::memcpy(cablingMapHostBuffer.get(), cablingMap.soaMetadata().baseAddress(), 
          SiPixelROCsStatusAndMapping::computeDataSize(pixelgpudetails::MAX_SIZE));

  std::copy(modToUnp.begin(), modToUnp.end(), modToUnpDefault.begin());
}

const SiPixelROCsStatusAndMapping SiPixelROCsStatusAndMappingWrapper::getGPUProductAsync(cudaStream_t cudaStream) const {
  const auto& data = gpuData_.dataForCurrentDeviceAsync(cudaStream, 
          [this](GPUData& data, cudaStream_t stream) {
            // allocate
            data.allocate(pixelgpudetails::MAX_SIZE, stream);
            // transfer
            cudaCheck(cudaMemcpyAsync(
                data.cablingMapBuffer.get(), this->cablingMapHostBuffer.get(),
                    SiPixelROCsStatusAndMapping::computeDataSize(pixelgpudetails::MAX_SIZE), cudaMemcpyDefault, stream));
          }
  );
  return data.cablingMapDevice;
}

const unsigned char* SiPixelROCsStatusAndMappingWrapper::getModToUnpAllAsync(cudaStream_t cudaStream) const {
  const auto& data =
      modToUnp_.dataForCurrentDeviceAsync(cudaStream, [this](ModulesToUnpack& data, cudaStream_t stream) {
        cudaCheck(cudaMalloc((void**)&data.modToUnpDefault, pixelgpudetails::MAX_SIZE_BYTE_BOOL));
        cudaCheck(cudaMemcpyAsync(data.modToUnpDefault,
                                  this->modToUnpDefault.data(),
                                  this->modToUnpDefault.size() * sizeof(unsigned char),
                                  cudaMemcpyDefault,
                                  stream));
      });
  return data.modToUnpDefault;
}

SiPixelROCsStatusAndMappingWrapper::ModulesToUnpack::~ModulesToUnpack() { cudaCheck(cudaFree(modToUnpDefault)); }
