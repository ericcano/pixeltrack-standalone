#ifndef CUDADataFormats_TrackingRecHit_interface_TrackingRecHit2DSOAView_h
#define CUDADataFormats_TrackingRecHit_interface_TrackingRecHit2DSOAView_h

#include <cuda_runtime.h>

#include "CUDADataFormats/gpuClusteringConstants.h"
#include "CUDACore/HistoContainer.h"
#include "CUDACore/cudaCompat.h"
#include "Geometry/phase1PixelTopology.h"
#include "DataFormats/SoAStore.h"
#include "DataFormats/SoAView.h"

namespace pixelCPEforGPU {
  struct ParamsOnGPU;
}

class TrackingRecHit2DSOAStore {
public:
  using hindex_type = uint32_t;  // if above is <=2^32

  using PhiBinner = cms::cuda::HistoContainer<int16_t, 128, -1, 8 * sizeof(int16_t), hindex_type, 10>;

  using AverageGeometry = phase1PixelTopology::AverageGeometry;

  template <typename>
  friend class TrackingRecHit2DHeterogeneous;
  
  __device__ __forceinline__ uint32_t nHits() const { return m_nHits; }

  // Our arrays do not require specific alignment as access will not be coalesced in the current implementation
  // Sill, we need the 32 bits integers to be aligned, so we simply declare the SoA with the 32 bits fields first
  // and the 16 bits behind (as they have a looser alignment requirement. Then the SoA can be create with a byte 
  // alignment of 1)
  generate_SoA_store(HitsStore,
    // 32 bits section
    // local coord
    SoA_column(float, xLocal),
    SoA_column(float, yLocal),
    SoA_column(float, xerrLocal),
    SoA_column(float, yerrLocal),
    
    // global coord
    SoA_column(float, xGlobal),
    SoA_column(float, yGlobal),
    SoA_column(float, zGlobal),
    SoA_column(float, rGlobal),
    // global coordinates continue in the 16 bits section

    // cluster properties
    SoA_column(int32_t, charge),
          
    // 16 bits section (and cluster properties immediately continued)
    SoA_column(int16_t, clusterSizeX),
    SoA_column(int16_t, clusterSizeY)
  );
  
  generate_SoA_store(SupportObjectsStore,
    // This is the end of the data which is transferred to host. The following columns are supporting 
    // objects, not transmitted 
    
    // Supporting data (32 bits aligned)
    SoA_column(TrackingRecHit2DSOAStore::PhiBinner::index_type, phiBinnerStorage),
          
    // global coordinates (not transmitted)
    SoA_column(int16_t, iphi),
          
    // cluster properties (not transmitted)
    SoA_column(uint16_t, detectorIndex)
  );
  
  generate_SoA_view(HitsAndSupportView,
    SoA_view_store_list(
      SoA_view_store(HitsStore, hitsStore),
      SoA_view_store(SupportObjectsStore, supportObjectsStore)
    ),
    SoA_view_value_list(
      SoA_view_value(hitsStore, xLocal, xLocal),
      SoA_view_value(hitsStore, yLocal, yLocal),
      SoA_view_value(hitsStore, xerrLocal, xerrLocal),
      SoA_view_value(hitsStore, yerrLocal, yerrLocal),
      
      SoA_view_value(hitsStore, xGlobal, xGlobal),
      SoA_view_value(hitsStore, yGlobal, yGlobal),
      SoA_view_value(hitsStore, zGlobal, zGlobal),
      SoA_view_value(hitsStore, rGlobal, rGlobal),
      
      SoA_view_value(hitsStore, charge, charge),
      SoA_view_value(hitsStore, clusterSizeX, clusterSizeX),
      SoA_view_value(hitsStore, clusterSizeY, clusterSizeY),
      
      SoA_view_value(supportObjectsStore, phiBinnerStorage, phiBinnerStorage),
      SoA_view_value(supportObjectsStore, iphi, iphi),
      SoA_view_value(supportObjectsStore, detectorIndex, detectorIndex)
    )
  );
  
  // Shortcut operator saving the explicit calls to view in usage.
  __device__ __forceinline__ HitsAndSupportView::element operator[] (size_t index) { 
    return m_hitsAndSupportView[index]; 
  }
  __device__ __forceinline__ const HitsAndSupportView::const_element operator[] (size_t index) const { 
    return m_hitsAndSupportView[index];
  }
  
  __device__ __forceinline__ pixelCPEforGPU::ParamsOnGPU const& cpeParams() const { return *m_cpeParams; }

  __device__ __forceinline__ uint32_t hitsModuleStart(int i) const { return __ldg(m_hitsModuleStart + i); }

  __device__ __forceinline__ uint32_t* hitsLayerStart() { return m_hitsLayerStart; }
  __device__ __forceinline__ uint32_t const* hitsLayerStart() const { return m_hitsLayerStart; }

  __device__ __forceinline__ PhiBinner& phiBinner() { return *m_phiBinner; }
  __device__ __forceinline__ PhiBinner const& phiBinner() const { return *m_phiBinner; }

  __device__ __forceinline__ AverageGeometry& averageGeometry() { return *m_averageGeometry; }
  __device__ __forceinline__ AverageGeometry const& averageGeometry() const { return *m_averageGeometry; }

private:
  // hits store
  HitsStore m_hitsStore;
  // supporting objects store
  SupportObjectsStore m_supportObjectsStore;
  // Global view simplifying usage
  HitsAndSupportView m_hitsAndSupportView;
  
  // individually defined supporting objects
  // m_averageGeometry is corrected for beam spot, not sure where to host it otherwise
  AverageGeometry* m_averageGeometry;              // owned by TrackingRecHit2DHeterogeneous
  pixelCPEforGPU::ParamsOnGPU const* m_cpeParams;  // forwarded from setup, NOT owned
  uint32_t const* m_hitsModuleStart;               // forwarded from clusters

  uint32_t* m_hitsLayerStart;

  PhiBinner* m_phiBinner;
  PhiBinner::index_type* m_phiBinnerStorage;

  uint32_t m_nHits;
};

#endif  // CUDADataFormats_TrackingRecHit_interface_TrackingRecHit2DSOAView_h