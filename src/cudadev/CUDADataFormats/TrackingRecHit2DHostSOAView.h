
#ifndef CUDADataFormats_TrackingRecHit_interface_TrackingRecHit2DSOAHostView_h
#define CUDADataFormats_TrackingRecHit_interface_TrackingRecHit2DSOAHostView_h

#include "CUDADataFormats/TrackingRecHit2DSOAView.h"
#include "CUDACore/host_unique_ptr.h"

template <typename Traits>
class TrackingRecHit2DHeterogeneous; 

struct TrackingRecHit2DHostSOAView {
  template <typename Traits>
  friend class TrackingRecHit2DHeterogeneous;
public:
  TrackingRecHit2DHostSOAView();
  void reset();
  __device__ __forceinline__ const auto operator[](size_t i) const  { return hitsStore_[i]; }
  __device__ __forceinline__ size_t size() { return hitsStore_.soaMetadata().size(); }
private:
  TrackingRecHit2DHostSOAView(size_t size, cudaStream_t stream);
  cms::cuda::host::unique_ptr<std::byte[]> hits_h;
  TrackingRecHit2DSOAStore::HitsStore hitsStore_;
};


#endif // ndef CUDADataFormats_TrackingRecHit_interface_TrackingRecHit2DSOAHostView_h