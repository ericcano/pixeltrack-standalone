#include "KokkosCore/kokkosConfig.h"
#include "KokkosCore/Product.h"
#include "KokkosCore/ScopedContext.h"
#include "DataFormats/ZVertexSoA.h"
#include "Framework/EventSetup.h"
#include "Framework/Event.h"
#include "Framework/PluginFactory.h"
#include "Framework/EDProducer.h"
#include "Framework/RunningAverage.h"

namespace KOKKOS_NAMESPACE {
  class PixelVertexSoAFromKokkos : public edm::EDProducerExternalWork {
  public:
    explicit PixelVertexSoAFromKokkos(edm::ProductRegistry& reg);
    ~PixelVertexSoAFromKokkos() override = default;

  private:
    void acquire(edm::Event const& iEvent,
                 edm::EventSetup const& iSetup,
                 edm::WaitingTaskWithArenaHolder waitingTaskHolder) override;
    void produce(edm::Event& iEvent, edm::EventSetup const& iSetup) override;

    using VerticesDeviceMemSpace = Kokkos::View<ZVertexSoA, KokkosDeviceMemSpace>;
    using VerticesHostMemSpace = Kokkos::View<ZVertexSoA, KokkosHostMemSpace>;

    edm::EDGetTokenT<cms::kokkos::Product<VerticesDeviceMemSpace>> tokenKokkos_;
    edm::EDPutTokenT<VerticesHostMemSpace> tokenSOA_;

    VerticesHostMemSpace m_soa;
  };

  PixelVertexSoAFromKokkos::PixelVertexSoAFromKokkos(edm::ProductRegistry& reg)
      : tokenKokkos_(reg.consumes<cms::kokkos::Product<VerticesDeviceMemSpace>>()),
        tokenSOA_(reg.produces<VerticesHostMemSpace>()) {}

  void PixelVertexSoAFromKokkos::acquire(edm::Event const& iEvent,
                                         edm::EventSetup const& iSetup,
                                         edm::WaitingTaskWithArenaHolder waitingTaskHolder) {
    auto const& inputDataWrapped = iEvent.get(tokenKokkos_);
    cms::kokkos::ScopedContextAcquire<KokkosExecSpace> ctx{inputDataWrapped, std::move(waitingTaskHolder)};
    auto const& inputData = ctx.get(inputDataWrapped);

    m_soa = VerticesHostMemSpace("vertices");
    Kokkos::deep_copy(ctx.execSpace(), m_soa, inputData);
  }

  void PixelVertexSoAFromKokkos::produce(edm::Event& iEvent, edm::EventSetup const& iSetup) {
    // No copies....
    iEvent.emplace(tokenSOA_, std::move(m_soa));
  }
}  // namespace KOKKOS_NAMESPACE

DEFINE_FWK_KOKKOS_MODULE(PixelVertexSoAFromKokkos);
