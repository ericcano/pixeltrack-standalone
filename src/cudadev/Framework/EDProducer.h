#ifndef EDProducerBase_h
#define EDProducerBase_h

#include "Framework/WaitingTaskWithArenaHolder.h"
#include <nvToolsExt.h>

namespace edm {
  class Event;
  class EventSetup;

  class EDProducer {
  public:
    EDProducer() = default;
    virtual ~EDProducer() = default;

    bool hasAcquire() const { return false; }

    void doAcquire(Event const& event, EventSetup const& eventSetup, WaitingTaskWithArenaHolder holder) {}

    void doProduce(Event& event, EventSetup const& eventSetup) {
      nvtxRangePush((std::string("Produce-") + typeid(*this).name()).c_str());
      produce(event, eventSetup); 
      // TODO: RAII me
      nvtxRangePop();
    }

    virtual void produce(Event& event, EventSetup const& eventSetup) = 0;

    void doEndJob() { endJob(); }

    virtual void endJob() {}

  private:
  };

  class EDProducerExternalWork {
  public:
    EDProducerExternalWork() = default;
    virtual ~EDProducerExternalWork() = default;

    bool hasAcquire() const { return true; }

    void doAcquire(Event const& event, EventSetup const& eventSetup, WaitingTaskWithArenaHolder holder) {
      nvtxRangePush((std::string("ExternalAcquire-") + typeid(*this).name()).c_str());
      acquire(event, eventSetup, std::move(holder));
      // TODO: RAII me
      nvtxRangePop();
    }

    void doProduce(Event& event, EventSetup const& eventSetup) { 
      nvtxRangePush((std::string("ExternalProduce-") + typeid(*this).name()).c_str());
      //nvtxRangePush(std::to_string(event.eventID()).c_str());
      produce(event, eventSetup); 
      // TODO: RAII me
      //nvtxRangePop();
      nvtxRangePop();
    }

    virtual void acquire(Event const& event, EventSetup const& eventSetup, WaitingTaskWithArenaHolder holder) = 0;
    virtual void produce(Event& event, EventSetup const& eventSetup) = 0;

    void doEndJob() { endJob(); }
    virtual void endJob() {}

  private:
  };
}  // namespace edm

#endif
