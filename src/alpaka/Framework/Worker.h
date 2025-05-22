#ifndef Framework_Worker_h
#define Framework_Worker_h

#include <atomic>
#include <exception>
//#include <iostream>
#include <utility>
#include <vector>

#include "nvtx3/nvtx3.hpp"

#include "Framework/WaitingTask.h"
#include "Framework/WaitingTaskHolder.h"
#include "Framework/WaitingTaskList.h"
#include "Framework/WaitingTaskWithArenaHolder.h"
#include "Framework/demangle.h"
#include "Framework/Event.h"

namespace {
  struct D_dwa { static constexpr const char name[] = "Worker::doWorkAsync"; };
  struct D_dw { static constexpr const char name[] = "Worker::doneWaiting"; };
  struct D_dp { static constexpr const char name[] = "Worker::doProduce"; };
  struct D_da { static constexpr const char name[] = "Worker::doAcquire"; };
}

namespace edm {
  class Event;
  class EventSetup;
  class ProductRegistry;

  class Worker {
  public:
    Worker() : prefetchRequested_{false} {}
    virtual ~Worker() = default;
    virtual const std::string & type() { static std::string name="edm::Worker"; return name; }

    // not thread safe
    void setItemsToGet(std::vector<Worker*> workers) { itemsToGet_ = std::move(workers); }

    // thread safe
    void prefetchAsync(Event& event, EventSetup const& eventSetup, WaitingTaskHolder iTask);

    // not thread safe
    virtual void doWorkAsync(Event& event, EventSetup const& eventSetup, WaitingTaskHolder iTask) = 0;

    // not thread safe
    virtual void doEndJob() = 0;

    // not thread safe
    void reset() {
      prefetchRequested_ = false;
      doReset();
    }

  protected:
    virtual void doReset() = 0;

  private:
    std::vector<Worker*> itemsToGet_;
    std::atomic<bool> prefetchRequested_;
  };

  template <typename T>
  class WorkerT : public Worker {
  public:
    explicit WorkerT(ProductRegistry& reg) : producer_(reg), workStarted_{false} {}

    const std::string & type() override {
      static const std::string name = edm::demangle<T>;
      return name;
    }

    void doWorkAsync(Event& event, EventSetup const& eventSetup, WaitingTaskHolder task) override {

      uint8_t r = (event.eventID() * 10) % 0xFF;
      uint8_t g = (event.eventID() * 15 + 0x50) % 0xFF;
      uint8_t b = (event.eventID() * 20 + 0xA0) % 0xFF;
      nvtx3::scoped_range_in<D_dwa> sri{this->type(), nvtx3::rgb{r,g,b}, nvtx3::payload{event.eventID()}};
      waitingTasksWork_.add(task);
      //std::cout << "doWorkAsync for " << this << " with iTask " << iTask << std::endl;
      bool expected = false;
      if (workStarted_.compare_exchange_strong(expected, true)) {
        //std::cout << "first doWorkAsync call" << std::endl;

        WaitingTask* moduleTask =
            make_waiting_task([this, &event, &eventSetup](std::exception_ptr const* iPtr) mutable {
              if (iPtr) {
                waitingTasksWork_.doneWaiting(*iPtr);
              } else {
                std::exception_ptr exceptionPtr;
                try {
                  //std::cout << "calling doProduce " << this << std::endl;
                  uint8_t r = (event.eventID() * 10) % 0xFF;
                  uint8_t g = (event.eventID() * 15 + 0x50) % 0xFF;
                  uint8_t b = (event.eventID() * 20 + 0xA0) % 0xFF;
                  nvtx3::scoped_range_in<D_dp> sri{this->type(), nvtx3::rgb{r,g,b}, nvtx3::payload{event.eventID()}};
                  producer_.doProduce(event, eventSetup);
                } catch (...) {
                  exceptionPtr = std::current_exception();
                }
                //std::cout << "waitingTasksWork_.doneWaiting " << this << std::endl;
                uint8_t r = (event.eventID() * 10) % 0xFF;
                uint8_t g = (event.eventID() * 15 + 0x50) % 0xFF;
                uint8_t b = (event.eventID() * 20 + 0xA0) % 0xFF;
                nvtx3::scoped_range_in<D_dw> sri{this->type() + "_after_exception", nvtx3::rgb{r,g,b}, nvtx3::payload{event.eventID()}};
                waitingTasksWork_.doneWaiting(exceptionPtr);
              }
            });
        auto* group = task.group();
        if (producer_.hasAcquire()) {
          WaitingTaskWithArenaHolder runProduceHolder{*group, moduleTask};
          moduleTask = make_waiting_task([this, &event, &eventSetup, runProduceHolder = std::move(runProduceHolder)](
                                             std::exception_ptr const* iPtr) mutable {
            if (iPtr) {
              uint8_t r = (event.eventID() * 10) % 0xFF;
              uint8_t g = (event.eventID() * 15 + 0x50) % 0xFF;
              uint8_t b = (event.eventID() * 20 + 0xA0) % 0xFF;
              nvtx3::scoped_range_in<D_dw> sri{this->type() + "_after_exception", nvtx3::rgb{r,g,b}, nvtx3::payload{event.eventID()}};
              runProduceHolder.doneWaiting(*iPtr);
            } else {
              std::exception_ptr exceptionPtr;
              try {
                uint8_t r = (event.eventID() * 10) % 0xFF;
                uint8_t g = (event.eventID() * 15 + 0x50) % 0xFF;
                uint8_t b = (event.eventID() * 20 + 0xA0) % 0xFF;
                nvtx3::scoped_range_in<D_dw> sri{this->type(), nvtx3::rgb{r,g,b}, nvtx3::payload{event.eventID()}};
                producer_.doAcquire(event, eventSetup, runProduceHolder);
              } catch (...) {
                exceptionPtr = std::current_exception();
              }
              uint8_t r = (event.eventID() * 10) % 0xFF;
              uint8_t g = (event.eventID() * 15 + 0x50) % 0xFF;
              uint8_t b = (event.eventID() * 20 + 0xA0) % 0xFF;
              nvtx3::scoped_range_in<D_dw> sri{this->type(), nvtx3::rgb{r,g,b}, nvtx3::payload{event.eventID()}};
              runProduceHolder.doneWaiting(exceptionPtr);
            }
          });
        }
        //std::cout << "calling prefetchAsync " << this << " with moduleTask " << moduleTask << std::endl;
        prefetchAsync(event, eventSetup, WaitingTaskHolder(*group, moduleTask));
      }
    }

    void doEndJob() override { producer_.doEndJob(); }

  private:
    void doReset() override {
      waitingTasksWork_.reset();
      workStarted_ = false;
    }

    T producer_;
    WaitingTaskList waitingTasksWork_;
    std::atomic<bool> workStarted_;
  };
}  // namespace edm
#endif  // Framework_Worker_h
