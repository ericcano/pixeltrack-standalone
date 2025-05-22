//#include <iostream>

#include "Framework/Worker.h"
#include "Framework/demangle.h"
#include "Framework/Event.h"
#include "nvtx3/nvtx3.hpp"

namespace {
  struct D_pfa { static constexpr const char name[] = "Worker::prefetchAsync"; };
}

namespace edm {
  void Worker::prefetchAsync(Event& event, EventSetup const& eventSetup, WaitingTaskHolder iTask) {
    //std::cout << "prefetchAsync for " << this << " iTask " << iTask << std::endl;
    bool expected = false;
    if (prefetchRequested_.compare_exchange_strong(expected, true)) {
      //std::cout << "first prefetch call" << std::endl;
      for (Worker* dep : itemsToGet_) {
        //std::cout << "calling doWorkAsync for " << dep << " with " << iTask << std::endl;
        uint8_t r = (event.eventID() * 10) % 0xFF;
        uint8_t g = (event.eventID() * 15 + 0x50) % 0xFF;
        uint8_t b = (event.eventID() * 20 + 0xA0) % 0xFF;
        nvtx3::scoped_range_in<D_pfa> sri{dep->type(), nvtx3::rgb{r,g,b}, nvtx3::payload{event.eventID()}};
        dep->doWorkAsync(event, eventSetup, iTask);
      }
    }
  }
}  // namespace edm