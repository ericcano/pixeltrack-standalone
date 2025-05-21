//#include <iostream>

#include "Framework/Worker.h"
#include "Framework/demangle.h"
#include "nvtx3/nvtx3.hpp"

namespace edm {
  void Worker::prefetchAsync(Event& event, EventSetup const& eventSetup, WaitingTaskHolder iTask) {
    //std::cout << "prefetchAsync for " << this << " iTask " << iTask << std::endl;
    bool expected = false;
    if (prefetchRequested_.compare_exchange_strong(expected, true)) {
      //std::cout << "first prefetch call" << std::endl;
      for (Worker* dep : itemsToGet_) {
        //std::cout << "calling doWorkAsync for " << dep << " with " << iTask << std::endl;
        nvtx3::scoped_range r{dep->type()};
        dep->doWorkAsync(event, eventSetup, iTask);
      }
    }
  }
}  // namespace edm