#ifndef Event_h
#define Event_h

#include <memory>
#include <utility>
#include <vector>
#include <nvToolsExt.h>

#include "Framework/ProductRegistry.h"

// type erasure
namespace edm {
  using StreamID = int;

  class WrapperBase {
  public:
    virtual ~WrapperBase() = default;
  };

  template <typename T>
  class Wrapper : public WrapperBase {
  public:
    template <typename... Args>
    explicit Wrapper(Args&&... args) : obj_{std::forward<Args>(args)...} {}

    T const& product() const { return obj_; }

  private:
    T obj_;
  };

  class Event {
  public:
    explicit Event(int streamId, int eventId, ProductRegistry const& reg)
        : streamId_(streamId), eventId_(eventId), products_(reg.size()), nvtxRange_(nvtxRangeStartA("Event")) {}
        
    ~Event() { nvtxRangeEnd(nvtxRange_); }

    StreamID streamID() const { return streamId_; }
    int eventID() const { return eventId_; }

    template <typename T>
    T const& get(EDGetTokenT<T> const& token) const {
      return static_cast<Wrapper<T> const&>(*products_[token.index()]).product();
    }

    template <typename T, typename... Args>
    void emplace(EDPutTokenT<T> const& token, Args&&... args) {
      products_[token.index()] = std::make_unique<Wrapper<T>>(std::forward<Args>(args)...);
    }

  private:
    StreamID streamId_;
    int eventId_;
    std::vector<std::unique_ptr<WrapperBase>> products_;
    nvtxRangeId_t nvtxRange_;
  };
}  // namespace edm

#endif
