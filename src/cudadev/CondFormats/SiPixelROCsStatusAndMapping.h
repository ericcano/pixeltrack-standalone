#ifndef CondFormats_SiPixelObjects_interface_SiPixelROCsStatusAndMapping_h
#define CondFormats_SiPixelObjects_interface_SiPixelROCsStatusAndMapping_h

#include "DataFormats/SoAMacros.h"

namespace pixelgpudetails {
  // Maximum fed for phase1 is 150 but not all of them are filled
  // Update the number FED based on maximum fed found in the cabling map
  constexpr unsigned int MAX_FED = 150;
  constexpr unsigned int MAX_LINK = 48;  // maximum links/channels for Phase 1
  constexpr unsigned int MAX_ROC = 8;
  constexpr unsigned int MAX_SIZE = MAX_FED * MAX_LINK * MAX_ROC;
  constexpr unsigned int MAX_SIZE_BYTE_BOOL = MAX_SIZE * sizeof(unsigned char);
}  // namespace pixelgpudetails

declare_SoA_template(SiPixelROCsStatusAndMapping,
  SoA_FundamentalTypeColumn(unsigned int, fed),
  SoA_FundamentalTypeColumn(unsigned int, link),
  SoA_FundamentalTypeColumn(unsigned int, roc),
  SoA_FundamentalTypeColumn(unsigned int, rawId),
  SoA_FundamentalTypeColumn(unsigned int, rocInDet),
  SoA_FundamentalTypeColumn(unsigned int, moduleId),
  SoA_FundamentalTypeColumn(unsigned char, badRocs),
  SoA_scalar(unsigned int, size)
);

#endif  // CondFormats_SiPixelObjects_interface_SiPixelROCsStatusAndMapping_h
