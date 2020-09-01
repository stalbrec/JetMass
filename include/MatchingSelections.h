#pragma once

#include "UHH2/core/include/fwd.h"
#include "UHH2/core/include/Selection.h"
#include "UHH2/core/include/TopJet.h"
#include "UHH2/common/include/Utils.h"

#include <vector>
namespace uhh2{

class MatchingSelection: public uhh2::Selection {
public:
  enum matchingOpt{oIsAnyGenW,oIsLeadingGenW,oIsMergedGenW,oIsMergedGenWNotFromTop,oIsAnyGenZ,oIsLeadingGenZ,oIsMergedGenZ};
  
  explicit MatchingSelection(uhh2::Context & ctx, matchingOpt opt);
  
  //this will not be used since we want to pass a (top)jet to be probed.
  //In order to ommit the warning that 'event' is not used we print this useless warning.
  virtual bool passes(const uhh2::Event & event) override{
    std::cout << "MatchingSelection: Using empty passes function on event "<< event.event << ". Are you sure you want to do that?" << std::endl;
    return true;
  };
  bool passes_matching(const uhh2::Event & event, const TopJet &probe_jet);
  
private:
  matchingOpt opt;
};

}
