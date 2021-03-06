#include "UHH2/JetMass/include/MatchingSelections.h"
#include "UHH2/core/include/Event.h"

using namespace std;
// using namespace uhh2;
namespace uhh2{
  MatchingSelection::MatchingSelection(uhh2::Context & ctx){
    TString version = ctx.get("dataset_version", "");
    version.ToLower();
    is_VJets = (version.Contains("wjets") || version.Contains("zjets") );
    is_TTbar = (version.Contains("ttbar") || version.Contains("ttjets") || version.Contains("ttto")) && version.Contains("semilep");
    is_valid = true;
  }

  void MatchingSelection::init(const uhh2::Event &event){
    if(event.isRealData){
      is_valid = false;
      return;
    }

    if(!is_VJets && !is_TTbar){
      cout << "This is not a TTbar or V+Jets sample. Matching is not implemented. Skipping MatchingSelection" << endl;
      is_valid = false;
      return;
    }

    assert(event.genparticles);
    if(is_VJets){
      unsigned int n_V = 0;
      for(unsigned int i=0; i<event.genparticles->size(); ++i) {
        const GenParticle & genp = event.genparticles->at(i);
        if(abs(genp.pdgId()) == 23 || abs(genp.pdgId()) == 24){
          ++n_V;
          genV = genp;
          genQ1 = *genp.daughter(event.genparticles,1);
          genQ2 = *genp.daughter(event.genparticles,2);
        }
      }
      if(n_V > 1){
        cout << "MatchinSelection (V+jets): There are more than 1 W/Z Boson among generator-particles! (event:"<<event.event << ")" << endl;
        is_valid = false;
      }
    }else{
      TTbarGen ttgen(*event.genparticles);
      if(ttgen.IsSemiLeptonicDecay()){
        genTop = ttgen.TopHad();
        genB = ttgen.BHad();
        genV = ttgen.WHad();
        genQ1 = ttgen.Q1();
        genQ2 = ttgen.Q2();
      }else{
        is_valid = false;
      }
    }
  }

  bool MatchingSelection::passes_matching(const TopJet &probe_jet, matchingOpt opt, float radius) {
    if(!is_VJets && !is_TTbar) return false;

    if(!is_valid){
      if(opt == oIsNotMerged) return true;
      else return false;
    }

    bool b_in_Jet = is_TTbar ? deltaR(probe_jet, genB) < radius : false;
    bool q1_in_Jet = deltaR(probe_jet, genQ1) < radius;
    bool q2_in_Jet = deltaR(probe_jet, genQ2) < radius;

    if(b_in_Jet && q1_in_Jet && q2_in_Jet ){
      if( (opt == oIsMergedTop)) return true;
    }else if( ((b_in_Jet && !q1_in_Jet && q2_in_Jet) || (b_in_Jet && q1_in_Jet && !q2_in_Jet)) ){
      if( ((opt == oIsMergedQB) || (opt == oIsSemiMergedTop)) ) return true;
    }else if( (!b_in_Jet && q1_in_Jet && q2_in_Jet) ){
      if( ((opt == oIsMergedV) || (opt == oIsMergedW) || (opt == oIsMergedZ) || (opt == oIsSemiMergedTop)) ) return true;
    }else if(opt == oIsNotMerged) return true;
    return false;
  }
}
