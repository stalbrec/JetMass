#include "UHH2/JetMass/include/MatchingSelections.h"
#include "UHH2/core/include/Event.h"

using namespace std;
// using namespace uhh2;
namespace uhh2{
  MatchingSelection::MatchingSelection(uhh2::Context & ctx, matchingOpt opt_):opt(opt_){
    string version = ctx.get("dataset_version", "");
    std::cout << "MatchingSelection: Using Selection with option: " << opt << std::endl;
  }

  bool MatchingSelection::passes_matching(const uhh2::Event &event, const TopJet &probe_jet) {
    int pdgId = (opt == oIsAnyGenW || opt == oIsLeadingGenW || opt == oIsMergedGenW || opt == oIsMergedGenWNotFromTop) ? 24 : 23;

    //finding all W in genparticles  
    assert(event.genparticles);
    std::vector<GenParticle> GenVs={};
    std::vector<GenParticle> GenTops={};
    std::vector<GenParticle> GenBsFromTops={};
    for(unsigned int i = 0 ; i < event.genparticles->size() ; i++){
        GenParticle thisGen = event.genparticles->at(i); 
        if(abs(thisGen.pdgId()) == pdgId ) GenVs.push_back(thisGen);
        if(abs(thisGen.pdgId()) == 6 ) {
          GenTops.push_back(thisGen);
          if(abs(event.genparticles->at(thisGen.daughter1()).pdgId()) == 5) GenBsFromTops.push_back(event.genparticles->at(thisGen.daughter1()));
          if(abs(event.genparticles->at(thisGen.daughter2()).pdgId()) == 5) GenBsFromTops.push_back(event.genparticles->at(thisGen.daughter2()));
        }
        //if matching with ANY V is wanted lets do it here
        if((opt == oIsAnyGenW || opt == oIsAnyGenZ) && deltaR(thisGen,probe_jet) < 0.8)return true;
    }
    //sort all found V by its pT
    sort_by_pt<GenParticle>(GenVs);
    if(GenVs.size()>0){
      //if there is any V match the leading
      if((opt == oIsLeadingGenW || opt == oIsLeadingGenZ) && deltaR(GenVs[0],probe_jet) < 0.8) return true;
      //or check if any V results from two merged quarks
      if(opt == oIsMergedGenW || opt == oIsMergedGenWNotFromTop || opt == oIsMergedGenZ){
        for(auto GenV: GenVs){
          GenParticle V_d1 = event.genparticles->at(GenV.daughter1());
          GenParticle V_d2 = event.genparticles->at(GenV.daughter2());
          int V_d1_pdgId = abs(V_d1.pdgId());
          int V_d2_pdgId = abs(V_d2.pdgId());
          if( (V_d1_pdgId >=1 && V_d1_pdgId <=6) && (V_d2_pdgId >=1 && V_d2_pdgId <=6) ){
           if(deltaR(probe_jet,V_d1) < 0.8 && deltaR(probe_jet,V_d2) < 0.8){
             if(opt == oIsMergedGenWNotFromTop){
               for(auto bFromTop: GenBsFromTops){
                 if(deltaR(probe_jet,bFromTop) < 1.5) return false;
               }
             }
             return true;
           }
          }
        }
      }
    }
      return false;
  }
}
