#include <iostream>
#include <memory>

#include "UHH2/core/include/AnalysisModule.h"
#include "UHH2/core/include/Event.h"
#include "UHH2/common/include/MCWeight.h"
#include "UHH2/common/include/CommonModules.h"
#include "UHH2/common/include/CleaningModules.h"
#include "UHH2/common/include/ElectronHists.h"
#include "UHH2/common/include/NSelections.h"
#include "UHH2/JetMass/include/JetMassSelections.h"
#include "UHH2/JetMass/include/JetMassHists.h"

using namespace std;
using namespace uhh2;

class JetMassModule: public AnalysisModule {
public:

  explicit JetMassModule(Context & ctx);
  virtual bool process(Event & event) override;

private:

  //std::unique_ptr<Selection> ;
  // uhh2::Event::Handle<double> h_rec_weight;
  // uhh2::Event::Handle<double> h_gen_weight;
  uhh2::Event::Handle<bool> h_passed_rec;
  std::unique_ptr<AnalysisModule> PUreweight, lumiweight, sf_btag, muo_tight_noniso_SF, muo_trigger_SF;


  std::unique_ptr<Hists> h_mjet_2jets_pt400, h_mjet_2jets_pt300, h_mjet_2jets_pt200;
  std::unique_ptr<Hists> h_mjet_pt400, h_mjet_pt300, h_mjet_pt200;

  bool isMC;

};


JetMassModule::JetMassModule(Context & ctx){

  isMC = (ctx.get("dataset_type") == "MC");

  h_passed_rec = ctx.get_handle<bool>("h_passed_rec");
  // h_rec_weight = ctx.get_handle<double>("h_rec_weight_kin");
  // h_gen_weight = ctx.get_handle<double>("h_gen_weight_kin");

  lumiweight.reset(new MCLumiWeight(ctx));
  PUreweight.reset(new MCPileupReweight(ctx, "central"));
  sf_btag.reset(new MCBTagScaleFactor(ctx, BTag::DEEPCSV, BTag::WP_MEDIUM, "jets", "central"));
  muo_tight_noniso_SF.reset(new MCMuonScaleFactor(ctx,"/nfs/dust/cms/user/schwarzd/CMSSW10/CMSSW_10_2_10/src/UHH2/common/data/2016/MuonID_EfficienciesAndSF_average_RunBtoH.root","MC_NUM_TightID_DEN_genTracks_PAR_pt_eta",1, "tightID", false, "central"));
  muo_trigger_SF.reset(new MCMuonScaleFactor(ctx,"/nfs/dust/cms/user/schwarzd/CMSSW10/CMSSW_10_2_10/src/UHH2/common/data/2016/MuonTrigger_EfficienciesAndSF_average_RunBtoH.root","IsoMu50_OR_IsoTkMu50_PtEtaBins",1, "muonTrigger", false, "central"));

  vector<double> ptbins = {0, 5, 10000000};
  vector<double> etabins = {0, 1, 10};
  double variation = 0.1;
  h_mjet_pt200.reset(new JetMassHists(ctx, "JetMass_pt200", ptbins, etabins, variation));
  h_mjet_pt400.reset(new JetMassHists(ctx, "JetMass_pt300", ptbins, etabins, variation));
  h_mjet_pt300.reset(new JetMassHists(ctx, "JetMass_pt400", ptbins, etabins, variation));
  h_mjet_2jets_pt200.reset(new JetMassHists(ctx, "JetMass_2jets_pt200", ptbins, etabins, variation));
  h_mjet_2jets_pt400.reset(new JetMassHists(ctx, "JetMass_2jets_pt300", ptbins, etabins, variation));
  h_mjet_2jets_pt300.reset(new JetMassHists(ctx, "JetMass_2jets_pt400", ptbins, etabins, variation));

}


bool JetMassModule::process(Event & event) {

  if(isMC){
    lumiweight->process(event);
    PUreweight->process(event);
    sf_btag->process(event);
    muo_tight_noniso_SF->process(event);
    muo_trigger_SF->process(event);
  }

  bool passed_presel = event.get(h_passed_rec);

  // SELECTION
  vector<TopJet>* topjets = event.topjets;
  if(!passed_presel) return false;
  if(topjets->size() < 1) return false;
  if(topjets->at(0).pt() < 200) return false;

  // FILL HISTOGRAM
  h_mjet_pt200->fill(event);
  if(topjets->at(0).pt() > 300) h_mjet_pt300->fill(event);
  if(topjets->at(0).pt() > 400) h_mjet_pt400->fill(event);

  if(topjets->size() == 2){
    h_mjet_2jets_pt200->fill(event);
    if(topjets->at(0).pt() > 300) h_mjet_2jets_pt300->fill(event);
    if(topjets->at(0).pt() > 400) h_mjet_2jets_pt400->fill(event);
  }

  // FINISHED
  return false;
}

// as we want to run the ExampleCycleNew directly with AnalysisModuleRunner,
// make sure the JetMassModule is found by class name. This is ensured by this macro:
UHH2_REGISTER_ANALYSIS_MODULE(JetMassModule)
