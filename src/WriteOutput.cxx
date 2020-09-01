#include "UHH2/core/include/Event.h"
#include "UHH2/JetMass/include/WriteOutput.h"


using namespace uhh2;
using namespace std;

WriteOutput::WriteOutput(uhh2::Context & ctx){

  softdrop_jec.reset(new StandaloneTopJetCorrector(ctx));

  h_pt = ctx.declare_event_output<double>("pt");
  h_N2 = ctx.declare_event_output<double>("N2");
  h_tau32 = ctx.declare_event_output<double>("tau32");
  h_tau21 = ctx.declare_event_output<double>("tau21");
  h_DeepBoost = ctx.declare_event_output<double>("DeepBoostWQCD");
  h_mjet = ctx.declare_event_output<double>("mjet");
  h_mjet_SD = ctx.declare_event_output<double>("mjet_SD");
  h_msubjets = ctx.declare_event_output<double>("msubjets");
  h_mgensubjets = ctx.declare_event_output<double>("mgensubjets");
  h_mgenparticles = ctx.declare_event_output<double>("mgenparticles");
  h_genpt = ctx.declare_event_output<double>("genpt");  
  h_weight = ctx.declare_event_output<double>("weight");
  h_matchedV = ctx.declare_event_output<bool>("matchedV");
  h_genjetpt = ctx.declare_event_output<double>("genjetpt");
  h_jecfactor = ctx.declare_event_output<double>("jecfactor");
  h_jecfactor_SD = ctx.declare_event_output<double>("jecfactor_SD");

  // read from xml file
  auto dataset_type = ctx.get("dataset_type");
  isMC = dataset_type == "MC";

  auto version = ctx.get("dataset_version");
  is_WSample = version.find("WJets") != std::string::npos;
  is_ZSample = version.find("ZJets") != std::string::npos;

  auto channel = ctx.get("channel", "");
  isTopSel = channel == "top";
  // isWfromTopSel = channel == "WfromTop";
  isWSel = channel == "W";

  do_genStudies = string2bool(ctx.get("doGenStudies", "true"));
  
  // read configuration from root file
  TString gfilename = ctx.get("GridFile");
  TFile* gfile = new TFile(locate_file((std::string)gfilename).c_str());
  grid = (TH2F*) gfile->Get("grid");
  TH1F* h_cat = (TH1F*) gfile->Get("categories");
  for(int bin=1; bin<=h_cat->GetXaxis()->GetNbins(); bin++) categories.push_back(h_cat->GetXaxis()->GetBinLabel(bin));
  ConstructOtherIDs();
  if(categories.size() > all_cat.size()+1) throw runtime_error("you gave too many categories");
  if(categories.size() + otherIDs.size() -1 != 8) throw runtime_error("categories and size of 'other' does not match");

  // get binnings and size of variation
  Nbins_pt =  grid->GetXaxis()->GetNbins();
  Nbins_eta = grid->GetYaxis()->GetNbins();
  Nbins_cat = categories.size();

  // name and create all handles for mjet variations
  if(isMC){
    h_jetmass_variations.resize(Nbins_pt);
    for(int i=0; i<Nbins_pt; i++){
      h_jetmass_variations[i].resize(Nbins_eta);
      for(int j=0; j<Nbins_eta; j++){
        h_jetmass_variations[i][j].resize(Nbins_cat);
        for(int k=0; k<Nbins_cat; k++){
          TString bin_name = "_" + to_string(i) + "_" + to_string(j) + "_" + categories[k];
          TString handlename = "mjet"+bin_name;
          h_jetmass_variations[i][j][k] = ctx.declare_event_output<std::vector<double>>((string)handlename);
        }
      }
    }
  }


  MatchV_sel.reset(new MatchingSelection(ctx, is_WSample ? MatchingSelection::oIsMergedGenWNotFromTop : MatchingSelection::oIsMergedGenZ ));

}

bool WriteOutput::process(uhh2::Event & event){

  vector<TopJet>* topjets = event.topjets;
  if(topjets->size() < 1) return false;
  vector<Jet> subjets = topjets->at(0).subjets();
  vector<PFParticle>* allparticles = event.pfparticles;

  vector<PFParticle> particles;

  LorentzVector softdrop_jet_raw;
  // find all pf particles inside the subjets
  for(auto subjet: subjets){
    softdrop_jet_raw += subjet.v4()*subjet.JEC_factor_raw();
    for(const auto candInd : subjet.pfcand_indexs()){
      particles.push_back(allparticles->at(candInd));
    }
  }

  double pt = topjets->at(0).v4().Pt();
  double N2 = topjets->at(0).ecfN2_beta1();
  double mjet = CalculateMJet(particles);
  double jecfactor = 1.0/topjets->at(0).JEC_factor_raw();
  double jecfactor_SD = softdrop_jec->getJecFactor(event,softdrop_jet_raw);
  double mjet_SD = topjets->at(0).softdropmass();

  double deepboost = topjets->at(0).btag_DeepBoosted_WvsQCD();
  double tau32 = 0;
  if(topjets->at(0).tau2() > 0) tau32 = topjets->at(0).tau3()/topjets->at(0).tau2();
  double tau21 = 0;
  if(topjets->at(0).tau1() > 0) tau21 = topjets->at(0).tau2()/topjets->at(0).tau1();

  // set variations for MC
  if(isMC){
    for(int i=0; i<Nbins_pt; i++){
      for(int j=0; j<Nbins_eta; j++){
        for(int k=0; k<Nbins_cat; k++){
          int ptbin = i+1;
          int etabin = j+1;
          vector<double> mjet_var = CalculateMJetVariation(particles, ptbin, etabin, categories[k]);
          event.set(h_jetmass_variations[i][j][k], mjet_var);
        }
      }
    }
  }

  // V matching
  bool Vmatched = false;
  double genjetpt = -1;
  if(isWSel && (is_WSample || is_ZSample)){
    Vmatched = MatchV_sel->passes_matching(event,event.topjets->at(0));

    //get genjet pt for k factors
    const GenJet * closest_genjet_1 = closestParticle(event.topjets->at(0), *event.genjets);
    const GenJet * closest_genjet_2 = event.topjets->size() > 1 ? closestParticle(event.topjets->at(1), *event.genjets) : closest_genjet_1;
    float gen_pt_1 = closest_genjet_1 ? closest_genjet_1->pt() : -9999;
    float gen_pt_2 = closest_genjet_2 ? closest_genjet_2->pt() : -9999;
    genjetpt = Vmatched ? gen_pt_1 : gen_pt_2;
  }

  float genpt,m_genparticles,m_gensubjets;

  if(isMC && do_genStudies){    
    if(event.gentopjets->size()<1)return false;
    const GenTopJet * matched_gentopjet = closestParticle(event.topjets->at(0), *event.gentopjets);    
    vector<GenJet> gensubjets = matched_gentopjet->subjets();
    LorentzVector softdrop_genjet;
    for(auto gensubjet: gensubjets){
      softdrop_genjet += gensubjet.v4();
    }
    m_gensubjets = softdrop_genjet.M();    
    genpt =  matched_gentopjet->pt();
  }else{
    genpt = -1.;
    m_gensubjets = -1.;
    m_genparticles = -1.;
  }
  
  event.set(h_pt, pt);
  event.set(h_mjet, mjet);
  
  event.set(h_msubjets, softdrop_jet_raw.M());
  event.set(h_mjet_SD,mjet_SD);

  event.set(h_mgensubjets, m_gensubjets);
  event.set(h_mgenparticles, m_genparticles);
  event.set(h_genpt, genpt);

  event.set(h_N2, N2);
  event.set(h_tau32, tau32);
  event.set(h_tau21, tau21);
  event.set(h_DeepBoost, deepboost);
  event.set(h_weight, event.weight);
  event.set(h_matchedV, Vmatched);
  event.set(h_genjetpt, genjetpt);
  event.set(h_jecfactor, jecfactor);
  event.set(h_jecfactor_SD, jecfactor_SD);

  return true;
}

double WriteOutput::CalculateMJet(vector<PFParticle> Particles){
  LorentzVector jet_v4;
  for(auto part:Particles){
    LorentzVectorXYZE old_v4XYZ = toXYZ(part.v4());
    double puppiWeight = part.puppiWeight();
    PFParticle p = part;
    p.set_v4(toPtEtaPhi(old_v4XYZ * puppiWeight));
    jet_v4 += p.v4();
  }
  double mjet = jet_v4.M();
  return mjet;
}



vector<double> WriteOutput::CalculateMJetVariation(vector<PFParticle>particles, int i, int j, TString cat){
  LorentzVector up, down;
  for(auto part:particles){
    LorentzVectorXYZE old_v4XYZ = toXYZ(part.v4());
    double puppiWeight = part.puppiWeight();
    PFParticle p = part;
    p.set_v4(toPtEtaPhi(old_v4XYZ * puppiWeight));
    double pt = p.pt();
    double eta = fabs(p.eta());
    int ptbin = grid->GetXaxis()->FindBin(pt);
    int etabin = grid->GetYaxis()->FindBin(eta);
    // if bin is overflow, set manually to last bin
    if(ptbin > Nbins_pt) ptbin = Nbins_pt;
    if(etabin > Nbins_eta) etabin = Nbins_eta;
    double factor = 1.0;
    if(isMC && ptbin == i && etabin == j && inCategory(p,cat)){
      up += (factor+variation) * p.v4();
      down += (factor-variation) * p.v4();
    }
    else{
      up += p.v4();
      down += p.v4();
    }
  }
  return (vector<double>) {up.M(), down.M()};
}

bool WriteOutput::inCategory(PFParticle p, TString cat){
  if(cat == "all") return true;
  bool inCat = false;
  int id = p.particleID();
  for(unsigned int i=0; i<all_cat.size(); i++){
    if(cat == all_cat[i] && abs(id) == i) inCat = true;
  }
  if(cat == "other"){
    for(auto oID: otherIDs){
      if(id == oID) inCat = true;
    }
  }
  return inCat;
}

void WriteOutput::ConstructOtherIDs(){
  vector<bool> AddIDToOther(8, true);
  for(unsigned int i=0; i<all_cat.size(); i++){
    for(auto cat: categories){
      if(cat == all_cat[i]) AddIDToOther[i] = false;
    }
  }
  for(unsigned int id=0; id<AddIDToOther.size(); id++){
    if(AddIDToOther[id]) otherIDs.push_back(id);
  }
  return;
}
