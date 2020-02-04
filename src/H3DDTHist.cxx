#include "UHH2/JetMass/include/H3DDTHist.h"
#include "UHH2/core/include/Event.h"

#include "TH1F.h"
#include "TH2F.h"
#include <iostream>

using namespace std;
using namespace uhh2;

H3DDTHist::H3DDTHist(Context & ctx, const string & dirname): Hists(ctx, dirname){
  auto dataset_type = ctx.get("dataset_type");
  isMC = dataset_type == "MC";
  auto version = ctx.get("dataset_version");
  use_PFMass = TString(dirname).Contains("PFMass");
  use_SD = ! TString(dirname).Contains("noSD");

  int pt_nbins = 300;
  float pt_min = 0.0;
  float pt_max = 1500.0;

  // int mSD_nbins = 250;
  // float mSD_min = 0.0;
  // float mSD_max = 500;

  int rho_nbins = 300;
  float rho_min = -10.0;
  float rho_max = 0.0;

  int disc_nbins = 110;
  float disc_min = -0.1;
  float disc_max = 1.0;
  
  N2_v_pT_v_rho_oldBinning=book<TH3D>("N2_v_pT_v_rho_oldBinning","x=#rho,y=p_{T},z=N_2",rho_nbins,-10,0,300,0,2000,300,-1.5,1); 

  N2_v_pT_v_rho=book<TH3D>("N2_v_pT_v_rho","x=#rho,y=p_{T},z=N_{2}^{#beta=1}",rho_nbins,rho_min,rho_max,pt_nbins, pt_min, pt_max,disc_nbins,disc_min, disc_max);
  N2_beta2_v_pT_v_rho=book<TH3D>("N2_beta2_v_pT_v_rho","x=#rho,y=p_{T},z=N_{2}^{#beta=2}",rho_nbins,rho_min,rho_max,pt_nbins, pt_min, pt_max,disc_nbins,disc_min, disc_max);
  DeepBoosted_WvsQCD_v_pT_v_rho=book<TH3D>("DeepBoosted_WvsQCD_v_pT_v_rho","x=#rho,y=p_{T},z=N_2",rho_nbins,rho_min,rho_max,pt_nbins, pt_min, pt_max,disc_nbins,disc_min, disc_max);  

  // N2_v_pT_v_mSD=book<TH3D>("N2_v_pT_v_mSD","x=m_{SD},y=p_{T},z=N_{2}^{#beta=1}",mSD_nbins,mSD_min, mSD_max,pt_nbins, pt_min, pt_max,disc_nbins,disc_min, disc_max);
  // N2_beta2_v_pT_v_mSD=book<TH3D>("N2_beta2_v_pT_v_mSD","x=m_{SD},y=p_{T},z=N_{2}^{#beta=2}",mSD_nbins,mSD_min, mSD_max,pt_nbins, pt_min, pt_max,disc_nbins,disc_min, disc_max);
  // DeepBoosted_WvsQCD_v_pT_v_mSD=book<TH3D>("DeepBoosted_WvsQCD_v_pT_v_mSD","x=m_{SD},y=p_{T},z=N_2",mSD_nbins,mSD_min, mSD_max,pt_nbins, pt_min, pt_max,disc_nbins,disc_min, disc_max); 
}

void H3DDTHist::fill(const Event & event){
  auto weight = event.weight;
  if(event.topjets->size() <1) return;
  auto jet = event.topjets->at(0);
  float mSD;
   if(use_PFMass){
    vector<PFParticle> * pfparticles = event.pfparticles;
    vector<PFParticle> particles;
    if(use_SD){
      for(auto subjet: jet.subjets()){
        for(const auto candInd: subjet.pfcand_indexs()){
          particles.push_back(pfparticles->at(candInd));
        }
      }
    }else{
      for(const auto candInd : jet.pfcand_indexs()){
        particles.push_back(pfparticles->at(candInd));
      }
    }
    LorentzVector jet_sd_v4;
    for(auto p: particles){
      jet_sd_v4 += p.v4();
    }
    mSD = jet_sd_v4.M();
  }else{
    if(use_SD) mSD = jet.softdropmass();
    else mSD = jet.v4().M();
  }
  double rho= 2* TMath::Log(mSD/jet.pt());

  N2_v_pT_v_rho_oldBinning->Fill(rho,jet.pt(),jet.ecfN2_beta1(),weight);

  N2_v_pT_v_rho->Fill(rho,jet.pt(),jet.ecfN2_beta1(),weight);
  N2_beta2_v_pT_v_rho->Fill(rho,jet.pt(),jet.ecfN2_beta2(),weight);
  DeepBoosted_WvsQCD_v_pT_v_rho->Fill(rho,jet.pt(),jet.btag_DeepBoosted_WvsQCD(),weight); 

  // N2_v_pT_v_mSD->Fill(mSD,jet.pt(),jet.ecfN2_beta1(),weight);
  // N2_beta2_v_pT_v_mSD->Fill(mSD,jet.pt(),jet.ecfN2_beta2(),weight);
  // DeepBoosted_WvsQCD_v_pT_v_rho->Fill(rho,jet.pt(),jet.btag_DeepBoosted_WvsQCD(),weight); 

}

H3DDTHist::~H3DDTHist(){}
