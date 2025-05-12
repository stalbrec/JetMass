#!/usr/bin/env pythonJMS.sh
from coffea.util import load, save


def skim_coffea(fname, fname_skimmed):
    out = load(fname)
    regions = ["pass", "fail", "inclusive"]
    hist_pattern = "vjets_mjet_unfolding_{REGION}"
    hist_id = {"fakes": False, "jecAppliedOn": "pt&mJ"}
    hists = {hist_pattern.format(REGION=region): out[hist_pattern.format(REGION=region)][hist_id] for region in regions}
    # sum remaining ptreco
    hists = {k: v.integrate("ptreco", 575j).integrate("mJreco", 50j, 300j) for k, v in hists.items()}
    save(hists, fname_skimmed)
    del out


years = ["UL16preVFP", "UL16postVFP", "UL17", "UL18"]
taggers = ["_particlenetDDT", ""]
hist_dir = "/nfs/dust/cms/user/albrechs/JetMassFits/coffea_hists/msdgen30n2cut/"
theory_systs = ["v_qcd", "w_ewk"]
theory_vars = [f"_{syst}_{direc}" for syst in theory_systs for direc in ["up", "down"]]
for year in years:
    for tagger in taggers:
        for var in [""] + theory_vars:
            fname = f"{hist_dir}/templates_{year}{tagger}{var}.coffea"
            fname_skimmed = f"{hist_dir}/templates_{year}{tagger}{var}_mctruth.coffea"
            print(f"skimming {fname}")
            skim_coffea(fname, fname_skimmed)
