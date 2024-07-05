#!/usr/bin/env pythonJMS.sh
import uproot
from coffea.util import save, load
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
import awkward as ak
import hist
from copy import deepcopy
import glob
import coffea.lookup_tools
import os
from utils import jms_correction_files, year_alias
import correctionlib
hep.style.use("CMS")


ddtmaps_n2_path = "/nfs/dust/cms/user/hinzmann/jetmass/ddtmaps/ddtmaps_n2.npy"
ddtmaps_particlenet_path = "/nfs/dust/cms/user/hinzmann/jetmass/ddtmaps/ddtmaps_particlenet.npy"


def load_tree(
        dirname: str,
        fname_pattern: str,
        year: str,
):
    tree_name = "AnalysisTree"

    tree_paths = [
        f"{fpath}:{tree_name}"
        for fpath in glob.glob(dirname + fname_pattern)
        if uproot.open(fpath)[tree_name].num_entries > 0
    ]

    def filter_(x):
        return x in [
            "pt_gen_ak8",
            "msd_gen_ak8",
            "pass_gen_selection",
            "n2_beta1_gen",
            "gentopjet_n2_0",
            "trigger_bits",
            "jetpfid",
            "pass_reco_selection",
            "pt",
            "IsMergedWZ",
            "mjet",
            "jecfactor",
            "weight",
            "eta",
            "phi",
            "ParticleNetMDDiscriminators_XbbvsQCD",
            "N2",
            "dR_reco_gen",
        ]

    events = uproot.concatenate(tree_paths, filter_name=filter_)

    events['rho'] = 2*np.log(events.mjet/events.pt)

    return events


def create_hists(events, year, n2_max=-999.0, nomatching=False):
    trigger_scalefactors = correctionlib.CorrectionSet.from_file(
        "/nfs/dust/cms/user/hinzmann/jetmass/JetMassNotebooks/data/"
        + "HLT_AK8PFJet_MC_trigger_sf_c2e731345f.json"
    )
    msd_corrector = correctionlib.CorrectionSet.from_file(
        "/nfs/dust/cms/user/hinzmann/jetmass/JetMass/python/"
        + jms_correction_files["notagger"]
    )[f"response_g_jec_{year}"]
    trigger_sf_evaluator_450 = trigger_scalefactors[f"HLT_AK8PFJet450_triggersf_{year}"]
    trigger_sf_evaluator_500 = trigger_scalefactors[f"HLT_AK8PFJet500_triggersf_{year}"]
    first_ptbin = events.pt * events.jecfactor < 650.0

    triggersf_weight = ak.where(
        first_ptbin,
        trigger_sf_evaluator_450.evaluate(events.pt * events.jecfactor, "nominal"),
        trigger_sf_evaluator_500.evaluate(events.pt * events.jecfactor, "nominal"),
    )
    HEM_affected_lumi_fraction = 0.64844705699  # (Run 319077 (17.370008/pb) + Run C + Run D) / all 2018
    HEM_eta_min, HEM_eta_max = -3.2, -1.3
    HEM_phi_min, HEM_phi_max = -1.57, -0.87
    HEM_affected_event_jets = (
        (events.eta < HEM_eta_max)
        & (events.eta > HEM_eta_min)
        & (events.phi < HEM_phi_max)
        & (events.phi > HEM_phi_min)
    )
    hem_weight = np.ones_like(events.weight)
    if year == "UL18":
        print("treating HEM")
        hem_weight = ak.where(
            HEM_affected_event_jets, (1 - HEM_affected_lumi_fraction), np.ones_like(events.weight)
        )

    n2ddtmap = np.load(ddtmaps_n2_path, allow_pickle=True).item()
    corrected_str = '_corrected_pt_mass'

    n2ddt_LUT = coffea.lookup_tools.dense_lookup.dense_lookup(
        n2ddtmap[f'discddt_map_{year}_smooth_4_0p05{corrected_str}'][0],
        dims=(n2ddtmap[f'discddt_map_{year}_smooth_4_0p05{corrected_str}'][1],
              n2ddtmap[f'discddt_map_{year}_smooth_4_0p05{corrected_str}'][2]
              )
    )

    particlenetddtmap = np.load(ddtmaps_particlenet_path, allow_pickle=True).item()
    pNetMDWvsQCDddt_LUT = coffea.lookup_tools.dense_lookup.dense_lookup(
        particlenetddtmap[
            f"discddt_map_{year}_smooth_4_0p975{corrected_str}"
        ][0],
        dims=(
            particlenetddtmap[
                f"discddt_map_{year}_smooth_4_0p975{corrected_str}"
            ][1],
            particlenetddtmap[
                f"discddt_map_{year}_smooth_4_0p975{corrected_str}"
            ][2],
        )
    )

    pass_gen_sel = (events.pass_gen_selection == 1) & (events.msd_gen_ak8 > 30.0)
    msd_correction = 1.0/msd_corrector.evaluate(events.pt*events.jecfactor)
    pass_reco_sel = (
        (events.pass_reco_selection == 1)
        & (events.rho < -2.1)
        & (events.jetpfid == 1)
        # & (events.dR_reco_gen < 0.4)
        & ((events.pt * events.jecfactor) > 575.)
        & (
            (((events.pt * events.jecfactor) < 650.) & (events.trigger_bits[:, 7] == 1))
            | (((events.pt * events.jecfactor) >= 650.) & (events.trigger_bits[:, 8] == 1))
        )
        & ((events.mjet * events.jecfactor * msd_correction) >= 50.)
        & ((events.mjet * events.jecfactor * msd_correction) < 300.)
    )
    if not args.nomatching:
        pass_reco_sel = pass_reco_sel & (events.IsMergedWZ == 1)

    def n2ddt(e):
        return (e.N2 - n2ddt_LUT(e.rho, e.pt)) < 0

    pass_reco_sel_n2ddt = pass_reco_sel & (n2ddt(events))

    def pNetddt(e):
        return (pNetMDWvsQCDddt_LUT(e.rho, e.pt) - e["ParticleNetMDDiscriminators_XbbvsQCD"]) < 0

    pass_reco_sel_pNetddt = pass_reco_sel & (n2ddt(events))

    if n2_max > 0:
        pass_gen_sel = pass_gen_sel & (events.gentopjet_n2_0 < n2_max)

    hists = {}

    pt_gen_edges = np.array([500, 650.0, 800.0, 1200.0, np.inf])
    pt_gen_ax = hist.axis.Variable(pt_gen_edges, name="ptgen", label=r"$p_{T,\mathrm{gen}}$ [GeV]")
    msd_gen_edges = np.array([30., 70.0, 80.0, 90.0, np.inf])
    msd_gen_ax = hist.axis.Variable(msd_gen_edges, name="msdgen", label=r"$m_{\mathrm{SD,gen}}$ [GeV]")

    hists["misses"] = hist.Hist(
        pt_gen_ax,
        msd_gen_ax,
        storage=hist.storage.Weight(),
    )

    hists["2d_fine"] = hist.Hist(
        hist.axis.Regular(100, 0, 2000, name="ptgen"),
        hist.axis.Regular(100, 0, 300, name="msdgen"),
        hist.axis.Regular(100, 0, 2000, name="ptreco"),
        hist.axis.Regular(100, 0, 300, name="msdreco"),
        storage=hist.storage.Weight(),
    )

    hists["2d_final"] = hist.Hist(
        pt_gen_ax,
        msd_gen_ax,
        hist.axis.Variable([500, 575, 650, 725, 800, 1000, 1200, np.inf], name="ptreco"),
        hist.axis.Regular(60, 0, 300, name="msdreco"),
        storage=hist.storage.Weight(),
    )

    for suffix in ["", "_withSF", "_withSFHEM"]:
        print(f"init hists ({suffix})")
        hists[f"total{suffix}"] = hist.Hist(
            pt_gen_ax,
            msd_gen_ax,
            storage=hist.storage.Weight(),
        )

        hists[f"pass_gen_pass_reco{suffix}"] = hist.Hist(
            pt_gen_ax,
            msd_gen_ax,
            storage=hist.storage.Weight(),
        )

        hists[f"pass_gen_pass_reco_pass_dR{suffix}"] = hist.Hist(
            pt_gen_ax,
            msd_gen_ax,
            storage=hist.storage.Weight(),
        )

        hists[f"pass_gen_pass_reco_n2ddt{suffix}"] = hist.Hist(
            pt_gen_ax,
            msd_gen_ax,
            storage=hist.storage.Weight(),
        )
        hists[f"pass_gen_pass_reco_pNetddt{suffix}"] = hist.Hist(
            pt_gen_ax,
            msd_gen_ax,
            storage=hist.storage.Weight(),
        )

    for sf_suffix in ["", "_withSF", "_withSFHEM"]:
        print(f"filling hists ({suffix})")
        weight = deepcopy(events.weight)
        if sf_suffix == "_withSFHEM":
            weight = weight * hem_weight * triggersf_weight
        elif sf_suffix == "_withSF":
            weight = weight * triggersf_weight
        hists[f"pass_gen_pass_reco{sf_suffix}"].fill(
            ptgen=events.pt_gen_ak8[pass_gen_sel & pass_reco_sel],
            msdgen=events.msd_gen_ak8[pass_gen_sel & pass_reco_sel],
            weight=weight[pass_gen_sel & pass_reco_sel],
        )
        hists[f"pass_gen_pass_reco_pass_dR{sf_suffix}"].fill(
            ptgen=events.pt_gen_ak8[pass_gen_sel & pass_reco_sel & (events.dR_reco_gen < 0.4)],
            msdgen=events.msd_gen_ak8[pass_gen_sel & pass_reco_sel & (events.dR_reco_gen < 0.4)],
            weight=weight[pass_gen_sel & pass_reco_sel & (events.dR_reco_gen < 0.4)],
        )
        hists[f"pass_gen_pass_reco_n2ddt{sf_suffix}"].fill(
            ptgen=events.pt_gen_ak8[pass_gen_sel & pass_reco_sel_n2ddt],
            msdgen=events.msd_gen_ak8[pass_gen_sel & pass_reco_sel_n2ddt],
            weight=weight[pass_gen_sel & pass_reco_sel_n2ddt],
        )
        hists[f"pass_gen_pass_reco_pNetddt{sf_suffix}"].fill(
            ptgen=events.pt_gen_ak8[pass_gen_sel & pass_reco_sel_pNetddt],
            msdgen=events.msd_gen_ak8[pass_gen_sel & pass_reco_sel_pNetddt],
            weight=weight[pass_gen_sel & pass_reco_sel_pNetddt],
        )

        hists[f"total{sf_suffix}"].fill(
            ptgen=events.pt_gen_ak8[pass_gen_sel],
            msdgen=events.msd_gen_ak8[pass_gen_sel],
            weight=weight[pass_gen_sel],
        )

    pass_all = (
        (events.pass_gen_selection == 1)
        & (events.dR_reco_gen < 0.4)
        & (events.pass_reco_selection == 1)
        & (events.rho < -2.1)
        & (events.jetpfid == 1)
        # & (events.dR_reco_gen < 0.4)
        & ((events.pt * events.jecfactor) > 575.)
        & (
            (((events.pt * events.jecfactor) < 650.) & (events.trigger_bits[:, 7] == 1))
            | (((events.pt * events.jecfactor) >= 650.) & (events.trigger_bits[:, 8] == 1))
        )
    )
    all_weight = deepcopy(events.weight) * hem_weight * triggersf_weight

    hists["2d_fine"].fill(
        ptgen=events.pt_gen_ak8[pass_all],
        msdgen=events.msd_gen_ak8[pass_all],
        ptreco=(events.pt * events.jecfactor)[pass_all],
        msdreco=(events.mjet * events.jecfactor * msd_correction)[pass_all],
        weight=all_weight[pass_all],
    )

    hists["2d_final"].fill(
        ptgen=events.pt_gen_ak8[pass_all],
        msdgen=events.msd_gen_ak8[pass_all],
        ptreco=(events.pt * events.jecfactor)[pass_all],
        msdreco=(events.mjet * events.jecfactor * msd_correction)[pass_all],
        weight=all_weight[pass_all],
    )

    misses = pass_gen_sel & (~pass_reco_sel | (events.dR_reco_gen > 0.4))

    hists["misses"].fill(
        ptgen=events.pt_gen_ak8[misses],
        msdgen=events.msd_gen_ak8[misses],
        weight=events.weight[misses],
    )
    hists["acceptance"] = [
        (1 - hists["misses"][{"ptgen": ipt}].values() / hists["total"][{"ptgen": ipt}].values(), msd_gen_edges)
        for ipt in range(len(pt_gen_edges) - 1)
    ]

    hists["sf_efficiency"] = {
        f"pass_gen_pass_reco{region}": [
            (
                hists[f"pass_gen_pass_reco{region}_withSFHEM"][{"ptgen": ipt}].values()
                / hists[f"pass_gen_pass_reco{region}"][{"ptgen": ipt}].values(),
                msd_gen_edges,
            )
            for ipt in range(len(pt_gen_edges) - 1)
        ]
        for region in ["", "_pass_dR", "_n2ddt", "_pNetddt"]
    }

    ### Andreas code to print cross sections after gen-selection
    lumis = {
    "UL16preVFP": 19301.591954787407 / 1000.0,
    "UL16postVFP": 16626.734093195286 / 1000.0,
    "UL17": 41479.68052876168 / 1000.0,
    "UL18": 59832.47533908866 / 1000.0,
    }
    lumis["RunII"] = sum(lumis[year] for year in ["UL16preVFP", "UL16postVFP", "UL17", "UL18"])
    pt_gen_ax0 = hist.axis.Variable(np.array([500.0, np.inf]), name="ptgen", label=r"$p_{T,\mathrm{gen}}$ [GeV]")
    msd_gen_ax0 = hist.axis.Variable(np.array([0., np.inf]), name="msdgen", label=r"$m_{\mathrm{SD,gen}}$ [GeV]")
    hists["gen"] = hist.Hist(
            pt_gen_ax0,
            msd_gen_ax0,
            storage=hist.storage.Weight(),
        )
    hists["gen"].fill(
            ptgen=events.pt_gen_ak8,
            msdgen=events.msd_gen_ak8,
            weight=events.weight,
        )
    print("Gen no cut xsec =", hists["gen"].values().sum()/lumis[year])
    hists["gen_pt"] = hist.Hist(
            pt_gen_ax0,
            msd_gen_ax0,
            storage=hist.storage.Weight(),
        )
    hists["gen_pt"].fill(
            ptgen=events.pt_gen_ak8[(events.pt_gen_ak8 > 650.0)],
            msdgen=events.msd_gen_ak8[(events.pt_gen_ak8 > 650.0)],
            weight=events.weight[(events.pt_gen_ak8 > 650.0)],
        )
    print("Gen pT cut xsec =", hists["gen_pt"].values().sum()/lumis[year])
    hists["gen_pt_msd"] = hist.Hist(
            pt_gen_ax0,
            msd_gen_ax0,
            storage=hist.storage.Weight(),
        )
    hists["gen_pt_msd"].fill(
            ptgen=events.pt_gen_ak8[(events.pt_gen_ak8 > 650.0) & (events.msd_gen_ak8 > 30.0)],
            msdgen=events.msd_gen_ak8[(events.pt_gen_ak8 > 650.0) & (events.msd_gen_ak8 > 30.0)],
            weight=events.weight[(events.pt_gen_ak8 > 650.0) & (events.msd_gen_ak8 > 30.0)],
        )
    print("Gen pT, mSD cut xsec =", hists["gen_pt_msd"].values().sum()/lumis[year])
    hists["gen_pt_msd_n2"] = hist.Hist(
            pt_gen_ax0,
            msd_gen_ax0,
            storage=hist.storage.Weight(),
        )
    hists["gen_pt_msd_n2"].fill(
            ptgen=events.pt_gen_ak8[(events.pt_gen_ak8 > 650.0) & (events.msd_gen_ak8 > 30.0) & (events.gentopjet_n2_0 < n2_max)],
            msdgen=events.msd_gen_ak8[(events.pt_gen_ak8 > 650.0) & (events.msd_gen_ak8 > 30.0) & (events.gentopjet_n2_0 < n2_max)],
            weight=events.weight[(events.pt_gen_ak8 > 650.0) & (events.msd_gen_ak8 > 30.0) & (events.gentopjet_n2_0 < n2_max)],
        )
    print("Gen pT, mSD, N2 cut xsec =", hists["gen_pt_msd_n2"].values().sum()/lumis[year])
    #####

    return hists


def plot_acceptance(hists, outdir, year):
    print("plotting acceptance")

    fs = 18
    pt_edges = hists["misses"].axes["ptgen"].edges

    acc = hists["acceptance"]
    markers = ["o", "v", "^", "s", "x"]
    f, ax = plt.subplots()
    for ipt in range(len(pt_edges)-1):
        if ipt == len(pt_edges)-2 and pt_edges[-1] == np.inf:
            pt_tex = r"$p_{T,\mathrm{gen}} \geq %i$ GeV" % (pt_edges[ipt])
        else:
            pt_tex = r"$%i \leq p_{T,\mathrm{gen}} < %i$ GeV" % (pt_edges[ipt], pt_edges[ipt + 1])
        msd_edges = acc[ipt][1]
        msd_edges[-1] = 260
        msd_centers = msd_edges[:-1] + 0.5 * np.diff(msd_edges)
        errbar_kwargs = dict(
            xerr=[msd_centers - msd_edges[:-1], msd_edges[1:] - msd_centers], markersize=6, lw=2, fmt=markers[ipt]
        )
        ax.errorbar(msd_centers, acc[ipt][0], label=pt_tex, **errbar_kwargs)
    ax.legend(fontsize=fs+2)
    hep.cms.label("Work in Progress", year=year_alias.get(year, year), fontsize=fs + 1, ax=ax, data=False)
    ax.set_ylim(0, 1.0)
    # ax.set_ylabel(r"$1-\frac{N( \mathrm{pass-gen} \wedge \mathrm{fail-reco})}{N(\mathrm{pass-gen})}$")
    ax.set_ylabel(r"acceptance")
    ax.set_xlabel(r"$m_{\mathrm{SD,gen}}$ [GeV]")
    ax.set_xlim(0, 270)
    ax.set_xticks([30.0, 50.0, 100.0, 150.0, 200.0, 260.0], [30, 50, 100, 150, 200, r"$\infty$"])
    f.savefig(f"{outdir}/acceptance_{year}.pdf", bbox_inches="tight")


def plot_sf_efficiency(hists, outdir, year):
    print("plotting sf efficiency")

    fs = 18
    pt_edges = hists["misses"].axes["ptgen"].edges

    sf_eff = hists["sf_efficiency"]["pass_gen_pass_reco_pass_dR"]
    markers = ["o", "v", "^", "s", "x"]
    f, ax = plt.subplots()
    for ipt in range(len(pt_edges)-1):
        if ipt == len(pt_edges)-2 and pt_edges[-1] == np.inf:
            pt_tex = r"$p_{T,\mathrm{gen}} \geq %i$ GeV" % (pt_edges[ipt])
        else:
            pt_tex = r"$%i \leq p_{T,\mathrm{gen}} < %i$ GeV" % (pt_edges[ipt], pt_edges[ipt + 1])
        msd_edges = sf_eff[ipt][1]
        msd_edges[-1] = 260
        msd_centers = msd_edges[:-1] + 0.5 * np.diff(msd_edges)
        errbar_kwargs = dict(
            xerr=[msd_centers - msd_edges[:-1], msd_edges[1:] - msd_centers], markersize=6, lw=2, fmt=markers[ipt]
        )
        ax.errorbar(msd_centers, sf_eff[ipt][0], label=pt_tex, **errbar_kwargs)
    ax.legend(fontsize=fs+2)
    hep.cms.label("Work in Progress", year=year_alias.get(year, year), fontsize=fs + 1, ax=ax, data=False)
    ax.set_ylim(0, 1.1)
    # ax.set_ylabel(r"$1-\frac{N( \mathrm{pass-gen} \wedge \mathrm{fail-reco})}{N(\mathrm{pass-gen})}$")
    ax.set_ylabel(r"$\varepsilon_{\mathrm{scale factors}}$")
    ax.set_xlabel(r"$m_{\mathrm{SD,gen}}$ [GeV]")
    ax.set_xlim(0, 270)
    ax.set_xticks([30.0, 50.0, 100.0, 150.0, 200.0, 260.0], [30, 50, 100, 150, 200, r"$\infty$"])
    f.savefig(f"{outdir}/sf_efficiency_{year}.pdf", bbox_inches="tight")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--year", default="UL17")
    parser.add_argument("--load", action="store_true")
    parser.add_argument("--outdir", "-o", default="acceptance_plots")
    parser.add_argument("--n2cut", default=-999.0, type=float)
    parser.add_argument("--nomatching", action="store_true")
    args = parser.parse_args()

    if "nomatching" not in args.outdir and args.nomatching:
        args.outdir += "_nomatching"

    outdir = os.path.abspath(args.outdir)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    print(f"saving plot in {outdir}")

    outfile = f"{args.outdir}/misses_acceptance{args.year}.coffea"
    if args.load:
        out = load(outfile)
    else:
        tree = load_tree(
            dirname=f"/nfs/dust/cms/user/albrechs/UHH2/JetMassOutput/vjetsTrees/workdir_vjets_{args.year}/",
            fname_pattern="*WJetsToQQ*.root",
            year=args.year,
        )
        out = {}
        out.update(create_hists(tree, args.year,  args.n2cut, args.nomatching))
        save(out, outfile)

    plot_acceptance(out, outdir, args.year)
    plot_sf_efficiency(out, outdir, args.year)
