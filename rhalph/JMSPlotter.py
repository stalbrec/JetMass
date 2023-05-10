#!/usr/bin/env pythonJMS.sh
# import matplotlib
# try:
#     matplotlib.use("module://imgcat")
# except ModuleNotFoundError as e:
#     print(e)
#     print("Install imgcat with `pip install imgcat`")
import numpy as np  # noqa
from typing import List, Dict, Union, Tuple # noqa
import json  # noqa
import matplotlib.pyplot as plt  # noqa
import matplotlib as mpl  # noqa
import mplhep as hep  # noqa
from matplotlib.ticker import FormatStrFormatter  # noqa
import re  # noqa
from copy import deepcopy  # noqa
import os # noqa
hep.style.use("CMS")
# mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=[
#     # "#1b9e77",
#     # "#d95f02",
#     # "#7570b3",
#     # "#e7298a",
#     # "#66a61e",
#     # "#e6ab02",

#     # "#66c2a5",
#     # "#fc8d62",
#     # "#8da0cb",
#     # "#e78ac3",
#     # "#a6d854",
#     # "#ffd92f",

#     # "#e41a1c",
#     # "#377eb8",
#     # "#ffff33",
#     # "#4daf4a",
#     # "#984ea3",
#     # "#ff7f00",

#     "ff5c5c",
#     "f9c784",
#     "7eb2dd",
#     "4c9f70",
#     "762e42"
# ])


class JMSPlotter:
    def __init__(
        self,
        fit_results: Dict,
        fits_to_plot: List[str],
        separate_jec_fits_: bool = False,
        legacy_json_format: bool = True,  # for now allow errors and jms central best fit value to be stored in same list 'vals' # noqa
    ) -> None:
        self.lFits = fits_to_plot
        self.dFitResults = {k: v for k, v in fit_results.items() if k in self.lFits}
        self.bLegacyJsonFormat = legacy_json_format

        self.f, self.ax = None, None

        self.sJmsParameters = set()
        self.sPtLow = set()
        self.sPtHigh = set()
        self.dJms = {f: [] for f in self.lFits}
        self.dErrorsUp = {f: [] for f in self.lFits}
        self.dErrorsDown = {f: [] for f in self.lFits}
        self.dUncSources = {f: ["total"] for f in self.lFits}
        self.aJmsParameters = None
        self.aPtLow = None
        self.aPtHigh = None
        self.aPtEdges = None
        self.extract_fit_results()

        self.dHists = {}

    def extract_fit_results(self) -> None:
        for fit in self.lFits:
            for par in self.dFitResults[fit]["jms"].keys():
                # getting parameter names for easier access ?
                self.sJmsParameters.add(par)
                # getting lower and upper pt_edges to construct pt_edges
                lo, hi = self.dFitResults[fit]["jms"][par]["edges"]
                self.sPtLow.add(float(lo))
                self.sPtHigh.add(float(hi))

        self.aJmsParameters = np.array(sorted(self.sJmsParameters))
        self.aPtLow = np.array(sorted(self.sPtLow))
        self.aPtHigh = np.array(sorted(self.sPtHigh))
        self.aPtEdges = np.concatenate([self.aPtLow, self.aPtHigh[-1:]])

        for fit in self.lFits:
            if len(self.dJms.get(fit, [])) != 0:
                continue
            else:
                self.dJms.update({fit: []})
                self.dErrorsUp.update({fit: []})
                self.dErrorsDown.update({fit: []})

            jms = self.get_fit_result_pars(fit, val="central")
            err_up = self.get_fit_result_pars(fit, val="err_up")
            err_down = self.get_fit_result_pars(fit, val="err_down")
            if not self.bLegacyJsonFormat:
                for ipar, par in enumerate(self.aJmsParameters):
                    if np.isnan(jms[ipar]) or jms[ipar] == 0.0:
                        continue
                    else:
                        self.dUncSources[fit] = self.get_fit_result_par(fit, par, self.dFitResults, "unc_sources")
                        break
            self.dJms[fit] = np.array(jms)
            ndim = err_up.ndim
            if ndim == 1:
                err_up = [err_up]
                err_down = [err_down]
            elif ndim == 2:
                err_up = list(err_up.T)
                err_down = list(err_down.T)

            self.dErrorsUp[fit] += err_up
            self.dErrorsDown[fit] += err_down

    def get_fit_result_pars(self, fit, fit_results: Dict = None, val="central"):
        if fit_results is None:
            fit_results = self.dFitResults
        result = [self.get_fit_result_par(fit, par, fit_results, val) for par in self.aJmsParameters]

        # check if all elements are float at first
        if all(isinstance(element, float) for element in result):
            return np.nan_to_num(result)

        # regularize result. In Fits without points in some bins the result is irregular as the default NaN placeholder
        # has always dim (1,X).with split uncertainty this will be (N,X)
        sublist_lengths = [len(sublist) for sublist in result]
        if sublist_lengths.count(sublist_lengths[0]) == len(sublist_lengths):
            # all sublist dimensions are the same. result is regular
            return np.nan_to_num(result)
        max_dim = max(sublist_lengths)
        for isub in range(len(result)):
            if sublist_lengths[isub] < max_dim:
                if sublist_lengths[isub] != 1:
                    raise NotImplementedError(
                        "Found sublist dimension != 1 in irregular uncertainty array. This should not happen!"
                    )
                result[isub] = result[isub] * max_dim

        return np.nan_to_num(result)

    def get_fit_result_par(self, fit, par, fit_results: Dict, val="central"):
        nan_dict = {"central": np.nan, "err_up": [np.nan], "err_down": [np.nan], "unc_sources": ["total"]}
        if self.bLegacyJsonFormat:
            nan_dict = {"vals": [np.nan, np.nan, np.nan]}
            idx = ["central", "err_up", "err_down"]
            return fit_results[fit]["jms"].get(par, nan_dict)["vals"][idx.index(val)]
        return fit_results[fit]["jms"].get(par, nan_dict)[val]

    def load_fit_results(self, fit_results: Dict, fits_to_plot: List[str], postfix: str = "") -> None:
        updated_fit_names = [fit + postfix for fit in fits_to_plot]
        if any(fit in self.lFits for fit in updated_fit_names):
            raise RuntimeError(
                f"Can't add fits ({fits_to_plot}). "
                f"At least one of them is already in list of fits to plot ({self.lFits})."
                "If these fits are from another set of fitResults, you can specify a postfix "
                "to be added to the fit-names."
            )
        [self.lFits.append(fit) for fit in updated_fit_names]
        self.dFitResults.update({k + postfix: v for k, v in fit_results.items() if k in fits_to_plot})
        self.extract_fit_results()

    def load_JEC_fits(self, fit_results: Dict, fits: List[str] = None):
        if fits is None:
            fits = self.lFits
        for fit in self.lFits:
            # pars = sorted(self.dFitResults[fit]["jms"].keys())
            if f"{fit}JECUP" in fit_results and f"{fit}JECDOWN" in fit_results:
                unc_down = np.abs(
                    self.dJms[fit] - np.array(self.get_fit_result_pars(f"{fit}JECDOWN", fit_results, "central"))
                )
                unc_up = np.abs(
                    self.dJms[fit] - np.array(self.get_fit_result_pars(f"{fit}JECUP", fit_results, "central"))
                )
                unc_lower = np.min([unc_down, unc_up], axis=0)
                unc_upper = np.max([unc_down, unc_up], axis=0)
                unc = np.average([unc_lower, unc_upper], axis=0)
                # unc = np.where(unc_upper / self.dJms[fit] < 2.0, unc_upper, unc_lower)
                unc = [unc_up, unc_down]
                self.add_uncertainty(fit, unc, "JEC as input")

    def add_uncertainty(
        self, fit, uncertainty: Union[List[np.ndarray], np.ndarray], source: Union[List[str], str] = "dummy"
    ) -> None:
        if isinstance(uncertainty, np.ndarray):
            uncertainty = [uncertainty, uncertainty]
        self.dUncSources[fit].append(source)
        self.dErrorsUp[fit].append(uncertainty[1])
        self.dErrorsDown[fit].append(uncertainty[0])
        for fit in self.lFits:
            self.dHists[fit].update_yerr([np.array(self.dErrorsUp[fit]), np.array(self.dErrorsDown[fit])], source)

    def construct_hists(self):
        for fit in self.lFits:
            self.dHists[fit] = SFHist(
                self.dJms[fit],
                self.aPtEdges,
                True,
                [np.array(self.dErrorsUp[fit]), np.array(self.dErrorsDown[fit])],
                self.dUncSources[fit],
            )

    def __str__(self) -> str:
        return (
            "JMSPlotter with input:"
            + "".join([f"\n{fit} {len(self.dFitResults[fit]['jms'])} parameters" for fit in self.lFits])
            + f"\npt-edges: {self.aPtEdges}"
        )


class SFHist(object):
    def __init__(
        self,
        content: np.ndarray,
        edges: np.ndarray,
        xerr: Union[bool, np.ndarray],
        yerr: Union[np.ndarray, List[np.ndarray]],
        sources: Union[List[str], str] = ""
    ) -> None:
        # mask
        self.mask = ~np.isnan(content)
        m = np.ones_like(content, dtype=bool)
        self._bin_content = content[m]
        self._pt_low = edges[:-1][m]
        self._pt_high = edges[1:][m]
        self.edges = np.concatenate([self.pt_low, self.pt_high[-1:]])
        self.bin_centers = 0.5 * (self.edges[:-1] + self.edges[1:])
        self.unc_sources = []
        self.xerr = xerr
        self.update_yerr(yerr, sources)
        self.color = None

    @property
    def bin_content(self, m: np.ndarray = None) -> np.ndarray:
        if m is None:
            m = self.mask
        return self._bin_content[m]

    @property
    def pt_low(self, m: np.ndarray = None) -> np.ndarray:
        if m is None:
            m = self.mask
        return self._pt_low[m]

    @property
    def pt_high(self, m: np.ndarray = None) -> np.ndarray:
        if m is None:
            m = self.mask
        return self._pt_high[m]

    def update_yerr(self, yerr, source: Union[List[str], str] = ""):
        if isinstance(source, list):
            [self.unc_sources.append(s) for s in source if s not in self.unc_sources]
        elif isinstance(source, str):
            if source != "":
                if source not in self.unc_sources:
                    self.unc_sources.add(source)
        else:
            raise NotImplementedError(f"Handling of provided uncertainty source type ({type(source)}) not implemented ")

        if isinstance(yerr, np.ndarray):
            self.yerr_up = [np.abs(yerr)]
            self.yerr_down = [np.abs(yerr)]

        elif isinstance(yerr, list):
            if all(isinstance(entry, np.ndarray) for entry in yerr) and len(yerr) == 2:
                if len(yerr[0].shape) == 3:
                    yerr = [yerr[0][0], yerr[1][0]]
                self.yerr_up = np.abs(yerr[0])
                self.yerr_down = np.abs(yerr[1])

            else:
                raise NotImplementedError("wrong format of yerr")
        else:
            self.yerr_up = None
            self.yerr_down = None

    def yerr(
        self, sum_sources=True, source_index: int = -1, mask: np.ndarray = None
    ) -> Union[List[np.ndarray], List[List[np.ndarray]]]:
        if mask is None:
            mask = self.mask

        if sum_sources and source_index >= 0:
            raise RuntimeError("You should not sum sources and specify sources index at the same time!")
        if sum_sources:
            up, down = np.zeros_like(self._bin_content), np.zeros_like(self._bin_content)
            if self.yerr_down is not None and self.yerr_up is not None:
                up += np.sqrt((self.yerr_up**2).sum(axis=0)).flatten()
                down += np.sqrt((self.yerr_down**2).sum(axis=0)).flatten()
            return np.array([down, up])[:, mask]
        elif (source_index >= 0):
            if source_index > self.yerr_down.shape[0]-1:
                raise RuntimeError(f"yerr out of bounds (lenght = {self.yerr_down.shape[0]}, requested {source_index})")
            return np.array([self.yerr_down[source_index, :], self.yerr_up[source_index, :]])[:, mask]
        else:
            return np.array([self.yerr_down, self.yerr_up])[:, :, mask]

    def to_numpy(self) -> Tuple[np.ndarray]:
        return (self.bin_content, self.edges)

    def plot_errorbar(
        self,
        ax=None,
        split_uncertainty: Union[bool, List[str]] = False,
        uncertainty_index: int = -1,
        **kwargs,
    ) -> None:
        if "color" in kwargs:
            self.save_color(kwargs["color"])
        elif self.color is not None:
            kwargs["color"] = self.color

        error_linestyle = "-"
        if "linestyle" in kwargs:
            error_linestyle = kwargs["linestyle"]
            del kwargs["linestyle"]

        if ax is None:
            ax = plt.gca()
        xerr = [self.bin_centers - self.pt_low, self.pt_high - self.bin_centers] if self.xerr else None
        if split_uncertainty:
            n_uncertainties = self.yerr(False)[0].shape[0]
            split_unc_ebs = []
            for iuncertainty in range(1, n_uncertainties):
                kwargs_copy = deepcopy(kwargs)
                kwargs_copy["capsize"] = 10
                del kwargs_copy["label"]
                split_unc_ebs.append(
                    ax.errorbar(
                        self.bin_centers,
                        self.bin_content,
                        yerr=np.sqrt((self.yerr(False)[:, :iuncertainty]**2).sum(axis=1)),
                        # xerr=xerr,
                        **kwargs_copy
                    )
                )
                # split_unc_ebs[-1][2][0].set_linestyle(error_linestyle)
                # split_unc_ebs[-1][0].set_linestyle(error_linestyle)

                kwargs_copy["color"] = split_unc_ebs[-1][0].get_color()
                kwargs["color"] = kwargs_copy["color"]

        if split_uncertainty:
            kwargs["label"] += (
                f" (unc. {'+'.join(self.unc_sources).replace('_',' ')})"
                if len(self.unc_sources) > 0
                else ""
            )
        else:
            if uncertainty_index >= 0:
                kwargs["label"] += (
                    f" (unc. {self.unc_sources[uncertainty_index].replace('_',' ')})"
                )

        eb = ax.errorbar(
            self.bin_centers,
            self.bin_content,
            yerr=self.yerr(sum_sources=uncertainty_index < 0, source_index=uncertainty_index),
            xerr=xerr,
            # label=label,
            capsize=5,
            **kwargs,
        )
        if split_uncertainty:
            eb[2][0].set_linestyle(error_linestyle)
            eb[2][1].set_linestyle(error_linestyle)
        try:
            self.save_color(eb[0].get_color())
        except RuntimeError as e:
            print(e)
            self.save_color(None)

    def save_color(self, color: Union[str, np.ndarray], alpha: float = -1.0) -> None:
        if self.color is None:
            self.color = mpl.colors.to_rgba_array(color).flatten()
            if alpha > 0:
                self.color[-1] = alpha

    def plot_uncertainty_band(self, ax=None, uncertainty_index=-1, extend: bool = True, **kwargs) -> None:
        if "color" in kwargs:
            self.save_color(kwargs["color"])
        else:
            if self.color is None:
                kwargs["color"] = mpl.colors.to_rgba_array(ax._get_lines.get_next_color())
                self.save_color(kwargs["color"])
            else:
                kwargs["color"] = deepcopy(self.color)

        hatch_facecolor = mpl.colors.to_rgba_array(kwargs["color"]).flatten()
        hatch_facecolor[-1] = min(
            1.0, kwargs.get("alpha", hatch_facecolor[-1]) * 1.5
        )  # setting alpha to 1 for hatch facecolor
        if "linewidth" not in kwargs:
            kwargs["linewidth"] = 0.5

        hatch_kwargs = deepcopy(kwargs)
        del kwargs["hatch"]
        del hatch_kwargs["label"]
        if "alpha" in hatch_kwargs:
            del hatch_kwargs["alpha"]
        hatch_kwargs["edgecolor"] = kwargs["color"]
        hatch_kwargs["color"] = "None"
        # hatch_kwargs["edgecolor"] = hatch_facecolor

        uncertainty = (
            self.yerr() if uncertainty_index < 0 else self.yerr(sum_sources=False, source_index=uncertainty_index)
        )

        # hatch_kwargs["label"] += (
        #     " (total unc.)"
        #     if uncertainty_index < 0
        #     else f" (unc. {'+'.join(self.unc_sources[:uncertainty_index+1]).replace('_',' ')})"
        # )
        kwargs["label"] += (
            " (unc. total)"
            if uncertainty_index < 0
            else f" (unc. {'+'.join(self.unc_sources[:uncertainty_index+1]).replace('_',' ')})"
        )

        lower = self.bin_content - np.abs(uncertainty[0])
        upper = self.bin_content + np.abs(uncertainty[1])
        if extend:
            for ibin in range(len(self.bin_centers)):
                # if ibin > 0 and "label" in hatch_kwargs:
                #     del hatch_kwargs["label"]
                if ibin > 0 and "label" in kwargs:
                    del kwargs["label"]
                mpl.rcParams['hatch.linewidth'] = kwargs["linewidth"]
                ax.fill_between(
                    self.edges[ibin:ibin+2],
                    lower[ibin],
                    upper[ibin],
                    **kwargs
                )
                ax.fill_between(
                    self.edges[ibin:ibin+2],
                    lower[ibin],
                    upper[ibin],
                    **hatch_kwargs
                )
                ax.plot(
                    self.edges[ibin:ibin+2],
                    [self.bin_content[ibin], self.bin_content[ibin]],
                    color=kwargs["color"],
                    linestyle="--",
                    linewidth=kwargs["linewidth"]
                )
        else:
            ax.fill_between(self.bin_centers, lower, upper, **kwargs)
            ax.fill_between(self.bin_centers, lower, upper, **hatch_kwargs)
            del kwargs["label"]
            kwargs["color"] = hatch_facecolor
            ax.plot(self.bin_centers, self.bin_content, "--", **kwargs)


def setup_ax(w: int = 10, h: int = 7, fontsize: float = 20.0) -> Tuple[plt.Figure, plt.Axes]:
    font_size = fontsize
    mpl.rcParams["axes.labelsize"] = font_size
    f, ax = plt.subplots(figsize=(w, h))

    return f, ax


def finalize_ax(
    ax: plt.Axes, font_size: float = 20.0, fname: str = None, reversered_legend: bool = True, year: str = ""
) -> None:
    f = ax.get_figure()
    # extra_text = ", private work"
    if year != "":
        hep.label.exp_label(llabel="Private work (CMS data/simulation)", ax=ax, fontsize=font_size, year=year)
        # hep.cms.label(label=extra_text, ax=ax, fontsize=font_size, year=year)
    else:
        # hep.cms.text(extra_text, ax=ax, fontsize=font_size)
        hep.label.exp_label(llabel="Private work (CMS data/simulation)", ax=ax, fontsize=font_size)

    handles, labels = ax.get_legend_handles_labels()
    if reversered_legend:
        ax.legend(reversed(handles), reversed(labels), fontsize=font_size - 6, loc="upper left")
    else:
        ax.legend(fontsize=font_size - 6, loc="upper left")
    plt.xticks(fontsize=font_size - 2)
    plt.yticks(fontsize=font_size - 2)
    ax.plot(ax.get_xlim(), [1] * 2, "k--", alpha=0.6)

    ax.set_ylim(0.95, 1.05)
    ax.set_xlabel(r"$p_{T}$ [GeV]")
    ax.set_ylabel("JMS-SF", loc="center")

    if fname:
        if not os.path.exists(os.path.dirname(fname)):
            os.makedirs(os.path.dirname(fname))
        f.savefig(fname, bbox_inches="tight")
    else:
        f.tight_layout()
    # f.show()


def create_plotter(fit_result_path: str, years: List, regions: List, JEC: bool = False, legacy: bool = False):
    fitResults = json.load(open(fit_result_path))
    fits = [f"{region}{year}" for region in regions for year in years]
    plotter = JMSPlotter(fitResults, fits, legacy_json_format=legacy)
    plotter.construct_hists()
    if JEC:
        plotter.load_JEC_fits(fitResults)
    return plotter


if __name__ == "__main__":
    years = ["UL16preVFP", "UL16postVFP", "UL17", "UL18"]
    regions = ["TTBar", "VJets", "Combined"]
    plotter = {}
    plotter["07_03_23"] = create_plotter("fitResults_07-03-23.json", years, regions)
    plotter["06_03_23"] = create_plotter("fitResults_06-03-23.json", years, regions)

    plotter["06_03_23"].load_fit_results(
        json.load(open("fitResults_06-03-23.json")),
        [f"{region}{year}JEC{jecdir}" for year in years for jecdir in ["UP", "DOWN"] for region in regions],
    )
    plotter["06_03_23"].construct_hists()
    label_suffix = {
        "07_03_23": "JEC Var nuisance",
        "06_03_23": "JEC nominal",
    }

    for region in regions:
        for year in years:
            f, ax = setup_ax(10, 7)
            for date, jms_plotter in plotter.items():
                jms_plotter.dHists[f"{region}{year}"].plot_errorbar(
                    ax=ax, split_uncertainty=True, alpha=0.8, label=f"{region} {year} {label_suffix[date]}", fmt=".",
                    linewidth=0.9,
                    linestyle="--" if date == "06_03_23" else "-"
                )
                if date == "06_03_23":
                    nominal_color = jms_plotter.dHists[f"{region}{year}"].color
                    # jms_plotter.dHists[f"{region}{year}JECDOWN"].plot_errorbar(
                    #     ax,
                    #     fmt=".",
                    #     color=nominal_color - np.array([0, 0.1, 0.0, 0.0]),
                    #     alpha=0.6,
                    #     label=f"{region} {year} JEC down",
                    # )
                    # jms_plotter.dHists[f"{region}{year}JECUP"].plot_errorbar(
                    #     ax,
                    #     fmt=".",
                    #     color=nominal_color + np.array([0, 0.1, 0.0, 0.0]),
                    #     alpha=0.6,
                    #     label=f"{region} {year} JEC up",
                    # )
                    extend = True
                    jms_plotter.dHists[f"{region}{year}JECDOWN"].plot_uncertainty_band(
                        ax,
                        extend=extend,
                        hatch="/",
                        alpha=0.4,
                        linewidth=.6,
                        # color=nominal_color - np.array([0, 0.1, 0.0, 0.0]),
                        label=f"{region} {year} JEC down",
                    )
                    var_color = jms_plotter.dHists[f"{region}{year}JECDOWN"].color

                    jms_plotter.dHists[f"{region}{year}JECUP"].plot_uncertainty_band(
                        ax,
                        extend=extend,
                        hatch="\\",
                        linewidth=.6,
                        alpha=0.4,
                        # color=var_color + np.array([0, 0.1, 0.0, 0.0]),
                        label=f"{region} {year} JEC up",
                    )

            finalize_ax(ax, fname=f"JMSSF_JERC_03/{region}_{year}_comparison.pdf", year=year)
            # plt.gca().set_prop_cycle(None)

    plotter["07_03_23"] = create_plotter("fitResults_07-03-23.json", years, regions)
    plotter["06_03_23"] = create_plotter("fitResults_06-03-23.json", years, regions)
    for date, jms_plotter in plotter.items():
        for region in regions:
            f, ax = setup_ax(10, 7)
            for year in years:
                jms_plotter.dHists[f"{region}{year}"].plot_errorbar(
                    ax=ax,
                    split_uncertainty=True,
                    alpha=0.8,
                    label=f"{region} {year}",
                    linewidth=0.9,
                    fmt=".",
                )
            finalize_ax(ax, fname=f"JMSSF_07-03-23/{date}_{region}_year_comparison.pdf")

    suffix_labels = {
        "NOJEC": "(noJEC on mass)",
        "ENDCAP": r"$1.3<|\eta|<2.4$",
        "BARREL": r"$|\eta|<1.3$",
    }
    for suffix in ["NOJEC", "BARREL", "ENDCAP"]:

        fitResults_nojec = json.load(open("fitResults_14-03-23.json"))
        plotter = JMSPlotter(fitResults_nojec, [f"Combined{year}{suffix}" for year in years], legacy_json_format=False)
        plotter.construct_hists()
        f, ax = setup_ax(10, 7)
        for year in years:
            plotter.dHists[f"Combined{year}{suffix}"].plot_errorbar(
                ax=ax,
                split_uncertainty=False,
                alpha=0.8,
                label=f"Combined {year} {suffix_labels[suffix]}",
                linewidth=0.9,
                fmt=".",
            )
        finalize_ax(ax, fname=f"JMSSF_JERC_03/Combiend_year_comparison_{suffix}.pdf")
        for year in years:
            f, ax = setup_ax(10, 7)
            plotter.dHists[f"Combined{year}{suffix}"].plot_errorbar(
                ax=ax,
                split_uncertainty=False,
                alpha=0.8,
                label=f"Combined {year} {suffix_labels[suffix]}",
                linewidth=0.9,
                fmt=".",
            )
            finalize_ax(ax, fname=f"JMSSF_JERC_03/Combiend_{year}_comparison_{suffix}.pdf",year=year)
