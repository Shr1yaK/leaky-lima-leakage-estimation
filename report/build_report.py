"""
Deliverable 10: the typed report.

Every number and table in the report is read from results/ when this script
runs, so the report can never drift from the code.  Figures come from
figures/ plus a few report-only plots written to report/fig/.

    .venv/bin/python report/build_report.py         -> report/report.html + report/Leaky_Lima_Report.pdf
"""
import html as H
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import config  # noqa: E402
from leakml import circuits  # noqa: E402

R, F = config.RESULTS_DIR, config.FIG_DIR
FIG = HERE / "fig"
FIG.mkdir(exist_ok=True)
PDF = HERE / "Leaky_Lima_Report.pdf"
_MAC_CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
CHROME = (os.environ.get("CHROME")
          or shutil.which("google-chrome") or shutil.which("chromium") or shutil.which("chrome")
          or (_MAC_CHROME if Path(_MAC_CHROME).exists() else None)
          or next((str(p) for p in Path.home().glob("Library/Caches/ms-playwright/chromium-*/chrome-mac*/*.app/Contents/MacOS/*")), None))
CUR = ["id", "ig", "is", "ib"]
FAM = {"mlp": "MLP (neural net)", "poly_ridge": "Polynomial + ridge",
       "lightgbm": "LightGBM", "random_forest": "Random forest"}
CASE = {"nmos_on": "NMOS ON", "nmos_off": "NMOS OFF", "pmos_on": "PMOS ON", "pmos_off": "PMOS OFF"}


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------
def sci(x, d=2):
    m, e = f"{x:.{d}e}".split("e")
    return f"{m}&times;10<sup>{int(e)}</sup>"


def p(x, d=2):
    return f"{x:.{d}f}"


def table(df, num=None, caption=None, best=None, cls=""):
    num = set(df.columns[1:] if num is None else num)
    h = "".join(f'<th class="{"n" if c in num else ""}">{c}</th>' for c in df.columns)
    rows = []
    for i, (_, r) in enumerate(df.iterrows()):
        b = ' class="best"' if best is not None and best(i, r) else ""
        rows.append(f"<tr{b}>" + "".join(
            f'<td class="{"n" if c in num else ""}">{r[c]}</td>' for c in df.columns) + "</tr>")
    cap = f"<caption>{caption}</caption>" if caption else ""
    return f'<table class="{cls}">{cap}<thead><tr>{h}</tr></thead><tbody>{"".join(rows)}</tbody></table>'


N_FIG = [0]


def fig(path, caption, width=100):
    N_FIG[0] += 1
    rel = os.path.relpath(path, HERE)
    return (f'<figure style="width:{width}%"><img src="{rel}">'
            f'<figcaption><b>Figure {N_FIG[0]}.</b> {caption}</figcaption></figure>')


def log_value(name, pattern, default="?"):
    try:
        m = re.search(pattern, (R / name).read_text())
        return m.group(1) if m else default
    except FileNotFoundError:
        return default


def mean_acc(true, pred):
    return 100 - 100 * np.mean(np.abs(np.asarray(pred) / np.asarray(true) - 1))


# --------------------------------------------------------------------------
# report-only figures
# --------------------------------------------------------------------------
def fig_model_comparison(mc):
    fig_, ax = plt.subplots(1, 4, figsize=(13, 3.2), sharey=True)
    fams = list(FAM)
    for a, case in zip(ax, CASE):
        d = mc[mc.case == case].set_index("model")
        x = np.arange(len(CUR))
        for k, fam in enumerate(fams):
            if fam in d.index:
                a.bar(x + (k - 1.5) * 0.2, [d.loc[fam, f"{c}_med%"] for c in CUR], 0.2,
                      label=FAM[fam])
        a.set_yscale("log")
        a.set_xticks(x, CUR)
        a.axhline(1, c="k", lw=0.6, ls=":")
        a.set_title(CASE[case])
    ax[0].set_ylabel("validation median error (%)")
    ax[-1].legend(fontsize=7)
    fig_.tight_layout()
    out = FIG / "model_comparison.png"
    fig_.savefig(out, dpi=140)
    plt.close(fig_)
    return out


def fig_learning_curve(lc):
    fig_, ax = plt.subplots(1, 4, figsize=(13, 3.2), sharey=True)
    for a, case in zip(ax, CASE):
        for k, c in enumerate(CUR):
            g = lc[(lc.case == case) & (lc.current == c)].sort_values("n_train")
            a.loglog(g.n_train, g.med_pct, "o-", c=f"C{k}", ms=4, label=f"{c} median")
            a.loglog(g.n_train, g.p95_pct, "x--", c=f"C{k}", ms=4, alpha=0.5, label=f"{c} p95")
        a.axhline(1, c="k", lw=0.6, ls=":")
        ns = sorted(lc.n_train.unique())
        a.set_xticks(ns, [str(n) for n in ns], fontsize=7)
        a.minorticks_off()
        a.set(title=CASE[case], xlabel="uniform training samples")
    ax[0].set_ylabel("validation error (%)")
    ax[-1].legend(fontsize=6, ncol=2)
    fig_.tight_layout()
    out = FIG / "learning_curve.png"
    fig_.savefig(out, dpi=140)
    plt.close(fig_)
    return out


def fig_gate_error_vs_corner(gs):
    fig_, ax = plt.subplots(1, 3, figsize=(13, 3.3), sharey=True)
    for a, (col, lab) in zip(ax, [("temp", "temperature (C)"), ("vdd", "VDD (V)"),
                                  ("pmos_vth0", "PMOS vth0 (V)")]):
        b = pd.qcut(gs[col], 10)
        g = gs.groupby(b, observed=True)
        x = g[col].median()
        a.fill_between(x, g.e.quantile(0.05), g.e.quantile(0.95), alpha=0.25, label="5-95 %")
        a.plot(x, g.e.median(), "o-", label="median")
        a.axhline(0, c="k", lw=0.6)
        a.set_xlabel(lab)
    ax[0].set_ylabel("signed gate error ML/SPICE - 1 (%)")
    ax[0].legend(fontsize=8)
    fig_.suptitle("Gate error vs corner conditions (3,600 sims, deciles)", fontsize=10)
    fig_.tight_layout()
    out = FIG / "gate_error_vs_corner.png"
    fig_.savefig(out, dpi=140)
    plt.close(fig_)
    return out


def fig_cla_vs_temp(cs):
    fig_, ax = plt.subplots(figsize=(6.6, 3.6))
    b = pd.qcut(cs.temp, 10)
    for col, lab in [("ml_gate_sum", "hierarchical ML (ideal 0/VDD levels)"),
                     ("spice_gate_sum", "SPICE gate sum (hierarchy alone)"),
                     ("ml_flat", "flat ML on circuit voltages"),
                     ("ml_levels", "levels-aware ML (SPICE-free)")]:
        e = 100 * (cs[col] / cs.spice_leak - 1)
        g = e.groupby(b, observed=True)
        ax.plot(cs.temp.groupby(b, observed=True).median(), g.median(), "o-", ms=4, label=lab)
    ax.axhline(0, c="k", lw=0.6)
    ax.set(xlabel="temperature (C)", ylabel="median signed error vs SPICE circuit (%)",
           title="CLA: where the hierarchy error comes from")
    ax.legend(fontsize=7)
    fig_.tight_layout()
    out = FIG / "cla_error_vs_temp.png"
    fig_.savefig(out, dpi=140)
    plt.close(fig_)
    return out


def fig_level_shift(gs):
    """How far a gate's logic-0 / logic-1 output sits from the rail (SPICE)."""
    fig_, ax = plt.subplots(figsize=(6.6, 3.4))
    for lv, lab in [(0, "output = 0: V(out)"), (1, "output = 1: VDD - V(out)")]:
        rows = []
        for (gate, vec), g in gs.groupby(["gate", "vector"]):
            bits = dict(zip(circuits.GATE_INPUTS[gate], map(int, vec)))
            cells = circuits.gate_cells(gate, circuits.GATE_INPUTS[gate], "out", "g")
            if circuits.evaluate_logic(cells, bits)["out"] != lv:
                continue
            dv = g.out if lv == 0 else g.vdd - g.out
            rows.append(pd.DataFrame({"temp": g.temp, "dv": 1e3 * dv}))
        d = pd.concat(rows)
        b = pd.qcut(d.temp, 10)
        ax.semilogy(d.temp.groupby(b, observed=True).median(), d.dv.groupby(b, observed=True).median(),
                    "o-", ms=4, label=lab)
    ax.set(xlabel="temperature (C)", ylabel="distance from the rail (mV)",
           title="Gate outputs are not at exactly 0 / VDD (SPICE, isolated gates)")
    ax.legend(fontsize=8)
    fig_.tight_layout()
    out = FIG / "level_shift.png"
    fig_.savefig(out, dpi=140)
    plt.close(fig_)
    return out


# --------------------------------------------------------------------------
def build():
    env = (R / "env.txt").read_text()
    envd = dict(re.findall(r"^(.+?)\s*:\s*(.+)$", env, re.M))
    corners = pd.read_csv(R / "corners.csv", index_col="corner")
    mc = pd.read_csv(R / "model_comparison.csv")
    allhp = pd.read_csv(R / "model_comparison_all_hparams.csv") if (R / "model_comparison_all_hparams.csv").exists() else None
    ch = pd.read_csv(R / "chosen_models_test.csv")
    kc = pd.read_csv(R / "kirchhoff_check.csv")
    tn = pd.read_csv(R / "trend_numbers.csv", index_col=0).iloc[:, 0]
    gr = pd.read_csv(R / "gate_results.csv", dtype={"inputs": str})
    gs = pd.read_csv(R / "gate_samples.csv", dtype={"vector": str}).join(corners, on="corner")
    gs["e"] = 100 * (gs.ml_leak / gs.spice_leak - 1)
    gs["ae"] = gs.e.abs()
    csum = pd.read_csv(R / "circuit_summary.csv")
    cres = pd.read_csv(R / "circuit_results.csv", dtype={"inputs": str})
    cs = pd.read_csv(R / "circuit_samples.csv", dtype={"vector": str}).join(corners, on="corner")
    lstats = pd.read_csv(R / "circuit_levels_solver_stats.csv")
    abl = pd.read_csv(R / "ablation.csv")
    lc = pd.read_csv(R / "learning_curve.csv")

    gate_acc = mean_acc(gs.spice_leak, gs.ml_leak)
    gate_acc_solved = mean_acc(gs.spice_leak, gs.ml_solved_leak)
    rel = np.abs(gs.ml_leak / gs.spice_leak - 1) * 100
    rel_s = np.abs(gs.ml_solved_leak / gs.spice_leak - 1) * 100
    acc = {k: mean_acc(cs.spice_leak, cs[k]) for k in ["ml_gate_sum", "ml_flat", "ml_levels"]}
    acc_mlonly = mean_acc(cs.spice_gate_sum, cs.ml_gate_sum)
    hier_err = csum.iloc[1]
    lev = csum[csum.comparison.str.contains("levels")].iloc[0]
    hml = csum.iloc[0]
    mlerr = csum.iloc[2]
    flat = csum.iloc[3]
    node_cols = [c for c in gs.columns if c.startswith("ml_") and c[3:] in gs.columns
                 and c not in ("ml_leak", "ml_ivdd")]
    node_dv = np.concatenate([np.abs(gs[c] - gs[c[3:]]).dropna().to_numpy() for c in node_cols]) * 1e3
    spice_gate_s = sum(float(x) for x in re.findall(r"sims, ([0-9.]+)s", (R / "gate_log.txt").read_text()))
    spice_cla_s = float(log_value("circuit_log.txt", r"simulations in ([0-9.]+)s", "nan"))
    lev_s = float(log_value("circuit_log.txt", r"levels-aware ML: ([0-9.]+)s", "nan"))
    hier_s = float(log_value("circuit_log.txt", r"predictions: ([0-9.]+)s", "nan"))
    extras = (R / "extras_summary.txt").read_text() if (R / "extras_summary.txt").exists() else ""

    # ---- values quoted in the text (all computed from results/ and data/) ----
    ivdd_lo, ivdd_hi = gr.ivdd_med_err_pct.min(), gr.ivdd_med_err_pct.max()
    u_off = pd.read_csv(config.DATA_DIR / "nmos_off.csv")
    n_rail_uniform = int(((u_off.vs < 0.02 * u_off.vdd) & (u_off.vd > 0.98 * u_off.vdd)).sum())
    dep_med = {c: pd.read_csv(config.DATA_DIR / f"{c}_deploy.csv")[["id", "ib"]].abs().median()
               for c in ("nmos_off", "pmos_off")}
    chi = ch.set_index("case")
    cmp_rows = []
    for case_ in CASE:
        g_ = mc[mc.case == case_].set_index("model")
        m_, o_ = g_.loc["mlp"], g_.drop("mlp")
        for c_ in CUR:
            cmp_rows.append(dict(case=case_, cur=c_, med=o_[f"{c_}_med%"].min() / m_[f"{c_}_med%"],
                                 p95=o_[f"{c_}_p95%"].min() / m_[f"{c_}_p95%"],
                                 best_med=o_[f"{c_}_med%"].idxmin()))
    cmp = pd.DataFrame(cmp_rows)
    cmp_better = cmp[cmp["med"] < 1]
    if len(cmp_better):
        cmp_text = (f"The MLP has a higher median error than another family in {len(cmp_better)} of the 16 combinations: "
                    + "; ".join(f"{CASE[r.case]} I<sub>{r.cur[1]}</sub> ({FAM[r.best_med]}, ratio {r.med:.2f})"
                                for r in cmp_better.itertuples()) + ".")
        for case_ in cmp_better.case.unique():
            top_ = cmp[cmp.case == case_].sort_values("med").iloc[-1]
            cmp_text += (f" In {CASE[case_]}, the largest median margin of the MLP is on "
                         f"I<sub>{top_.cur[1]}</sub> ({top_.med:.1f}&times;).")
    else:
        cmp_text = "The MLP has the lowest median error in all 16 combinations."
    poly_ib = mc[mc.model == "poly_ridge"]["ib_med%"]
    n64 = int(ch.hparams.str.contains(r"\(64,").sum())
    test_med_max = ch[[f"{c}_med%" for c in CUR]].to_numpy().max()
    dep_idis = chi[["deploy_id_med%", "deploy_is_med%"]].max(axis=1)
    dep_idis_max, dep_idis_arg, n_dep_below1 = dep_idis.max(), dep_idis.idxmax(), int((dep_idis < 1).sum())
    sm = gr.set_index(["gate", "inputs"]).spice_median
    top4 = gr.sort_values("p95_err_pct", ascending=False).head(4)
    p95_lo, p95_hi = gr.p95_err_pct.min(), gr.p95_err_pct.max()
    pcols = [c for c in corners.columns if c not in ("vdd", "temp")]
    r_t, r_v = gs.ae.corr(gs.temp), gs.ae.corr(gs.vdd)
    r_proc = max(abs(gs.e.corr(gs[c])) for c in pcols)
    r_proc_abs = max(abs(gs.ae.corr(gs[c])) for c in pcols)
    wc = max(((r.case, c, r[f"{c}_p95%"]) for _, r in ch.iterrows() for c in CUR), key=lambda t: t[2])
    train_s = log_value("train_log.txt", r"done in ([0-9.]+)s")
    n_sub = log_value("train_log.txt", r"training (\d+) sub-models")
    _sf = R / "extras_sensitivity.csv"
    sv = pd.read_csv(_sf).set_index("case").ratio_to_nominal if _sf.exists() else None
    sv_small = (100 * max(abs(v - 1) for k, v in sv.items() if any(x in k for x in ("toxe", "u0", "xj")))
                if sv is not None else float("nan"))
    _mf = R / "extras_mechanisms.csv"
    mech0 = pd.read_csv(_mf, dtype={"vector": str}) if _mf.exists() else None
    if mech0 is not None:
        _t = mech0.temp.unique()[np.argmin(np.abs(mech0.temp.unique() - 25))]
        _m25 = mech0[mech0.temp == _t]
        tun_cold = 100 * mech0[mech0.temp == mech0.temp.min()].tunnelling_frac.mean()
        tun_hot = 100 * mech0[mech0.temp == mech0.temp.max()].tunnelling_frac.mean()
        tun_stack = [100 * _m25[(_m25.gate == g) & (_m25.vector == v)].tunnelling_frac.iloc[0]
                     for g, v in (("NAND2", "00"), ("NOR2", "11"))]
    else:
        tun_cold = tun_hot = float("nan")
        tun_stack = [float("nan")] * 2
    platform = H.escape(envd.get("platform", "?"))
    cores = envd.get("cpu cores used", "?")

    S = []   # html sections

    # ---------------------------------------------------------------- title
    S.append(f"""
<header>
<p class="course">Digital VLSI Design: Course Project</p>
<h1>ML-Driven PVT-Aware Leakage Estimation</h1>
<p class="sub">Team 1 &ldquo;Leaky Lima&rdquo; &middot; {config.TECH.replace('_', ' ')} PTM &middot;
VDD<sub>nom</sub> = {config.VDD_NOM:.1f} V &middot; 3-bit carry-lookahead adder</p>
<p class="sub">Samyak Soni (2024112004) &middot; Satvik Shrivastava (2024102029)</p>
<p class="sub">Shriya Kansal (2024102075) &middot; Udhav Malpani (2024112009)</p>
<p class="sub">Repository: <a href="https://github.com/Shr1yaK/leaky-lima-leakage-estimation/">https://github.com/Shr1yaK/leaky-lima-leakage-estimation/</a></p>
</header>
<div class="abstract"><b>Summary.</b>
We trained four neural-network models (NMOS and PMOS, ON and OFF) that estimate the four terminal
currents of a 16&nbsp;nm transistor under process, voltage and temperature variation. We combined them into
the five library gates and compared the estimated leakage with ngspice for 3,600 gate simulations. The
<b>mean accuracy is {p(gate_acc)}&nbsp;%</b> (median error {p(np.median(rel))}&nbsp;%, 95th percentile
{p(np.quantile(rel, .95))}&nbsp;%). When the same models solve the internal node voltages, with no SPICE voltage
as input, the mean accuracy is {p(gate_acc_solved)}&nbsp;%.
For the 204-transistor 3-bit carry-lookahead adder (CLA, 25,600 ngspice simulations), the hierarchical sum
that the brief specifies has a mean accuracy of {p(acc['ml_gate_sum'])}&nbsp;% (median error
{p(hml.med_err_pct)}&nbsp;%, bias {hml.bias_pct:+.2f}&nbsp;%). The sum of the ngspice gate leakages has a bias of
{hier_err.bias_pct:+.2f}&nbsp;% against the ngspice circuit, so most of this error comes from the hierarchy. The ML
models add a median error of {p(mlerr.med_err_pct)}&nbsp;%.
In an extension (Section 7.2), each gate receives the estimated output voltage of its driver instead of an
ideal 0 or VDD. The CLA <b>mean accuracy is then {p(acc['ml_levels'])}&nbsp;%</b> (median error
{p(lev.med_err_pct)}&nbsp;%). Section 7.9 applies both hierarchies to the 8-bit parity checker without retraining.
</div>
""")

    # ---------------------------------------------------------------- 1 setup
    S.append(f"""
<h2>1&nbsp; Setup</h2>
{table(pd.DataFrame([
    ["Model file", f"{config.TECH}.pm (PTM, BSIM4 level 54)"],
    ["Nominal VDD", f"{config.VDD_NOM:.1f} V (sampled &plusmn;10 %: {config.VDD_RANGE[0]:.2f}&ndash;{config.VDD_RANGE[1]:.2f} V)"],
    ["Channel length", f"L = {config.LMIN*1e9:.0f} nm; W = 1&ndash;8 &times; L = {config.W_MIN*1e9:.0f}&ndash;{config.W_MAX*1e9:.0f} nm"],
    ["Gate sizing", f"W<sub>n</sub> = 2L = {config.WN_GATE*1e9:.0f} nm, W<sub>p</sub> = 4L = {config.WP_GATE*1e9:.0f} nm (interior of the training range)"],
    ["Test circuit", "3-bit carry-lookahead adder: 42 library gates, 204 transistors, 7 inputs (128 vectors)"],
    ["Simulator", H.escape(envd.get("simulator", "?"))],
    ["Software", f"Python {envd.get('python','?')}, scikit-learn {envd.get('scikit-learn','?')}, LightGBM {envd.get('lightgbm','?')}, numpy {envd.get('numpy','?')}, pandas {envd.get('pandas','?')}"],
    ["Platform", H.escape(envd.get("platform", "?")) + f", {envd.get('cpu cores used','?')} cores"],
    ["SPICE options", f"<code>{config.SPICE_OPTIONS}</code> in every netlist"],
], columns=["Item", "Value"]), num=[])}
<p><b>Evidence labels.</b> Each claim in this report carries one of four labels.
<i>Simulated</i> values are ngspice operating-point (<code>.op</code>) results, produced with
{H.escape(envd.get('simulator', '?').split(' : ')[0])} and the options line in the table above. <i>Estimated</i> values are outputs
of the ML models and quantities computed from them, such as the leakage of a gate. <i>Measured</i> values are
run times and counts taken from the logs, recorded on {platform} with {cores} cores. <i>Assumed</i> statements are
design inputs and physical explanations that we did not test. Every error in this report compares an estimated
value with the simulated value for the same case, so we do not repeat the labels for errors.</p>
<p><b>Leakage definition.</b> We define the leakage of a gate or circuit as the total static power divided by
VDD, P<sub>static</sub>/VDD = &Sigma; V<sub>k</sub>I<sub>k</sub>/VDD, summed over the supply and the input sources.
We include the input sources because the driving gate must supply the gate-tunnelling current into an input pin,
so this current is part of the leakage of the whole circuit. With this definition, the leakages of the gates add
up to the leakage of the circuit. We also recorded the supply current I(VDD) (the <code>*_ivdd</code> columns of
<code>results/gate_samples.csv</code>). Its median error per gate case ranges from {p(ivdd_lo)} to
{p(ivdd_hi)}&nbsp;% (<code>gate_results.csv</code>). Terminal currents are positive into the device, so
I<sub>d</sub>+I<sub>g</sub>+I<sub>s</sub>+I<sub>b</sub> = 0.</p>
<p><b>Accuracy metric.</b> We define the mean accuracy of a group of simulations as 100&nbsp;% minus the mean of
|ML/SPICE &minus; 1|. We report the median, the 95th percentile and the maximum of the same error next to it.</p>
""")

    # ---------------------------------------------------------------- 2 data
    kct = kc.copy()
    kct["case"] = kct.case.map(CASE)
    kct = pd.DataFrame({"Case": kct.case, "samples": kct.n,
                        "median |&Sigma;I| (A)": [sci(x) for x in kct.abs_sum_median],
                        "max |&Sigma;I| (A)": [sci(x) for x in kct.abs_sum_max],
                        "median |&Sigma;I|/max|I|": [sci(x) for x in kct.rel_median],
                        "p95 |&Sigma;I|/max|I|": [sci(x) for x in kct.rel_p95]})
    S.append(f"""
<h2>2&nbsp; Data generation</h2>
<h3>2.1 Sampling</h3>
<p>We drew 10,000 samples for each of the four cases, with every parameter independent. VDD was drawn first,
uniform within &plusmn;10&nbsp;% of nominal. V<sub>d</sub> and V<sub>s</sub> are uniform in [0, VDD], W is uniform in
[1, 8]&times;L and T is uniform in [&minus;40, 125]&nbsp;&deg;C. Each of the five process parameters (vth0, toxe,
ndep, u0, xj) is uniform within &plusmn;10&nbsp;% of the model-card nominal, with separate draws for NMOS and PMOS.
The case fixes V<sub>g</sub> and V<sub>b</sub>. In an ON device V<sub>g</sub> is VDD for NMOS and 0 for PMOS, and in an
OFF device the reverse. The body is at the rail of the device type. All draws use fixed seeds
(<code>SEED_DATA = {config.SEED_DATA}</code>). These 10,000-row files are deliverable 1.</p>
<p>Uniform sampling rarely places a terminal on a rail: {n_rail_uniform} of the 10,000 NMOS OFF samples have
V<sub>s</sub> within 2&nbsp;% of 0 and V<sub>d</sub> within 2&nbsp;% of VDD (measured count). In a gate, the leaking
transistor has its source on a rail and its drain at the output, which is near 0 or VDD, so a model trained only
on uniform samples has little data where the gates operate. We therefore added a rail-focused supplement of
5,000 samples per case. In it, V<sub>d</sub> and V<sub>s</sub> are each exactly 0 or VDD in one third of the samples
and otherwise follow an arcsine law, which is dense near both rails. In half of its samples we moved V<sub>g</sub>
up to 10&nbsp;% of VDD away from its ideal level, because a gate driven by another gate sees a logic 0 a few tens
of mV above ground. We also drew a deployment-check set of 2,000 samples per case at the gate widths
(W<sub>n</sub> = 2L, W<sub>p</sub> = 4L) and at gate-like rail voltages, and used it only for validation and test. No
gate or circuit simulation entered training or model selection.</p>
<h3>2.2 Netlists</h3>
<p>The single-transistor netlist of each case (<code>netlists/single_*.cir</code>) holds one transistor with a
separate voltage source on every terminal, so ngspice reports all four terminal currents directly. To avoid the
start-up time of ngspice for every sample, one process runs a batch of samples. Between <code>.op</code> analyses it
changes the process parameters with <code>altermod</code> and the width, voltages and temperature with
<code>alter</code>. A self-check re-simulates samples as standalone netlists and reproduces the batch results
(development log, <code>CHANGES.md</code>). We also recorded the internal BSIM4 current components
(I<sub>ds</sub>, I<sub>gs</sub>, I<sub>gd</sub>, I<sub>gcs</sub>, I<sub>gcd</sub>, I<sub>gb</sub>, GIDL and GISL,
I<sub>sub</sub> and the junction currents) and regrouped them into six physical <b>bipoles</b> (ds, gs, gd, gb, db,
sb). The bipole currents rebuild the four terminal currents exactly. We generated the gate and circuit netlists
from the same transistor list that the ML side uses, so SPICE and ML share one topology. Each gate and circuit
simulation starts from a logic-level <code>.nodeset</code>; Section 6 gives the reason.</p>
<h3>2.3 Splits</h3>
<p>We split each 10,000-row dataset 70/15/15 (7,000 training, 1,500 validation and 1,500 test samples) with
<code>SEED_SPLIT = {config.SEED_SPLIT}</code>. We split the rail supplement in the same way and added its training
part to the training data. The test parts were used once, after model selection. The split indices are in
<code>results/split_*.npz</code>.</p>
<h3>2.4 Data check</h3>
{table(kct, caption="Table 1. Kirchhoff check, |I<sub>d</sub>+I<sub>g</sub>+I<sub>s</sub>+I<sub>b</sub>| of the simulated terminal currents for each case (10,000 samples, ngspice).")}
<p>For NMOS ON, NMOS OFF and PMOS ON, the median ratio |&Sigma;I| / max|I| is at most
{sci(kc[kc.case != 'pmos_off'].rel_median.max())} and the 95th percentile is at most
{sci(kc[kc.case != 'pmos_off'].rel_p95.max())} (simulated). For PMOS OFF, the median ratio is
{sci(kc.set_index('case').loc['pmos_off','rel_median'])}, the 95th percentile
{sci(kc.set_index('case').loc['pmos_off','rel_p95'])} and the maximum
{p(100*kc.set_index('case').loc['pmos_off','rel_max'],1)}&nbsp;%. In the PMOS OFF samples with the largest ratios,
every terminal current is near 10<sup>&minus;15</sup>&nbsp;A, and the residual is limited by the solver tolerances
(assumed: abstol = 10<sup>&minus;18</sup>&nbsp;A, amplified by the convergence check). The largest absolute residual in
any case is {sci(kc.abs_sum_max.max())}&nbsp;A.</p>
{fig(F / 'data_check_trends.png', f"Trend checks on the simulated data. The OFF current is {p(tn['ioff_ratio_125C_vs_25C'])}&times; larger at 125&nbsp;&deg;C than at 25&nbsp;&deg;C. It falls to {p(tn['ioff_ratio_vth0_plus10pct'])}&times; of its value for +10&nbsp;% vth0, and the gate current falls to {p(tn['ig_ratio_toxe_plus10pct'])}&times; for +10&nbsp;% toxe. These directions agree with the physical models of leakage (assumed): sub-threshold current depends exponentially on V<sub>th</sub>, which falls with temperature, and direct tunnelling current depends exponentially on oxide thickness.")}
""")

    # ---------------------------------------------------------------- 3 models
    mct = mc.copy()
    mct["Case"] = mct.case.map(CASE)
    mct["Model"] = mct.model.map(FAM)
    best = mct.groupby("case").score.transform("min") == mct.score
    rows = [mct.Case, mct.Model.str.replace(" (neural net)", ""), mct.hparams.str.replace(r"hidden=\((\d+), \d+, \d+\),n_seeds=(\d+)", r"3&times;\1, \2 seeds", regex=True).str.replace("n_estimators=", "trees=").str.replace("num_leaves=", "leaves=")]
    cols = ["Case", "Model", "hyper-params"]
    for c in CUR:
        rows += [mct[f"{c}_r2"].map(lambda x: f"{x:.4f}"), mct[f"{c}_med%"].map(p), mct[f"{c}_p95%"].map(p)]
        cols += [f"{c} R&sup2;", f"{c} med %", f"{c} p95 %"]
    mct_t = pd.DataFrame(dict(zip(cols, rows)))
    cht = pd.DataFrame({"Case": ch.case.map(CASE), "Model": ch.model.map(FAM).str.replace(" (neural net)", ""),
                        **{f"{c} med %": ch[f"{c}_med%"].map(p) for c in CUR},
                        **{f"{c} p95 %": ch[f"{c}_p95%"].map(p) for c in CUR}})
    dep = pd.DataFrame({"Case": ch.case.map(CASE),
                        **{f"{c} med %": ch[f"deploy_{c}_med%"].map(p) for c in CUR},
                        **{f"{c} p95 %": ch[f"deploy_{c}_p95%"].map(p) for c in CUR}})
    ab = abl.copy()
    abt = pd.DataFrame({"Case": ab.case.map(CASE), "current": ab.current,
                        "direct med %": ab.direct_med_pct.map(p), "direct p95 %": ab.direct_p95_pct.map(p),
                        "bipole med %": ab.bipole_med_pct.map(p), "bipole p95 %": ab.bipole_p95_pct.map(p)})
    ab_ratio = np.median(ab.direct_med_pct / ab.bipole_med_pct)
    piv = lc.pivot_table(index=["case", "current"], columns="n_train", values="med_pct")
    nmax = piv.columns.max()
    # smallest n whose median error is within 1.25x of the full-size model, for every current
    ok = piv.le(1.25 * piv[nmax].to_numpy()[:, None] + 0.05)
    n_enough = {case: int(min(n for n in piv.columns if ok.loc[case][n].all())) for case in CASE}
    piv_t = piv.copy()
    piv_t.columns = [f"n = {c}" for c in piv_t.columns]
    piv_t = piv_t.map(p).reset_index()
    piv_t.insert(0, "Case", piv_t.pop("case").map(CASE))

    S.append(f"""
<h2>3&nbsp; Models</h2>
<h3>3.1 What is learned</h3>
<p>The terminal currents change sign and span many orders of magnitude, and I<sub>s</sub> is often the difference of
two much larger currents of opposite sign (simulated data). We assume that this makes a direct regression of the
terminal currents inaccurate, because a model of log|I| cannot follow a sign change or a small difference of
large terms; Section 3.4 measures the resulting error. We therefore split each transistor into six bipoles. Each
bipole is a passive two-terminal element, and its current flows from its higher to its lower terminal. For each
bipole the model learns the conductance G = I/V, in the form log G per unit width, and the terminal currents
follow from KCL: I<sub>d</sub> = I<sub>ds</sub> &minus; I<sub>gd</sub> + I<sub>db</sub>, and so on. As a result, KCL holds
exactly, every bipole current has the correct sign, and I = 0 at V = 0. Each case has five sub-models with shared
inputs: channel (ds), gate-to-diffusion tunnelling (gs and gd, one model evaluated twice), gate-to-body, drain
junction and source junction.</p>
<p><b>Features.</b> Each sub-model takes VDD, T in K, log(W<sub>eff</sub>/L), &Delta;V<sub>g</sub>/V<sub>T</sub> (the
offset of the gate from its ideal level), the five process parameters relative to nominal, the two terminal
voltages of the bipole, V/V<sub>T</sub> and log|V|. It also takes three Arrhenius-type inputs, 1/V<sub>T</sub>,
V<sub>th0</sub>/V<sub>T</sub> and (VDD&minus;V<sub>th0</sub>)/V<sub>T</sub>, because the sub-threshold current depends on
exp(&minus;V<sub>th</sub>/nV<sub>T</sub>) (assumed physical model).</p>
<p><b>Width normalisation.</b> We chose the width law of each mechanism from a nominal ngspice width sweep
(simulated):</p>
<ul>
<li>The OFF channel current scales with the clamped effective width. Below 20&nbsp;nm, BSIM4 replaces
W<sub>eff</sub> by 2&middot;10<sup>&minus;8</sup>(4&middot;10<sup>&minus;8</sup>&minus;W<sub>eff</sub>)/(6&middot;10<sup>&minus;8</sup>&minus;2W<sub>eff</sub>).
A 1L device (W<sub>eff</sub> = 6&nbsp;nm) then conducts like a 14&nbsp;nm device.</li>
<li>The ON channel current scales with the drawn W.</li>
<li>The tunnelling current scales with W&minus;2&middot;wint.</li>
<li>The junction currents do not depend on W in this model card.</li>
</ul>
<p>Without the clamp, the gate results carried a +2&nbsp;% bias (development log, <code>CHANGES.md</code>).</p>
<h3>3.2 Model families and hyper-parameters</h3>
<ul>
<li><b>Polynomial + ridge:</b> standardised features, polynomial degree 3 or 4, and a ridge strength chosen by
internal cross-validation between 10<sup>&minus;8</sup> and 1.</li>
<li><b>Random forest:</b> 300 trees, max_features 0.6, fully grown.</li>
<li><b>LightGBM:</b> 4,000 trees, learning rate 0.02, 31 or 63 leaves, row subsampling 0.8 and column subsampling 0.9.</li>
<li><b>MLP:</b> three hidden layers of 64 or 128 tanh units, L-BFGS, L2 = 10<sup>&minus;6</sup>, up to 5,000
iterations, standardised inputs and target. We averaged five networks with different seeds in log-G space (a
geometric mean of the conductances), which reduces the run-to-run noise of L-BFGS.</li>
</ul>
<p>All families use the same bipole targets and features. We selected the model of each case by a score computed
on three validation sets: uniform, rail-focused and deployment check. For each set, the score is the mean over the
four currents of (median + 95th percentile)/2 of the relative error, and the final score is the mean over the
three sets.</p>
{table(mct_t, num=cols[2:], caption="Table 2. Model comparison on the uniform validation set (1,500 samples per case). Errors are relative, on currents above 10<sup>&minus;14</sup>&nbsp;A; R&sup2; is computed on asinh(I/1&nbsp;fA), which weights every decade equally. Bold = selected.", best=lambda i, r: bool(best.iloc[i]), cls="small")}
{fig(fig_model_comparison(mc), "Validation median error by model family (log scale). The dotted line is 1&nbsp;%.")}
<p><b>Choice.</b> The MLP has the lowest selection score in all four cases. Over the 16 combinations of case and
current, the median error of the best other family is {cmp["med"].min():.1f} to {cmp["med"].max():.1f} times the median
error of the MLP, and its 95th-percentile error is {cmp["p95"].min():.1f} to {cmp["p95"].max():.1f} times that of the
MLP (Table 2). {cmp_text} The polynomial model has a median I<sub>b</sub> error of {p(poly_ib.min())} to
{p(poly_ib.max())}&nbsp;% in all four cases. We assume that the smooth dependence of log G on the inputs suits a tanh
network, and that the piecewise-constant fits of the tree models cause their larger 95th-percentile errors
(assumed). The selected MLP has 64 units per hidden layer in {n64} of the four cases.</p>
<h3>3.3 Test-set performance of the chosen models</h3>
{table(cht, caption="Table 3. Chosen models on the held-out test set (1,500 uniform samples per case). Errors are relative.")}
{table(dep, caption="Table 4. The same models on the deployment-check test set (1,000 samples per case at the gate widths and rail voltages).")}
<p>On the uniform test set, the median error is at most {p(test_med_max)}&nbsp;% for every current and case. On the
deployment-check test set, the median error of I<sub>d</sub> and I<sub>s</sub> is at most {p(dep_idis_max)}&nbsp;%
({CASE[dep_idis_arg]}), and it is below 1&nbsp;% in {n_dep_below1} of the four cases. The median I<sub>b</sub> error in
the OFF cases of this set is {p(chi.loc['nmos_off','deploy_ib_med%'])}&nbsp;% for NMOS OFF and
{p(chi.loc['pmos_off','deploy_ib_med%'])}&nbsp;% for PMOS OFF. The median |I<sub>b</sub>| of these cases is
{sci(dep_med['nmos_off']['ib'])}&nbsp;A and {sci(dep_med['pmos_off']['ib'])}&nbsp;A, against a median |I<sub>d</sub>| of
{sci(dep_med['nmos_off']['id'])}&nbsp;A and {sci(dep_med['pmos_off']['id'])}&nbsp;A (simulated). For PMOS OFF,
I<sub>g</sub> is above the 10<sup>&minus;14</sup>&nbsp;A evaluation floor in {p(100*chi.loc['pmos_off','ig_eval_frac'],0)}&nbsp;%
of the test samples and I<sub>s</sub> in {p(100*chi.loc['pmos_off','is_eval_frac'],0)}&nbsp;%, and the errors of these
currents refer to those samples.</p>
{fig(F / 'pred_vs_spice_nmos_on.png', 'Predicted vs SPICE, NMOS ON, test set (Id, Ig, Is, Ib).')}
{fig(F / 'pred_vs_spice_nmos_off.png', 'Predicted vs SPICE, NMOS OFF, test set.')}
{fig(F / 'pred_vs_spice_pmos_on.png', 'Predicted vs SPICE, PMOS ON, test set.')}
{fig(F / 'pred_vs_spice_pmos_off.png', 'Predicted vs SPICE, PMOS OFF, test set. The spread in I<sub>g</sub> is at 10<sup>&minus;16</sup>&ndash;10<sup>&minus;15</sup> A, below the evaluation floor.')}
<h3>3.4 Ablation and data size</h3>
<p>In the ablation, we trained the same MLP (same size, same rows, same ensemble size) to predict each terminal
current directly, on an asinh scale. Its median error is {p(ab_ratio,1)} times that of the bipole model (median over
the 16 currents). In the learning curve, we retrained the chosen recipe on 500 to {nmax} uniform samples, each with
half as many rail samples. The median error is within 25&nbsp;% plus 0.05 percentage points of the error of the
full-size model from the following numbers of uniform training samples: {", ".join(f"{CASE[c]} {n}" for c, n in n_enough.items())}.
We therefore take a few thousand ngspice samples per case as the dataset size needed for this accuracy.</p>
{table(abt, caption="Table 5. Ablation: direct terminal-current MLP vs the bipole model (validation set).", cls="small")}
{fig(fig_learning_curve(lc), "Learning curve: validation median (solid) and 95th-percentile (dashed) error vs the number of uniform training samples.")}
{table(piv_t, caption="Table 6. Learning curve, validation median error (%).", cls="small")}
""")

    # ---------------------------------------------------------------- 4 gates
    grt = pd.DataFrame({"Gate": gr.gate, "inputs": gr.inputs,
                        "SPICE median (A)": gr.spice_median.map(sci),
                        "ML median (A)": gr.ml_median.map(sci),
                        "bias %": gr.bias_pct.map(lambda x: f"{x:+.2f}"),
                        "median err %": gr.med_err_pct.map(p), "p95 err %": gr.p95_err_pct.map(p),
                        "max err %": gr.max_err_pct.map(p),
                        "ML-solved med %": gr.solved_med_err_pct.map(p),
                        "ML-solved p95 %": gr.solved_p95_err_pct.map(p),
                        "max |&Delta;V| node (mV)": gr.max_node_dv_mV.map(lambda x: f"{x:.2f}")})
    by_gate = gs.groupby("gate", sort=False).apply(lambda g: mean_acc(g.spice_leak, g.ml_leak), include_groups=False)
    worst_case = gr.loc[gr.p95_err_pct.idxmax()]
    best_case = gr.loc[gr.p95_err_pct.idxmin()]
    S.append(f"""
<h2>4&nbsp; Gate results</h2>
<p>We simulated each of the 18 gate and input cases at the same 200 random corners. A corner is one draw of VDD, T
and the process parameters, with one draw shared by all NMOS and one by all PMOS. As the brief specifies, the main
ML estimate takes the internal and output voltages recorded in ngspice, derives the terminal voltages of every
transistor from them, and sums V&middot;I over the estimated bipole currents. In an extension (Section 7.1) we also
solved the node voltages with the ML models alone. <code>results/gate_samples.csv</code> holds both sets of node
voltages.</p>
{table(grt, caption=f"Table 7. Gate results, 200 corners per case. Over all 3,600 simulations: mean accuracy {p(gate_acc)} %, median error {p(np.median(rel))} %, p95 {p(np.quantile(rel,.95))} %, max {p(rel.max())} %. ML-solved: {p(gate_acc_solved)} %, {p(np.median(rel_s))} %, {p(np.quantile(rel_s,.95))} %, {p(rel_s.max())} %.", cls="small")}
{fig(F / 'gate_parity.png', 'Gate parity plot, all 3,600 simulations (log axes; dotted lines at &plusmn;1 % and &plusmn;5 %).', 62)}
{fig(fig_gate_error_vs_corner(gs), 'Signed gate error vs temperature, VDD and PMOS vth0, in deciles of each variable (median and 5&ndash;95 % band).')}
<p><b>Accuracy by gate and input.</b> The mean accuracy per gate is
{", ".join(f"{g} {p(v)}&nbsp;%" for g, v in by_gate.items())}. The four cases with the largest 95th-percentile error are
{", ".join(f"{r.gate} {r.inputs} ({p(r.p95_err_pct)}&nbsp;%)" for r in top4.itertuples())}. The smallest is
{best_case.gate} {best_case.inputs} ({p(best_case.p95_err_pct)}&nbsp;%). The 95th-percentile errors of all 18 cases lie
between {p(p95_lo)} and {p(p95_hi)}&nbsp;%.</p>
<p>The simulated medians show the stack effect. NAND2 00 leaks {p(sm['INV','0']/sm['NAND2','00'],0)} times less than
INV with input 0 and {p(sm['INV','1']/sm['NAND2','00'],0)} times less than INV with input 1. NOR2 11 leaks
{p(sm['INV','0']/sm['NOR2','11'],0)} and {p(sm['INV','1']/sm['NOR2','11'],0)} times less. In both gates two OFF
transistors are in series, and we assume that the internal node then moves to a voltage that reverse-biases the
upper OFF device (assumed). The ML-solved internal node voltages differ from ngspice by at most {p(node_dv.max())} mV
(median {p(np.median(node_dv),3)} mV), so the models reproduce this shift of the node voltage.</p>
""")

    # ---------------------------------------------------------------- 5 circuit
    cst = pd.DataFrame({"Comparison": csum.comparison.str.replace(r"\s+", " ", regex=True),
                        "SPICE median (A)": csum.spice_median.map(sci), "ML median (A)": csum.ml_median.map(sci),
                        "bias %": csum.bias_pct.map(lambda x: f"{x:+.2f}"), "median err %": csum.med_err_pct.map(p),
                        "p95 err %": csum.p95_err_pct.map(p), "max err %": csum.max_err_pct.map(p)})
    cst["mean acc %"] = [p(mean_acc(cs.spice_leak, cs.ml_gate_sum)), p(mean_acc(cs.spice_leak, cs.spice_gate_sum)),
                         p(acc_mlonly), p(acc["ml_flat"]), p(acc["ml_levels"])]
    worst_vec = cres.loc[cres.med_err_pct.idxmax()]
    best_vec = cres.loc[cres.med_err_pct.idxmin()]
    S.append(f"""
<h2>5&nbsp; Test-circuit results: 3-bit CLA</h2>
<p>We built the adder only from library gates. The propagate signals p<sub>i</sub> = a<sub>i</sub>&oplus;b<sub>i</sub> use
XOR gates made of four NAND2 each, and the generate signals g<sub>i</sub> = a<sub>i</sub>b<sub>i</sub> use AND2. Each
carry is computed directly from g, p and c<sub>0</sub>, for example
c<sub>3</sub> = g<sub>2</sub>+p<sub>2</sub>g<sub>1</sub>+p<sub>2</sub>p<sub>1</sub>g<sub>0</sub>+p<sub>2</sub>p<sub>1</sub>p<sub>0</sub>c<sub>0</sub>
from AND2 and OR2 trees, and s<sub>i</sub> = p<sub>i</sub>&oplus;c<sub>i</sub>. The adder has 42 gates and 204 transistors,
and a unit test checks that it adds correctly for all 128 inputs. We simulated the full netlist in ngspice for all
128 vectors at the same 200 corners (25,600 operating points, {spice_cla_s:.0f}&nbsp;s wall-clock time on {cores}
cores, measured). For the hierarchical estimate that the brief specifies, we propagated the logic values, looked up
the Part B ML leakage of each gate for its input combination and corner, and summed the 42 values.</p>
{table(cst, caption="Table 8. CLA metrics over 128 vectors &times; 200 corners = 25,600 points. Rows 2&ndash;4 split the error into its causes; row 5 is our extension (Section&nbsp;7.2).", cls="small")}
<div class="grid2">
{fig(F / 'circuit_parity.png', 'CLA parity plot (log axes): hierarchical ML (the brief) and levels-aware ML vs the SPICE circuit.')}
{fig(fig_cla_vs_temp(cs), 'Median signed CLA error vs temperature for each method.')}
</div>
<p><b>Comparison with the gates.</b> The hierarchical estimate has a mean accuracy of
{p(acc['ml_gate_sum'])}&nbsp;% (median error {p(hml.med_err_pct)}&nbsp;%, bias {p(hml.bias_pct)}&nbsp;%), which is below the
gate level. Table 8 separates the error into its causes. The ML error alone (ML gate sum against ngspice gate sum)
has a median of {p(mlerr.med_err_pct)}&nbsp;% and a mean accuracy of {p(acc_mlonly)}&nbsp;%. The hierarchy error alone
(ngspice gate sum against the ngspice circuit) has a bias of {p(hier_err.bias_pct)}&nbsp;% and a 95th percentile of
{p(hier_err.p95_err_pct)}&nbsp;%, and it does not depend on the ML models. Flat ML, which applies the models to all
204 transistors at the voltages of the ngspice circuit, has a median error of {p(flat.med_err_pct)}&nbsp;%, against
{p(np.median(rel))}&nbsp;% for the gates, so the transistor models are as accurate inside the large circuit as in the
gates.</p>
<p>The bias of the hierarchy changes with temperature, from
{p(np.median(100*(cs.spice_gate_sum/cs.spice_leak-1)[cs.temp<0]),1)}&nbsp;% below 0&nbsp;&deg;C to
{p(np.median(100*(cs.spice_gate_sum/cs.spice_leak-1)[cs.temp>80]),1)}&nbsp;% above 80&nbsp;&deg;C (median of the ngspice
gate sum against the ngspice circuit). The cause is the level of the logic signals between gates. In ngspice, a
logic 0 at a gate output sits a few to tens of mV above ground and a logic 1 sits below VDD (Figure&nbsp;{N_FIG[0]+1}),
and the offset grows with temperature. The next gate's OFF transistor has this offset on its gate, and its
sub-threshold current depends exponentially on the gate voltage (assumed physical model). The circuit therefore
leaks more than the sum of the ideal-input gate leakages. The levels-aware hierarchy of Section 7.2 removes the bias
without a model of loading between gates. Per vector, the median error ranges from {p(best_vec.med_err_pct)}&nbsp;%
(inputs {best_vec.inputs}) to {p(worst_vec.med_err_pct)}&nbsp;% (inputs {worst_vec.inputs}); <code>results/circuit_results.csv</code>
lists all 128.</p>
{fig(fig_level_shift(gs), 'How far isolated-gate outputs sit from the rails in SPICE (median per temperature decile, log scale).', 70)}
""")

    # ---------------------------------------------------------------- 6 error analysis
    worst = gs.nlargest(8, "ae")[["gate", "vector", "corner", "temp", "vdd", "nmos_vth0", "pmos_vth0", "e"]].copy()
    worst_t = pd.DataFrame({"Gate": worst.gate, "inputs": worst.vector, "corner": worst.corner,
                            "T (&deg;C)": worst.temp.map(lambda x: f"{x:.1f}"), "VDD (V)": worst.vdd.map(lambda x: f"{x:.3f}"),
                            "NMOS vth0": worst.nmos_vth0.map(lambda x: f"{x:.3f}"), "PMOS vth0": worst.pmos_vth0.map(lambda x: f"{x:.3f}"),
                            "error %": worst.e.map(lambda x: f"{x:+.2f}")})
    cold = gs[gs.temp < 0].ae
    warm = gs[gs.temp > 40].ae
    hv = gs[gs.vdd > 1.05 * config.VDD_NOM].ae
    lv = gs[gs.vdd < 0.95 * config.VDD_NOM].ae
    S.append(f"""
<h2>6&nbsp; Error analysis</h2>
<p><b>Where.</b> At gate level, the median |error| is {p(np.median(cold))}&nbsp;% below 0&nbsp;&deg;C (95th percentile
{p(np.quantile(cold,.95))}&nbsp;%) and {p(np.median(warm))}&nbsp;% above 40&nbsp;&deg;C ({p(np.quantile(warm,.95))}&nbsp;%).
Above 1.05&nbsp;V the 95th percentile is {p(np.quantile(hv,.95))}&nbsp;%, and below 0.95&nbsp;V it is
{p(np.quantile(lv,.95))}&nbsp;%. Over the 3,600 gate simulations, |error| has a correlation of {p(r_t)} with T and
{p(r_v)} with VDD. For the ten process parameters, the largest correlation magnitude is {p(r_proc)} with the signed
error and {p(r_proc_abs)} with |error|. The eight largest errors (Table 9) occur at corners with T from
{worst.temp.min():.0f} to {worst.temp.max():.0f}&nbsp;&deg;C and VDD from {worst.vdd.min():.3f} to {worst.vdd.max():.3f}&nbsp;V.</p>
{table(worst_t, caption="Table 9. The eight largest gate errors.", cls="small")}
<p><b>Why.</b> We explain the pattern with three causes (assumed). (1) The OFF currents at low temperature are the
smallest in the dataset, and log G changes fastest with V<sub>th</sub>/V<sub>T</sub> there, so a given error in log G
gives a larger relative error in the current. (2) At high VDD, drain-induced barrier lowering and GIDL make the OFF
current depend more steeply on V<sub>ds</sub> at the point where the gate operates (V<sub>ds</sub> = VDD). (3) Cold and
high-VDD corners lie in the part of the 12-dimensional input box where the uniform training data is thinnest. At
transistor level, the highest 95th-percentile test error is {p(wc[2])}&nbsp;% for {CASE[wc[0]]} I<sub>{wc[1][1]}</sub>
(Table 3).</p>
<p><b>A reference error.</b> For INV with A&nbsp;=&nbsp;1 at corner 198 (112&nbsp;&deg;C, 1.088&nbsp;V), ngspice with its default
Newton start converged to a spurious operating point, which gave an error of +126&nbsp;% against the ML estimate. A
sweep of V(out) with a forcing source (simulated) shows a single KCL root at 42&nbsp;mV (3.58&nbsp;&micro;A). The default
start settles at a narrow kink of the model near 22&nbsp;mV (1.85&nbsp;&micro;A). We start every gate and circuit
simulation from a logic-level <code>.nodeset</code>, which removes this effect. Of the 3,600 gate points, only this
one changed by more than 0.1&nbsp;%, and no CLA point changed (development log, <code>CHANGES.md</code>).</p>
<p><b>Circuit level.</b> The CLA error of the brief's method has a bias of {p(hier_err.bias_pct)}&nbsp;% that changes
with temperature (Section 5). With the levels corrected (Section 7.2), the median CLA error is
{p(lev.med_err_pct)}&nbsp;%, against a gate-level median of {p(np.median(rel))}&nbsp;%.</p>
""")

    # ---------------------------------------------------------------- 7 extensions
    mlv = pd.read_csv(R / "extras_mlv.csv") if (R / "extras_mlv.csv").exists() else None
    mech = pd.read_csv(R / "extras_mechanisms.csv", dtype={"vector": str}) if (R / "extras_mechanisms.csv").exists() else None
    sens = pd.read_csv(R / "extras_sensitivity.csv") if (R / "extras_sensitivity.csv").exists() else None
    mcs = pd.read_csv(R / "extras_montecarlo_summary.csv", dtype={"vector": str}) if (R / "extras_montecarlo_summary.csv").exists() else None
    unc = pd.read_csv(R / "extras_uncertainty.csv", dtype={"vector": str}) if (R / "extras_uncertainty.csv").exists() else None
    cost = pd.read_csv(R / "extras_cost.csv") if (R / "extras_cost.csv").exists() else None
    ext = [f"""
<h2>7&nbsp; Extensions</h2>
<h3>7.1 SPICE-free node voltages</h3>
<p>At DC, the ML terminal currents into every internal node sum to zero. <code>leakml/solve.py</code> solves this KCL
system for all unknown nodes at once with a damped Newton method: a finite-difference Jacobian, and each step
limited to 50&nbsp;mV as in SPICE. It runs on all corners together and iterates only on the rows that have not
converged. The initial guess puts logic nodes at their rails and stack nodes at the rail of their ON neighbour. The
solved node voltages differ from ngspice by at most {p(node_dv.max())} mV, and the leakage computed from them has a
mean accuracy of {p(gate_acc_solved)}&nbsp;% ({p(gate_acc)}&nbsp;% with the ngspice voltages).</p>
<h3>7.2 Levels-aware hierarchy</h3>
<p>The same solver supplies the gate-input levels of the CLA hierarchy. We visit the gates in topological order and
solve each one with its inputs set to the ML-estimated output voltages of its drivers instead of an ideal 0 or VDD.
Rows with the same corner and input voltages are solved once. Over the 42 gates the solver needs at most
{int(lstats.iterations.max())} iterations, and {int(lstats.unconverged.sum())} rows remain unconverged (largest
residual {sci(lstats.max_residual.max())}&nbsp;A, from the solver statistics). The estimated gate-output voltages differ
from the ngspice circuit by {log_value('circuit_log.txt', r'median \|dV\| ([0-9.]+) mV')}&nbsp;mV in the median and
{log_value('circuit_log.txt', r'max ([0-9.]+) mV')}&nbsp;mV at most. The estimated CLA leakage has a <b>mean accuracy of
{p(acc['ml_levels'])}&nbsp;%</b>, with median error {p(lev.med_err_pct)}&nbsp;%, 95th percentile {p(lev.p95_err_pct)}&nbsp;%,
maximum {p(lev.max_err_pct)}&nbsp;% and bias {lev.bias_pct:+.2f}&nbsp;%. Flat ML on the ngspice voltages has a mean
accuracy of {p(acc['ml_flat'])}&nbsp;%. The levels-aware estimate uses no ngspice voltage as input, so the whole chain
from the model card to the circuit leakage runs on the four ML models.</p>
"""]
    if mlv is not None:
        m = mlv.set_index("method")
        ext.append(f"""
<h3>7.3 Minimum-leakage standby vector</h3>
<p>In standby, a block can be parked at the input vector that leaks least. The mean leakage of the 128 CLA vectors
spans a factor of {cs.groupby('vector').spice_leak.mean().max() / cs.groupby('vector').spice_leak.mean().min():.2f}
(simulated). The Spearman correlation between the ML ranking of the vectors and the ngspice ranking is
{m.loc['ml_levels','spearman']:.3f} for the levels-aware estimate and {m.loc['ml_gate_sum','spearman']:.3f} for the
ideal-level hierarchy. The estimated minimum-leakage vector is {m.loc['ml_gate_sum','ml_mlv']} for the ideal-level
hierarchy and {m.loc['ml_levels','ml_mlv']} for the levels-aware estimate, and ngspice gives
{m.loc['ml_levels','spice_mlv']}. When the best vector is chosen separately in each corner, the levels-aware estimate
finds the ngspice optimum in {100*m.loc['ml_levels','corners_exact']:.0f}&nbsp;% of the corners, and its leakage is at most
{p(m.loc['ml_levels','corner_regret_max_pct'])}&nbsp;% above the optimum.</p>
{fig(F / 'extras_mlv.png', 'Mean CLA leakage of the 128 input vectors, sorted by SPICE.')}
""")
    if mech is not None:
        t25 = mech[np.isclose(mech.temp, mech.temp.unique()[np.argmin(np.abs(mech.temp.unique() - 25))])]
        tlo, thi = mech[mech.temp == mech.temp.min()], mech[mech.temp == mech.temp.max()]
        ext.append(f"""
<h3>7.4 Leakage by mechanism</h3>
<p>ngspice reports only total currents. The bipole models split the leakage into mechanisms. The following
shares are averages over the 18 gate cases at nominal process and VDD (estimated):</p>
<ul>
<li>Channel (sub-threshold) leakage is {p(100*tlo.channel_frac.mean(),1)}&nbsp;% at &minus;40&nbsp;&deg;C,
{p(100*t25.channel_frac.mean(),1)}&nbsp;% at {t25.temp.iloc[0]:.0f}&nbsp;&deg;C and
{p(100*thi.channel_frac.mean(),1)}&nbsp;% at 125&nbsp;&deg;C.</li>
<li>Gate tunnelling is the remainder: {p(100*tlo.tunnelling_frac.mean(),1)}&nbsp;% at &minus;40&nbsp;&deg;C and
{p(100*thi.tunnelling_frac.mean(),1)}&nbsp;% at 125&nbsp;&deg;C. Between these temperatures the mean tunnelling current
changes by a factor of {thi.tunnelling.mean()/tlo.tunnelling.mean():.2f} and the mean channel current by a factor of
{thi.channel.mean()/tlo.channel.mean():.0f}.</li>
<li>Junction leakage and GIDL are below {p(100*mech.junction_frac.max(),3)}&nbsp;% in every case of this model card.</li>
</ul>
<p>In NAND2 00 and NOR2 11 the stack reduces the channel current, and tunnelling has a share of
{p(100*t25[(t25.gate=='NAND2')&(t25.vector=='00')].tunnelling_frac.iloc[0],0)}&nbsp;% and
{p(100*t25[(t25.gate=='NOR2')&(t25.vector=='11')].tunnelling_frac.iloc[0],0)}&nbsp;% at
{t25.temp.iloc[0]:.0f}&nbsp;&deg;C.</p>
{fig(F / 'extras_mechanisms.png', 'Left: mechanism currents vs temperature for three gate cases. Right: share of each mechanism per gate case near room temperature.')}
""")
    if sens is not None:
        s = sens.iloc[1:].copy()
        s["span"] = s.ratio_to_nominal
        top = s.iloc[np.argsort(-np.abs(np.log(s.span.to_numpy())))].head(4)
        ext.append(f"""
<h3>7.5 Sensitivity</h3>
<p>We moved each PVT parameter by itself (process parameters and VDD by &plusmn;10&nbsp;%, T to &minus;40 and
125&nbsp;&deg;C) and averaged the estimated levels-aware CLA leakage over all 128 vectors. The four largest effects are
{", ".join(f"{r.case} ({r.ratio_to_nominal:.2f}&times;)" for r in top.itertuples())}. VDD has the largest effect: +10&nbsp;%
gives {sv['VDD +10%']:.2f}&times; and &minus;10&nbsp;% gives {sv['VDD -10%']:.2f}&times;. We assume that the cause is
drain-induced barrier lowering, which lowers the threshold of the OFF devices at higher V<sub>ds</sub> (assumed). PMOS
vth0 and temperature follow, in line with the exp(&minus;V<sub>th</sub>/nV<sub>T</sub>) dependence of the sub-threshold
current (assumed). PMOS parameters have larger effects than NMOS parameters: &minus;10&nbsp;% vth0 gives
{sv['pmos vth0 -10%']:.2f}&times; for PMOS and {sv['nmos vth0 -10%']:.2f}&times; for NMOS. We assume that this is because the
PMOS devices are twice as wide and carry more of the CLA leakage (assumed). Oxide thickness, mobility and junction
depth each change the leakage by at most {p(sv_small,1)}&nbsp;%.</p>
{fig(F / 'extras_sensitivity.png', 'One-at-a-time sensitivity of the CLA leakage (log scale).', 75)}
""")
    if mcs is not None:
        rows = " ".join(
            f"For vector {r.vector}, the median is {sci(r.median)}&nbsp;A and the 99th percentile {r.p99_over_median:.2f}&times; the median "
            f"(lognormal &sigma; = {r.lognorm_sigma:.2f}; the 200 SPICE corners give median {sci(r.spice200_median)}&nbsp;A, KS p = {r.ks_pvalue_vs_spice200:.2f})."
            for r in mcs.itertuples())
        ext.append(f"""
<h3>7.6 Monte-Carlo leakage distribution</h3>
<p>Evaluating the trained models is cheap enough for large statistical studies. We estimated the levels-aware CLA
leakage at {int(mcs.n.iloc[0]):,} new random corners for the minimum- and maximum-leakage vectors. {rows} The
distribution is right-skewed: the 99th percentile is {mcs.p99_over_median.min():.2f} to
{mcs.p99_over_median.max():.2f} times the median.</p>
{fig(F / 'extras_montecarlo.png', 'CLA leakage over new random corners (ML) vs the 200 SPICE corners.')}
""")
    if unc is not None:
        from scipy import stats
        rho = stats.spearmanr(unc.member_spread_pct, unc.abs_err_pct).statistic
        q = pd.qcut(unc.member_spread_pct, 5, labels=False)
        lo_e, hi_e = unc.abs_err_pct[q == 0].median(), unc.abs_err_pct[q == 4].median()
        ext.append(f"""
<h3>7.7 Uncertainty from the ensemble</h3>
<p>Each of the five ensemble members gives its own gate leakage, and the spread between the members can serve as an
error bar. The Spearman correlation between the spread and the actual |error| is {rho:.2f}. The 20&nbsp;% of the
points with the smallest spread have a median |error| of {p(lo_e)}&nbsp;%, and the 20&nbsp;% with the largest spread have
{p(hi_e)}&nbsp;%. We assume that the part of the error shared by all members comes from the training distribution, and
the spread cannot show this part (assumed).</p>
{fig(F / 'extras_uncertainty.png', 'Ensemble spread vs actual error, all 3,600 gate simulations.', 60)}
""")
    if cost is not None:
        c = cost.set_index("method")
        spice_ms = c.ms_per_cpu_eval.iloc[0]
        flat_ms = c.ms_per_cpu_eval.iloc[1]
        ext.append(f"""
<h3>7.8 Cost</h3>
<p>ngspice needs {p(spice_ms)}&nbsp;ms of CPU time per CLA operating point (measured on {platform}). The ML models
need {p(flat_ms,3)}&nbsp;ms per point on the same 204 transistors when the node voltages are given, which is
{spice_ms/flat_ms:.0f} times less CPU time. The lookup of the brief's hierarchy takes {hier_s:.1f}&nbsp;s for all 25,600
points once the gate tables exist. The levels-aware solve of all 25,600 points takes {lev_s:.0f}&nbsp;s on one core,
which is {c.cpu_seconds.iloc[0]/lev_s:.1f} times less CPU time than ngspice. In wall-clock time, ngspice took
{spice_cla_s:.0f}&nbsp;s on {cores} cores. Training the {n_sub} sub-models took {train_s}&nbsp;s of wall-clock time
(measured; the sub-models were trained in parallel).</p>
""")
    if (R / "parity_summary.csv").exists():
        ps = pd.read_csv(R / "parity_summary.csv")
        pt = pd.DataFrame({"Comparison": ps.comparison, "bias %": ps.bias_pct.map(lambda x: f"{x:+.2f}"),
                           "median err %": ps.med_err_pct.map(p), "p95 err %": ps.p95_err_pct.map(p),
                           "max err %": ps.max_err_pct.map(p), "mean acc %": ps.mean_acc.map(p)})
        lv, hr = ps.iloc[-1], ps.iloc[0]
        ext.append(f"""
<h3>7.9 Second test circuit: 8-bit parity checker</h3>
<p>We also built the brief's other test circuit from the same library: a balanced tree of seven XORs of four NAND2
each (28 gates, 112 transistors). We simulated it in ngspice for all 256 vectors at the same 200 corners (51,200
points) and applied the models without retraining or other changes. The brief's hierarchy has a mean accuracy of
{p(hr.mean_acc)}&nbsp;% (bias {hr.bias_pct:+.2f}&nbsp;%), and the levels-aware hierarchy has {p(lv.mean_acc)}&nbsp;%
(median error {p(lv.med_err_pct)}&nbsp;%, 95th percentile {p(lv.p95_err_pct)}&nbsp;%). The ranking of the two
hierarchies is the same as for the CLA.</p>
{table(pt, caption="Table 10. 8-bit parity checker, 256 vectors &times; 200 corners.", cls="small")}
{fig(F / 'parity_parity.png', 'Parity checker: ML vs SPICE circuit leakage (log axes).', 55)}
""")
    S.append("".join(ext))

    # ---------------------------------------------------------------- 8 limitations
    S.append(f"""
<h2>8&nbsp; Limitations and improvements</h2>
<ul>
<li><b>Training distribution.</b> The 10,000 uniform samples rarely lie where gates operate (terminals on the rails,
cold and high-VDD corners). The rail supplement and the deployment-check set reduce the error there, and the largest
remaining gate errors occur at cold, high-VDD corners (Section 6). Adding ngspice samples where the ensemble members
disagree most would target these corners. We have not tried this.</li>
<li><b>Ideal-level hierarchy.</b> The hierarchy that the brief specifies has a bias of {p(hier_err.bias_pct)}&nbsp;% on
this circuit for any ML model, because the exact ngspice gate leakages give a mean accuracy of
{p(mean_acc(cs.spice_leak, cs.spice_gate_sum))}&nbsp;%. The levels-aware hierarchy (Section 7.2) removes most of this
bias.</li>
<li><b>Loading between gates.</b> The levels-aware hierarchy ignores the tunnelling current that a gate input draws
from its driver. Averaged over the 18 gate cases, tunnelling is {p(tun_hot,1)}&nbsp;% of the leakage at 125&nbsp;&deg;C and
{p(tun_cold,1)}&nbsp;% at &minus;40&nbsp;&deg;C (Section 7.4), and it is {p(tun_stack[0],0)}&nbsp;% in NAND2 00 and
{p(tun_stack[1],0)}&nbsp;% in NOR2 11 near room temperature. We did not test the effect of loading on these stacked
cases.</li>
<li><b>Gate sizes and types.</b> Both test circuits use the same five library gates at W<sub>n</sub> = 2L and
W<sub>p</sub> = 4L, and the deployment-check set uses these widths. We did not test other gate sizes between 1 and 8
times L or other gate types, such as three-input stacks.</li>
<li><b>Reference model.</b> The models learn the BSIM4 PTM model card as simulated by ngspice. Currents below
10<sup>&minus;14</sup>&nbsp;A are simulator noise, and we excluded them from the percentage errors. The junction geometry
(AS, AD, PS, PD) is a default constant. We did not compare the models with measurements.</li>
<li><b>Accuracy metric.</b> The mean accuracy hides the tails, which the 95th percentile and the maximum describe.</li>
<li><b>Solver speed.</b> The levels-aware solve uses finite-difference Jacobians, which need many model calls per
Newton step. We assume that an analytic Jacobian or a small learned gate-output model would reduce the run time to
that of the lookup (assumed); we have not tried either.</li>
<li><b>Scope.</b> The models cover W = 1 to 8 L, &plusmn;10&nbsp;% process variation and &minus;40 to
125&nbsp;&deg;C, and they extrapolate outside this range. As the brief specifies, one process draw is shared by all
devices of a type, so within-die mismatch is not modelled. The models accept per-transistor process parameters, so
mismatch needs only per-transistor draws (assumed). The simulations are DC operating points.</li>
<li><b>Simulator noise.</b> According to the development log, data regenerated with ngspice 44.2 and 45.2 differ only
below about 10<sup>&minus;12</sup>&nbsp;A. L-BFGS amplifies this to &plusmn;0.02 points in gate accuracy with five seeds
(&plusmn;0.07 with three), and training is bit-reproducible on identical data. We did not test whether the
{p(abs(acc['ml_levels'] - acc['ml_flat']))}-point difference between the levels-aware and flat-ML CLA accuracies
(Section 7.2) exceeds this variation.</li>
</ul>

<h2>9&nbsp; Reproducibility and deliverables</h2>
<p>All scripts run in order (<code>./run_all.sh --full</code>) from the model file with fixed seeds. The fast path
(<code>./run_all.sh</code>) reuses the stored data, the saved models and the cached CLA ngspice reference, and reruns
the validation in minutes. The test file <code>tests/test_pipeline.py</code> defines 14 test functions, and the
development log reports 26 collected test cases. They check the KCL balance of the saved models, the passivity of
the bipoles, the identity of the netlist and ML topologies, the truth tables, the addition of the CLA, the BSIM4
width clamp, the convergence of the solver, and the parity function of the parity checker. The repository is at
<a href="https://github.com/Shr1yaK/leaky-lima-leakage-estimation/">https://github.com/Shr1yaK/leaky-lima-leakage-estimation/</a>.</p>
{table(pd.DataFrame([
    ["1", "Datasets (4 &times; 10,000 rows)", "data/{nmos,pmos}_{on,off}.csv"],
    ["2", "Example netlists", "netlists/single_*.cir, gate_*.cir, cla3.cir"],
    ["3", "Kirchhoff check, trend plots", "results/kirchhoff_check.csv, figures/data_check_trends.png (Table 1, Figure 1)"],
    ["4", "Model comparison", "results/model_comparison.csv (Table 2)"],
    ["5", "Chosen models", "results/chosen_models_test.csv, figures/pred_vs_spice_*.png, saved_models/*.joblib (Table 3)"],
    ["6", "Gate results", "results/gate_results.csv, results/gate_samples.csv (Table 7)"],
    ["7", "Gate parity plot", "figures/gate_parity.png"],
    ["8", "CLA results", "results/circuit_summary.csv, circuit_results.csv, figures/circuit_parity.png (Table 8)"],
    ["9", "Code", "00_ &hellip; 08_*.py, leakml/, run_all.sh, tests/"],
    ["10", "Report", "report/Leaky_Lima_Report.pdf (this document, generated by report/build_report.py)"],
], columns=["#", "Deliverable", "Where"]), num=[], cls="small")}
""")
    return "\n".join(S)


CSS = """
@page { size: A4; margin: 16mm 15mm 16mm 15mm; }
body { font-family: "Charter", "Georgia", serif; font-size: 10.2pt; line-height: 1.42; color: #111; background: #fff; max-width: 190mm; margin: auto; }
header { text-align: center; margin-bottom: 8px; }
.course { font-variant: small-caps; letter-spacing: .05em; margin: 0; color: #444; }
h1 { font-size: 19pt; margin: 4px 0 4px; }
.sub { margin: 0 0 6px; color: #333; }
.abstract { border-left: 3px solid #2a5d9f; background: #f3f6fa; padding: 8px 12px; margin: 10px 0 14px; }
h2 { font-size: 13.5pt; border-bottom: 1px solid #999; padding-bottom: 2px; margin-top: 20px; break-after: avoid; }
h3 { font-size: 11pt; margin: 14px 0 4px; break-after: avoid; }
p { margin: 5px 0; text-align: justify; }
ul { margin: 4px 0 6px; padding-left: 20px; }
li { margin: 2px 0; }
code { font-family: "Menlo", monospace; font-size: 8.4pt; background: #f2f2f2; padding: 0 2px; }
table { border-collapse: collapse; margin: 8px auto; font-size: 8.6pt; break-inside: avoid; }
table.small { font-size: 7.4pt; }
caption { caption-side: top; text-align: left; font-size: 8.6pt; padding-bottom: 3px; }
th, td { border-bottom: 1px solid #ddd; padding: 2px 5px; text-align: left; vertical-align: top; }
thead th { border-bottom: 1.5px solid #333; border-top: 1.5px solid #333; }
td.n, th.n { text-align: right; font-variant-numeric: tabular-nums; }
tr.best td { font-weight: bold; }
figure { margin: 8px auto; break-inside: avoid; text-align: center; }
figure img { width: 100%; }
figcaption { font-size: 8.4pt; text-align: left; margin-top: 2px; }
.grid2 { display: grid; grid-template-columns: 1fr 1fr; gap: 6px 12px; }
.grid2 figure { width: 100% !important; }
"""


def main():
    body = build()
    page = (f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width, initial-scale=1">'
            f"<title>Leaky Lima Report</title><style>{CSS}</style></head><body>{body}</body></html>")
    out = HERE / "report.html"
    out.write_text(page, encoding="utf-8")
    print("wrote", out)
    if CHROME:
        subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                        f"--print-to-pdf={PDF}", out.as_uri()], check=True, capture_output=True)
        print("wrote", PDF)
    else:
        print("Chrome not found: open report.html and print to PDF")


if __name__ == "__main__":
    main()
