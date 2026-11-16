"""Publication figures (vector PDF) from a results.json produced by run_all_experiments / run_ablation / run_scalability."""
from _common import *
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from fed_matd3_iout.evaluation.report_generator import seed_scores
from fed_matd3_iout.evaluation.statistical_tests import mean_sd_ci

plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.labelsize": 9, "legend.fontsize": 8,
                     "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False})
p = argparse.ArgumentParser()
p.add_argument("--results", required=True)
p.add_argument("--metrics", nargs="*", default=["U_total", "delivered_mbit", "overflow_mbit", "E_total_j"])
a = p.parse_args()
rp = Path(a.results)
res = json.load(open(rp))
LAB = {"U_total": "Total semantic utility (–)", "delivered_mbit": "Delivered data (Mbit)", "overflow_mbit": "Buffer overflow (Mbit)",
       "E_total_j": "Total energy (J)", "U_norm": "Normalised semantic utility (–)"}
for m in a.metrics:
    names = list(res)
    st = [mean_sd_ci(seed_scores({int(k): v for k, v in res[n].items()}, m)) for n in names]
    fig, ax = plt.subplots(figsize=(max(4, 0.6 * len(names) + 2), 3))
    x = np.arange(len(names))
    ax.bar(x, [s["mean"] for s in st], yerr=[[s["mean"] - s["ci_lo"] for s in st], [s["ci_hi"] - s["mean"] for s in st]], capsize=3,
           color="0.6", edgecolor="k", linewidth=0.5)   # bars start at zero (no truncated axis)
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=40, ha="right"); ax.set_ylabel(LAB.get(m, m))
    ax.set_title(f"mean and 95% CI over {st[0]['n']} training seed(s)", fontsize=8)
    fig.tight_layout(); fig.savefig(rp.parent / f"fig_{m}.pdf"); plt.close(fig)
print("figures written next to", rp)
