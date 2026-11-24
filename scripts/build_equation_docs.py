"""Render docs/MATHEMATICAL_MODEL_AUDIT.md and docs/EQUATION_IMPLEMENTATION_MAP.md from scripts/equation_registry.py."""
from _common import *
from equation_registry import E

CL = {"A": "A  established", "B": "B  valid under stated assumptions", "C": "C  proposed (needs experimental justification)",
      "D": "D  approximation (needs validation)", "E": "E  inconsistent as printed -> corrected"}
head = ("# Mathematical Model Audit\n\nGenerated from `scripts/equation_registry.py` (do not edit by hand). Classes: A established; B valid under "
        "stated assumptions; C proposed modelling choice requiring experimental justification; D approximation requiring validation; E incorrect/inconsistent "
        "as printed and corrected. **Nothing here is claimed to be experimentally verified**; class C/D items are unvalidated. P1 = SAC-TD3 paper, "
        "P2 = STA-TA paper (both read for this audit). Original `dt_smarl_step8` code could not be inspected (see DEVELOPMENT_STATUS.md).\n\n")
out = [head]
for c in "ABCDE":
    n = sum(1 for e in E if e["cls"] == c)
    out.append(f"- Class {CL[c]}: {n} entries\n")
out.append("\n## Discrepancies and decisions\n\n"
 "1. **P1 Eq.(8)/(11) (path loss, SNR).** Printed with l = d/1000 km and no reference range, so TL < 0 dB for d < 1 km; P_t (W) is divided by a dB-domain noise PSD. "
 "Implemented instead: 1 m reference, SL re 1 uPa @1 m, SNR in dB (C2, C4, C5). Class E as printed.\n"
 "2. **Energy.** P1/P2 use a quadratic action penalty; this project uses cubic propulsion power (E1) plus separate tx/rx/compute terms. They are not mixed; coefficients uncalibrated.\n"
 "3. **P1 sigma (Eq.18)** is a category-mixture mean per node; here category is drawn per packet with the same mean (S5). c enters once (S2).\n"
 "4. **P1 eta_sem = U/N_del** is reported as `U_per_packet`; `U_norm` (generated-referenced) is additional (S4).\n"
 "5. **P2 reward (Eq.14)** is not reproduced; R1 is the proposed reward of the task specification.\n"
 "6. **mu for the 'high' category and initial buffer law** are not in the papers; assumed and exposed in config.\n"
 "7. **Own implementation bug found and fixed:** truncated Poisson tail (P4).\n"
 "8. **Reward shaping** invariance is not strict under target switching (R2).\n"
 "9. **Link range** is a 3-D slant distance; AUVs sit 50 m below the USV, so range < 50 m can never connect.\n\n## Equation registry\n")
for e in E:
    out.append(f"\n### {e['id']} - {e['name']}  \n**Class:** {CL[e['cls']]}\n\n"
               f"- **Expression:** `{e['expr']}`\n- **Meaning:** {e['meaning']}\n- **Symbols / units:** {e['symbols']}\n- **Assumptions:** {e['assumptions'] or '-'}\n"
               f"- **Relation to Paper 1/2:** {e['paper']}\n- **Implementation:** `{e['impl']}`\n- **Test:** `{e['test'] or 'NONE (documented gap)'}`\n"
               f"- **Config:** {e['config'] or '-'}\n- **Known limitations:** {e['limits'] or '-'}\n")
(ROOT / "docs" / "MATHEMATICAL_MODEL_AUDIT.md").write_text("".join(out))
rows = ["# Equation -> Implementation Map\n\nGenerated from `scripts/equation_registry.py`; `tests/test_equation_registry.py` verifies every path/function/test below exists.\n\n"
        "| ID | Class | Module : function | Test | Config parameter |\n|---|---|---|---|---|\n"]
for e in E:
    rows.append(f"| {e['id']} | {e['cls']} | `{e['impl']}` | `{e['test'] or 'GAP'}` | {e['config'] or '-'} |\n")
(ROOT / "docs" / "EQUATION_IMPLEMENTATION_MAP.md").write_text("".join(rows))
print(len(E), "equations rendered")
