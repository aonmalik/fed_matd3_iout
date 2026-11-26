from fed_matd3_iout.baselines.registry import METHODS
from fed_matd3_iout.evaluation.comparability import ATTR, FEDERATED, comparability, deployment
from fed_matd3_iout.evaluation.report_generator import comparisons, table
from fed_matd3_iout.validation import fairness


def test_every_method_classified_and_matches_built_configs(cfg):
    assert set(ATTR) == set(METHODS)
    fa = fairness.run_audit(cfg)
    assert fa["checks"]["every_method_has_comparability_attributes"] and fa["checks"]["comparability_attributes_match_built_configs"]
    assert "NOT equivalent" in fa["statement"] and len(fa["non_equivalent"]) >= 10        # declared non-equivalences are preserved
    assert fa["n_pairs_like_for_like"] == 10 and fa["n_pairs_total"] == 55                # only the 10 federated-vs-federated pairs


def test_global_federated_vs_local_actors_is_never_like_for_like():
    c = comparability("s3a_fedmatd3", "centralized_matd3")
    assert c["class"] == "NOT_like_for_like" and any(x.startswith("N5") for x in c["inherited_non_equivalences"])
    assert c["deployment_a"] == "global_federated_actor" and c["deployment_b"] == "per_auv_local_actors"
    assert comparability("s3a_fedmatd3", "mappo")["class"] == "NOT_like_for_like"
    assert comparability("s3a_fedmatd3", "greedy_matd3")["inherited_non_equivalences"]
    assert comparability("s3a_fedmatd3", "fedavg_matd3")["class"] == "like_for_like_weighting_only"
    assert comparability("s3a_fedmatd3", "fedavg_matd3")["inherited_non_equivalences"] == []


def test_comparison_rows_and_table_carry_the_labels():
    res = {m: {s: [{"U_total": 10.0 + s + (0.3 if m == "s3a_fedmatd3" else 0.0) + 0.1 * s * len(m) % 3}] for s in range(4)}
           for m in ("s3a_fedmatd3", "fedavg_matd3", "centralized_matd3")}
    rows = {r["method"]: r for r in comparisons(res, "s3a_fedmatd3", ("U_total",))}
    assert rows["fedavg_matd3"]["comparability"]["class"] == "like_for_like_weighting_only"
    assert rows["centralized_matd3"]["comparability"]["class"] == "NOT_like_for_like"
    assert {r["method"]: r["deployment"] for r in table(res, ("U_total",))}["s3a_fedmatd3"] == deployment("s3a_fedmatd3")
