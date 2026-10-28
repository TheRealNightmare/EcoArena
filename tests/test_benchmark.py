from ecoarena.env import load_config
from ecoarena.eval.benchmark import evaluate_worlds


def test_evaluate_worlds_records_each_world():
    cfg = load_config("S1")
    res = evaluate_worlds(cfg, {0: "heuristic", 1: "random"}, worlds=2, steps=30, seed=5, device="cpu",
                          record_every=10)
    assert res["pops"].shape == (3, 2, 2)
    assert len(res["per_world"]) == 2
    w = res["per_world"][0]
    assert set(w["extinct_at"]) == {"prey", "predator"}
    assert w["survived"] and w["extinct_at"]["prey"] is None
    assert res["summary"]["worlds_survived"] == 2
    assert 0 <= res["summary"]["attack_rate_e6"]
