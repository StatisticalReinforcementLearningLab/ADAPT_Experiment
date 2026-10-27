from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from testbed import STE_LAYER1_DIRS


ROOT = Path(__file__).resolve().parents[1]
PARAMS = ROOT / "env_para_vanilla"


def test_benefit_path_kappa1_is_noop():
    from tune_ste import KNOBS, scale_params

    params = json.loads((PARAMS / "params_env_13.json").read_text(encoding="utf-8"))
    knob = KNOBS["benefit_path"]
    scaled, audit = scale_params(params, knob.build(1.0), apply=knob.apply)
    assert audit == []
    for key in ("theta_fourSC", "theta_antic", "theta_CAE"):
        np.testing.assert_array_equal(
            np.asarray(params[key], dtype=float),
            np.asarray(scaled[key], dtype=float),
        )


def test_benefit_path_centering_keeps_never_send_cae(tmp_path, monkeypatch):
    """Never-send mean weekly CAE (linear predictor at x̄) stays within 1e-6."""
    from tune_ste import KNOBS, write_knob_params
    from vani_env import THETA_CAE_NAMES, EnvConfig

    centers = {
        "E": 0.4,
        "fourSC_ewma": 0.15,
        "antic_ewma": -0.25,
        "PV_week": 0.0,
        "FW_week": 0.0,
        "PJ_week": 0.0,
    }

    def fake_centers(src_dir, user_ids, **_kw):
        return {int(u): dict(centers) for u in user_ids}

    monkeypatch.setattr("tune_ste.covariate_centers", fake_centers)
    dst = tmp_path / "env_para_ste0.2_benefit"
    write_knob_params(PARAMS, dst, [13], KNOBS["benefit_path"], 2.0)

    van = EnvConfig(13, params_dir=PARAMS, activity="off")
    got = EnvConfig(13, params_dir=dst, activity="off")
    i_int = THETA_CAE_NAMES.index("intercept")
    i_sc = THETA_CAE_NAMES.index("fourSC_ewma")
    i_aa = THETA_CAE_NAMES.index("anticipated_affect_ewma")
    x = np.zeros(len(van.theta_CAE), dtype=float)
    x[i_int] = 1.0
    x[i_sc] = centers["fourSC_ewma"]
    x[i_aa] = centers["antic_ewma"]
    assert float(got.theta_CAE @ x) == pytest.approx(float(van.theta_CAE @ x), abs=1e-6)
    assert got.theta_CAE[i_sc] != van.theta_CAE[i_sc]

    src_block = json.loads(
        (PARAMS / "generator_calibration.json").read_text(encoding="utf-8")
    )["activity_process"]
    dst_block = json.loads(
        (dst / "generator_calibration.json").read_text(encoding="utf-8")
    )["activity_process"]
    assert dst_block == src_block
    assert "stayer_variants" in dst_block


def test_layer1_dry_run_lists_seven_targets(capsys):
    from tune_ste import build_parser

    args = build_parser().parse_args(["layer1", "--dry-run"])
    args.func(args)
    out = capsys.readouterr().out
    for name in STE_LAYER1_DIRS:
        assert name in out
    assert "knob=action" in out
    assert "knob=benefit_path" in out
    assert "knob=burden_path" in out
    assert "knob=burden" in out
    assert "scan to kappa=0" in out
