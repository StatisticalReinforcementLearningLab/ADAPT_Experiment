"""Testbed spec: params dir, activity, stayer fraction, and misspecification.

A testbed is the four-tuple used by the 28-environment catalog. The same
object is read from ``ADAPR_*`` environment variables or CLI flags, written
into run metadata, and turned into a results-folder suffix so two specs
never share an output directory.

Activity ``off`` is the existing ``activity_lambda = 0`` hook (every day
engaged). Activity ``on`` is the pooled activity process. Stayer fraction is
ignored, with a warning, when activity is off. Misspec ``D`` requires
activity on.
"""

from __future__ import annotations

import argparse
import os
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path

from misspec import MISSPEC_C_BLURB
from vani_env import resolve_params_dir


ACTIVITY_CHOICES = ("on", "off")
MISSPEC_CHOICES = ("none", "A", "B", "C", "D")
STAYER_CHOICES = (0.0, 0.25, 0.5)
STE_PARAM_DIRS = ("env_para_ste0.2", "env_para_ste0.5", "env_para_ste0.8")
STE_UNIFORM_DIRS = STE_PARAM_DIRS
STE_LAYER1_DIRS = (
    "env_para_ste0.2",
    "env_para_ste0.5",
    "env_para_ste0.8",
    "env_para_ste0.2_benefit",
    "env_para_ste0.8_benefit",
    "env_para_ste0.2_burden",
    "env_para_ste0.8_burden",
)

_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off"}


def parse_activity(value=None) -> str:
    """``on`` or ``off``. Default ``on`` (``ADAPR_ACTIVITY``)."""
    raw = value if value is not None else os.environ.get("ADAPR_ACTIVITY", "on")
    text = str(raw).strip().lower()
    if text in _TRUE or text == "on":
        return "on"
    if text in _FALSE or text == "off":
        return "off"
    raise ValueError(f"activity must be 'on' or 'off', got {value!r}")


def parse_stayer_fraction(value=None) -> float:
    """Stayer fraction. ``ADAPR_STAYER_FRACTION``, else ``ADAPR_ACTIVITY_STAYER_FRACTION``."""
    if value is None:
        raw = os.environ.get("ADAPR_STAYER_FRACTION")
        if raw is None or str(raw).strip() == "":
            raw = os.environ.get("ADAPR_ACTIVITY_STAYER_FRACTION", "0")
        value = raw
    fraction = float(value)
    if not 0.0 <= fraction <= 1.0:
        raise ValueError(f"stayer fraction must be in [0, 1], got {value!r}")
    return fraction


def parse_misspec(value=None) -> str:
    """``none``, ``A``, ``B``, ``C``, or ``D``. Default ``none`` (``ADAPR_MISSPEC``)."""
    raw = value if value is not None else os.environ.get("ADAPR_MISSPEC", "none")
    text = str(raw).strip()
    if text == "" or text.lower() in {"none", "off", "0"}:
        return "none"
    letter = text.upper()
    if letter not in {"A", "B", "C", "D"}:
        raise ValueError(
            f"misspec must be one of {MISSPEC_CHOICES}, got {value!r}"
        )
    return letter


def params_variant_name(params_dir) -> str:
    """``env_para_ste0.5`` → ``ste0.5``; used in results-folder suffixes."""
    name = Path(params_dir).name
    prefix = "env_para_"
    if name.startswith(prefix):
        return name[len(prefix):]
    return name or "params"


def format_stayer(fraction: float) -> str:
    value = float(fraction)
    if abs(value - round(value)) < 1e-12:
        return str(int(round(value)))
    text = f"{value:.2f}".rstrip("0").rstrip(".")
    return text


def results_suffix(params_dir, activity: str, stayer_fraction: float, misspec: str) -> str:
    """``ste0.5_gapon_psi0.25_misspecA``."""
    gap = "gapon" if parse_activity(activity) == "on" else "gapoff"
    miss = parse_misspec(misspec)
    miss_tag = "misspecnone" if miss == "none" else f"misspec{miss}"
    return (
        f"{params_variant_name(params_dir)}_"
        f"{gap}_psi{format_stayer(stayer_fraction)}_{miss_tag}"
    )


def results_root_name(params_dir, activity: str, stayer_fraction: float, misspec: str) -> str:
    return f"results_{results_suffix(params_dir, activity, stayer_fraction, misspec)}"


@dataclass(frozen=True)
class TestbedSpec:
    """One selectable testbed: params dir, activity, stayer fraction, misspec."""

    params_dir: Path
    activity: str
    stayer_fraction: float
    misspec: str
    layer: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "params_dir", Path(self.params_dir).expanduser())
        object.__setattr__(self, "activity", parse_activity(self.activity))
        object.__setattr__(self, "stayer_fraction", float(self.stayer_fraction))
        object.__setattr__(self, "misspec", parse_misspec(self.misspec))
        if self.misspec == "D" and self.activity != "on":
            raise ValueError("misspec D requires activity on")

    @property
    def suffix(self) -> str:
        return results_suffix(
            self.params_dir, self.activity, self.stayer_fraction, self.misspec
        )

    @property
    def results_root(self) -> str:
        return f"results_{self.suffix}"

    @property
    def ste_results_root(self) -> str:
        """STE always uses activity off; the folder still records that fact."""
        ste = self.with_activity("off")
        return f"results_ste_{ste.suffix}"

    def with_activity(self, activity: str) -> "TestbedSpec":
        misspec = "none" if parse_activity(activity) == "off" and self.misspec == "D" else self.misspec
        return TestbedSpec(
            params_dir=self.params_dir,
            activity=activity,
            stayer_fraction=self.stayer_fraction,
            misspec=misspec,
            layer=self.layer,
        )

    def as_dict(self) -> dict:
        payload = asdict(self)
        payload["params_dir"] = str(self.params_dir)
        payload["suffix"] = self.suffix
        payload["results_root"] = self.results_root
        return payload

    def export_env(self) -> dict[str, str]:
        """Values for ``sbatch --export`` / ``os.environ``."""
        return {
            "ADAPR_PARAMS_DIR": str(self.params_dir),
            "ADAPR_ACTIVITY": self.activity,
            "ADAPR_STAYER_FRACTION": str(self.stayer_fraction),
            "ADAPR_ACTIVITY_STAYER_FRACTION": str(self.stayer_fraction),
            "ADAPR_MISSPEC": self.misspec,
        }

    def ste_spec(self) -> "TestbedSpec":
        """STE always uses activity off; misspec D is dropped with that."""
        return self.with_activity("off")


def resolve_testbed_spec(
    params_dir=None,
    activity=None,
    stayer_fraction=None,
    misspec=None,
    *,
    warn_stayer: bool = True,
) -> TestbedSpec:
    """Build a spec from explicit values, else ``ADAPR_*`` / defaults."""
    params = resolve_params_dir(params_dir)
    act = parse_activity(activity)
    stayer = parse_stayer_fraction(stayer_fraction)
    miss = parse_misspec(misspec)
    if act == "off" and stayer != 0.0 and warn_stayer:
        warnings.warn(
            f"stayer fraction {stayer:g} is ignored when activity is off",
            RuntimeWarning,
            stacklevel=2,
        )
    if miss == "D" and act != "on":
        raise ValueError("misspec D requires activity on")
    return TestbedSpec(
        params_dir=params, activity=act, stayer_fraction=stayer, misspec=miss
    )


def spec_from_env(params_dir=None) -> TestbedSpec:
    return resolve_testbed_spec(params_dir=params_dir)


def add_testbed_arguments(parser: argparse.ArgumentParser, *, params_dir: bool = False) -> None:
    """Attach ``--activity``, ``--stayer-fraction``, ``--misspec`` (and optionally ``--params-dir``)."""
    if params_dir:
        parser.add_argument(
            "--params-dir",
            type=Path,
            default=None,
            help="Parameter folder (default: ADAPR_PARAMS_DIR or env_para_vanilla).",
        )
    parser.add_argument(
        "--activity",
        choices=ACTIVITY_CHOICES,
        default=None,
        help="Activity process: on (pooled) or off (lambda=0, no gaps). "
             "Default: ADAPR_ACTIVITY or on.",
    )
    parser.add_argument(
        "--stayer-fraction",
        type=float,
        default=None,
        help="Stayer fraction 0, 0.25, or 0.5 (ignored when activity is off). "
             "Default: ADAPR_STAYER_FRACTION or 0.",
    )
    parser.add_argument(
        "--misspec",
        choices=MISSPEC_CHOICES,
        default=None,
        help="Misspecification none/A/B/C/D. Default: ADAPR_MISSPEC or none.",
    )


def catalog(params_root: Path | None = None) -> list[TestbedSpec]:
    """The 28 official testbeds (7 Layer 1 + 9 Layer 2 + 12 Layer 3).

    Layers 2 and 3 use only the three uniform (action-knob) folders.
    """
    root = Path(params_root) if params_root is not None else Path(".")
    rows: list[TestbedSpec] = []
    for name in STE_LAYER1_DIRS:
        rows.append(
            TestbedSpec(root / name, "off", 0.0, "none", layer=1)
        )
    for name in STE_UNIFORM_DIRS:
        for psi in STAYER_CHOICES:
            rows.append(
                TestbedSpec(root / name, "on", psi, "none", layer=2)
            )
    for name in STE_UNIFORM_DIRS:
        for miss in ("A", "B", "C", "D"):
            rows.append(
                TestbedSpec(root / name, "on", 0.0, miss, layer=3)
            )
    if len(rows) != 28:
        raise RuntimeError(f"expected 28 testbeds, got {len(rows)}")
    return rows


def catalog_24(params_root: Path | None = None) -> list[TestbedSpec]:
    """Alias for :func:`catalog` (28 testbeds)."""
    return catalog(params_root)


MISSPEC_BLURBS = {
    "none": "Fitted generator; no extra misspecification.",
    "A": "Unmeasured weekly confounder of SC, AA, and Y; causal SC/AA→Y halved.",
    "B": "Unmeasured direct A→Y path; mediator action effects halved.",
    "C": MISSPEC_C_BLURB,
    "D": "Recent burden raises the gap-start hazard; activity must be on.",
}


def log_ste_activity_off(requested: TestbedSpec | None = None) -> str:
    """STE tuning/eval always run with activity off. Print and return the line."""
    extra = ""
    if requested is not None and requested.activity != "off":
        extra = (
            f" (testbed activity={requested.activity} "
            f"stayer={requested.stayer_fraction:g} ignored for no-gap STE)"
        )
    if requested is not None and requested.misspec == "D":
        extra += " (misspec D disabled because STE has no gaps)"
    msg = f"STE: activity off (no-gap, lambda=0){extra}"
    print(msg, flush=True)
    return msg


def apply_spec_to_env_config(cfg, spec: TestbedSpec) -> None:
    """Write activity / stayer / misspec onto an ``EnvConfig``."""
    cfg.activity = spec.activity
    cfg.misspec = spec.misspec
    cfg.activity_stayer_fraction = (
        0.0 if spec.activity == "off" else float(spec.stayer_fraction)
    )
    if spec.activity == "off":
        cfg.activity_lambda = 0.0
    if spec.misspec == "D" and spec.activity != "on":
        raise ValueError("misspec D requires activity on")
