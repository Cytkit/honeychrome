"""Per-sample gating — dynamic dimension specs and resolvers (foundation).

See ``docs/per_sample_gating_design.md``. A gate dimension is either:

* ``absolute`` — fixed display-coordinate ``(min, max)`` (today's behaviour), or
* ``dynamic`` — resolved *per sample* by a resolver registered under a ``kind``
  (e.g. ``"time"``), so the same template adapts to each sample.

This module is intentionally free of Qt and flowkit so it can be unit-tested in
isolation and reused from any process.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Callable


@dataclass(frozen=True)
class SampleContext:
    """Per-sample information a dynamic resolver may need.

    ``channel_ranges`` maps a channel name to its ``(min, max)`` in the same
    coordinate space as gate dimensions. Resolvers read only what they need, so
    new dynamic kinds can be added without changing this structure.
    """

    channel_ranges: dict[str, tuple[float, float]] = field(default_factory=dict)


@dataclass(frozen=True)
class DimensionSpec:
    """A single gate dimension.

    ``mode == 'absolute'``: use ``(min, max)`` directly (display coordinates).
    ``mode == 'dynamic'`` : resolve ``(min, max)`` per sample via the resolver
    registered for ``kind``.
    """

    channel: str
    mode: str = "absolute"  # 'absolute' | 'dynamic'
    min: float | None = None
    max: float | None = None
    kind: str | None = None  # used when dynamic, e.g. 'time'
    params: dict = field(default_factory=dict)

    def is_dynamic(self) -> bool:
        return self.mode == "dynamic"


# --- Resolver registry -------------------------------------------------------

Resolver = Callable[[dict, SampleContext, str], tuple[float, float]]
_RESOLVERS: dict[str, Resolver] = {}


def register_dynamic_resolver(kind: str, resolver: Resolver) -> None:
    """Register a resolver for a dynamic-dimension ``kind`` (extensibility hook)."""
    _RESOLVERS[kind] = resolver


def get_dynamic_resolver(kind: str) -> Resolver:
    try:
        return _RESOLVERS[kind]
    except KeyError as exc:
        raise KeyError(
            f"No dynamic gate resolver registered for kind {kind!r}. "
            f"Registered kinds: {sorted(_RESOLVERS)}"
        ) from exc


def resolve_spec(spec: DimensionSpec, context: SampleContext) -> tuple[float, float]:
    """Resolve a dimension spec to an absolute ``(min, max)`` for one sample."""
    if not spec.is_dynamic():
        return (spec.min, spec.max)
    resolver = get_dynamic_resolver(spec.kind)
    return resolver(spec.params, context, spec.channel)


# --- Built-in resolvers ------------------------------------------------------

def _time_resolver(params: dict, context: SampleContext, channel: str) -> tuple[float, float]:
    """Resolve a Time gate expressed as fractions of the sample's own range.

    ``params``: ``{'min_frac': 0.0..1.0, 'max_frac': 0.0..1.0}``. Each sample's
    own ``(min, max)`` for ``channel`` is used, so every sample gets its own
    absolute Time window — replacing the current global-max + clamp workaround.
    """
    lo, hi = context.channel_ranges[channel]
    span = hi - lo
    min_frac = params.get("min_frac", 0.0)
    max_frac = params.get("max_frac", 1.0)
    return (lo + min_frac * span, lo + max_frac * span)


register_dynamic_resolver("time", _time_resolver)


# --- Template model + persistence (Phase 2) ---------------------------------

DEFAULT_TEMPLATE_NAME = "default"


def dynamic_key(gate_id: str, channel: str) -> str:
    """Stable key for a dynamic dimension within a template/mode."""
    return f"{gate_id}|{channel}"


@dataclass
class GatingTemplate:
    """A named gating template.

    The base hierarchy is stored as GatingML (``raw_gml`` / ``unmixed_gml``);
    ``dynamic_dimensions`` holds the specs GatingML cannot express, keyed by
    mode then ``dynamic_key(gate_id, channel)``. ``raw_plots`` / ``unmixed_plots``
    are the plot definitions that travel with this template, so a template is a
    self-contained analysis view (gates + plots) — switching a sample's template
    swaps both, keeping plots and their gate references consistent.
    """

    name: str
    raw_gml: str = ""
    unmixed_gml: str = ""
    dynamic_dimensions: dict = field(default_factory=lambda: {"raw": {}, "unmixed": {}})
    raw_plots: list = field(default_factory=list)
    unmixed_plots: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "raw_gml": self.raw_gml,
            "unmixed_gml": self.unmixed_gml,
            "dynamic_dimensions": self.dynamic_dimensions,
            "raw_plots": self.raw_plots,
            "unmixed_plots": self.unmixed_plots,
        }

    @classmethod
    def from_dict(cls, name: str, data: dict) -> "GatingTemplate":
        dyn = data.get("dynamic_dimensions") or {}
        dyn.setdefault("raw", {})
        dyn.setdefault("unmixed", {})
        return cls(
            name=name,
            raw_gml=data.get("raw_gml", ""),
            unmixed_gml=data.get("unmixed_gml", ""),
            dynamic_dimensions=dyn,
            raw_plots=data.get("raw_plots", []) or [],
            unmixed_plots=data.get("unmixed_plots", []) or [],
        )


def build_default_template(raw_gml: str, unmixed_gml: str,
                           raw_plots: list | None = None,
                           unmixed_plots: list | None = None) -> GatingTemplate:
    """Synthesize the ``default`` template from legacy experiment gating + plots.

    All dimensions stay absolute, so gating output is identical to today.
    """
    return GatingTemplate(
        name=DEFAULT_TEMPLATE_NAME,
        raw_gml=raw_gml or "",
        unmixed_gml=unmixed_gml or "",
        raw_plots=raw_plots or [],
        unmixed_plots=unmixed_plots or [],
    )


def migrate_cytometry_templates(cytometry: dict) -> dict:
    """Return the ``gating_templates`` dict for an experiment's cytometry.

    Old ``.kit`` files (no ``gating_templates``) get a single ``default``
    template synthesized from the legacy ``raw_gating`` / ``gating`` GML and the
    legacy global ``raw_plots`` / ``plots``. Templates from before plots became
    per-template are back-filled with a copy of the legacy plots so they are
    self-contained. Non-destructive: existing content is preserved.
    """
    templates = cytometry.get("gating_templates")
    legacy_raw_plots = cytometry.get("raw_plots", []) or []
    legacy_unmixed_plots = cytometry.get("plots", []) or []
    if templates:
        for template in templates.values():
            if "raw_plots" not in template:
                template["raw_plots"] = deepcopy(legacy_raw_plots)
            if "unmixed_plots" not in template:
                template["unmixed_plots"] = deepcopy(legacy_unmixed_plots)
        return templates
    default = build_default_template(
        cytometry.get("raw_gating", ""),
        cytometry.get("gating", ""),
        deepcopy(legacy_raw_plots),
        deepcopy(legacy_unmixed_plots),
    )
    return {default.name: default.to_dict()}


def template_for_sample(sample_path: str, samples: dict) -> str:
    """Template id assigned to a sample, defaulting to ``default``."""
    assignments = samples.get("sample_template_assignments", {})
    return assignments.get(sample_path, DEFAULT_TEMPLATE_NAME)


# --- Scoped templates: independent raw / unmixed template lists --------------
#
# A "scoped" template is single-scope: {"gml", "plots", "dynamic_dimensions"}.
# They live in two independent dicts so the Raw and Unmixed tabs each have their
# own template list and picker (their channels differ, so a raw plot on R1/R2 is
# meaningless in unmixed). A sample is assigned one template per scope.

RAW_TEMPLATES_KEY = "raw_gating_templates"
UNMIXED_TEMPLATES_KEY = "unmixed_gating_templates"


def _templates_key(scope: str) -> str:
    return RAW_TEMPLATES_KEY if scope == "raw" else UNMIXED_TEMPLATES_KEY


def empty_scoped_template() -> dict:
    """A new empty single-scope template (root-only gates, no plots)."""
    return {"gml": "", "plots": [], "dynamic_dimensions": {}}


def migrate_scoped_gating_templates(cytometry: dict) -> None:
    """Ensure ``raw_gating_templates`` and ``unmixed_gating_templates`` exist.

    Builds them (once) from either the older *unified* ``gating_templates``
    (each entry had both scopes) or the legacy flat fields. Non-destructive:
    runs only when the scoped dicts are absent, and back-fills missing
    ``plots`` / ``dynamic_dimensions`` keys on existing scoped templates.
    """
    have_raw = RAW_TEMPLATES_KEY in cytometry
    have_unmixed = UNMIXED_TEMPLATES_KEY in cytometry
    if have_raw and have_unmixed:
        for key in (RAW_TEMPLATES_KEY, UNMIXED_TEMPLATES_KEY):
            for tpl in cytometry[key].values():
                tpl.setdefault("gml", "")
                tpl.setdefault("plots", [])
                tpl.setdefault("dynamic_dimensions", {})
        return

    unified = cytometry.get("gating_templates")
    raw_templates: dict = {}
    unmixed_templates: dict = {}
    if unified:
        for name, tpl in unified.items():
            dyn = tpl.get("dynamic_dimensions") or {}
            raw_templates[name] = {
                "gml": tpl.get("raw_gml", ""),
                "plots": deepcopy(tpl.get("raw_plots", []) or []),
                "dynamic_dimensions": deepcopy(dyn.get("raw", {}) or {}),
            }
            unmixed_templates[name] = {
                "gml": tpl.get("unmixed_gml", ""),
                "plots": deepcopy(tpl.get("unmixed_plots", []) or []),
                "dynamic_dimensions": deepcopy(dyn.get("unmixed", {}) or {}),
            }
    else:
        raw_templates[DEFAULT_TEMPLATE_NAME] = {
            "gml": cytometry.get("raw_gating", "") or "",
            "plots": deepcopy(cytometry.get("raw_plots", []) or []),
            "dynamic_dimensions": {},
        }
        unmixed_templates[DEFAULT_TEMPLATE_NAME] = {
            "gml": cytometry.get("gating", "") or "",
            "plots": deepcopy(cytometry.get("plots", []) or []),
            "dynamic_dimensions": {},
        }
    if not raw_templates:
        raw_templates[DEFAULT_TEMPLATE_NAME] = empty_scoped_template()
    if not unmixed_templates:
        unmixed_templates[DEFAULT_TEMPLATE_NAME] = empty_scoped_template()
    cytometry[RAW_TEMPLATES_KEY] = raw_templates
    cytometry[UNMIXED_TEMPLATES_KEY] = unmixed_templates


def scoped_template_for_sample(scope: str, sample_path: str, samples: dict) -> str:
    """Template assigned to a sample for ``scope`` ('raw'/'unmixed').

    Assignments are ``{sample_path: {'raw': name, 'unmixed': name}}``; a missing
    entry (or old flat ``{sample_path: name}`` value) defaults to ``default``.
    """
    assignments = samples.get("sample_template_assignments", {})
    entry = assignments.get(sample_path)
    if isinstance(entry, dict):
        return entry.get(scope, DEFAULT_TEMPLATE_NAME)
    return DEFAULT_TEMPLATE_NAME

