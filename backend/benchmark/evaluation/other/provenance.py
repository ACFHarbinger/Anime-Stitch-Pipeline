"""Case-level provenance and content-safety schema (M0 — issue #24;
C0.5 dual-veto — issue #41).

Distinct from ``other/schema.py``'s ``RatingEntry``: a ``RatingEntry`` scores
a *comparison* (asp vs. simple, per-output dimensions/defects); a
``CaseProvenance`` describes what a case *is* (source, licence, content
safety) — case-level, referenced by whichever output artifacts exist for
that case, never a second evaluation schema. See
``asp_sfw_corpus_roadmap_2026q3.md`` §3 and §C0.5.

**Every safety observation is append-only.** The dual-veto gate depends on
this: a later adjudication must never silently overwrite an earlier
observation, or the audit trail C0.5's adversarial re-audit needs to catch
labeling-pipeline drift at scale would be lying about its own history.
"""

from __future__ import annotations

import dataclasses
import datetime
import json
import os

from ..constants.schema import (
    CONTENT_TAGS,
    MINOR_RISK_CLEAR,
    MINOR_RISK_HIGH_RISK,
    MINOR_RISK_VERDICTS,
    SAFETY_TIERS,
)

# The one source whose "clear" verdict is load-bearing for inclusion, per
# C0.5's decision rule: "Inclusion in dump_sfw/ requires a human clear and
# no unresolved high-likelihood high_risk." Automated ensemble members can
# contribute a high_risk veto (once validated) but cannot clear a case on
# their own — see the decision-rule table in the roadmap.
SOURCE_HUMAN = "human"


def _now() -> str:
    return datetime.datetime.now().isoformat(timespec="seconds")


@dataclasses.dataclass
class SafetyObservation:
    """One independent vote — human or automated — never rewritten, only
    appended to. ``source`` identifies the assessor (e.g. ``"human"``,
    ``"board_tags"``, ``"wd14"``, ``"clip_dinov2"``, a named commercial
    endpoint); ``model_or_rule_id`` pins the exact version so an observation
    stays interpretable after the assessor itself changes."""

    source: str
    verdict: str  # MINOR_RISK_VERDICTS
    evidence: str = ""
    model_or_rule_id: str = ""
    policy_version: str = ""
    observed_at: str = dataclasses.field(default_factory=_now)

    def __post_init__(self) -> None:
        if self.verdict not in MINOR_RISK_VERDICTS:
            raise ValueError(f"unknown minor-presenting verdict: {self.verdict!r}")

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @staticmethod
    def from_dict(d: dict) -> SafetyObservation:
        return SafetyObservation(
            source=d["source"],
            verdict=d["verdict"],
            evidence=d.get("evidence", ""),
            model_or_rule_id=d.get("model_or_rule_id", ""),
            policy_version=d.get("policy_version", ""),
            observed_at=d.get("observed_at", ""),
        )


@dataclasses.dataclass
class SafetyAdjudication:
    """A separately-stored, reasoned effective decision over one or more
    observations — appended, never a destructive replacement for the
    observations it's based on."""

    decision: str  # MINOR_RISK_VERDICTS — the resolved verdict
    reason: str = ""
    adjudicated_by: str = ""
    adjudicated_at: str = dataclasses.field(default_factory=_now)

    def __post_init__(self) -> None:
        if self.decision not in MINOR_RISK_VERDICTS:
            raise ValueError(f"unknown minor-presenting verdict: {self.decision!r}")

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @staticmethod
    def from_dict(d: dict) -> SafetyAdjudication:
        return SafetyAdjudication(
            decision=d["decision"],
            reason=d.get("reason", ""),
            adjudicated_by=d.get("adjudicated_by", ""),
            adjudicated_at=d.get("adjudicated_at", ""),
        )


@dataclasses.dataclass
class CaseProvenance:
    """Case-level fields for one ``asp_testNN``-style case, referenced by
    (not owned by) whichever raw_asp/safe_asp/scans artifacts exist for it.

    ``source_work_nsfw`` is deliberately a three-state field (``True`` /
    ``False`` / ``None`` for unknown) distinct from ``safety_tier`` — a
    case-level-SFW image can still come from a source series that's NSFW
    overall (series-taint), and collapsing that into ``tier_nsfw`` or a
    dual-veto ``high_risk`` would conflate two different policy questions.
    """

    case_id: str
    corpus_id: str = ""  # e.g. "nsfw_97", "sfw_q3"
    source_url: str = ""
    source_board: str = ""
    licence: str = ""
    web_redistribution_ok: bool = False
    source_work_nsfw: bool | None = None  # None = unknown, not False
    content_tags: list[str] = dataclasses.field(default_factory=list)
    safety_tier: str | None = None
    policy_version: str = ""
    safety_observations: list[SafetyObservation] = dataclasses.field(default_factory=list)
    safety_adjudications: list[SafetyAdjudication] = dataclasses.field(default_factory=list)
    gt_known_defects: list[str] = dataclasses.field(default_factory=list)
    updated_at: str = ""

    def __post_init__(self) -> None:
        if self.safety_tier is not None and self.safety_tier not in SAFETY_TIERS:
            raise ValueError(f"unknown safety_tier: {self.safety_tier!r}")
        unknown_tags = set(self.content_tags) - set(CONTENT_TAGS)
        # content_tags is deliberately open-ended (roadmap: "others added as
        # needed") -- don't reject an unrecognized tag, a human may be ahead
        # of this list. Nothing to validate here beyond type; kept as an
        # explicit no-op so the intent isn't mistaken for an oversight.
        del unknown_tags

    # -- dual-veto gate -------------------------------------------------

    def add_observation(self, observation: SafetyObservation) -> None:
        """Append-only, per the dual-veto's audit-trail requirement."""
        self.safety_observations.append(observation)
        self.touch()

    def add_adjudication(self, adjudication: SafetyAdjudication) -> None:
        self.safety_adjudications.append(adjudication)
        self.touch()

    def minor_presenting_high_risk(self) -> bool:
        """OR-logic for exclusion: True if *any* observation independently
        flags high_risk. A hard drop per C0.5 -- callers must exclude the
        case entirely, not demote its tier."""
        return any(
            obs.verdict == MINOR_RISK_HIGH_RISK for obs in self.safety_observations
        )

    def minor_presenting_includable(self) -> bool:
        """AND-logic for acceptance, per C0.5's decision rule: inclusion
        requires no unresolved high_risk from any source, AND a human
        `clear` observation specifically -- an automated ensemble member
        cannot clear a case on its own, only veto or abstain (`uncertain`).
        """
        if self.minor_presenting_high_risk():
            return False
        return any(
            obs.source == SOURCE_HUMAN and obs.verdict == MINOR_RISK_CLEAR
            for obs in self.safety_observations
        )

    def touch(self) -> None:
        self.updated_at = _now()

    # -- serialization ----------------------------------------------------

    def to_dict(self) -> dict:
        doc: dict = {"case_id": self.case_id}
        if self.corpus_id:
            doc["corpus_id"] = self.corpus_id
        if self.source_url:
            doc["source_url"] = self.source_url
        if self.source_board:
            doc["source_board"] = self.source_board
        if self.licence:
            doc["licence"] = self.licence
        if self.web_redistribution_ok:
            doc["web_redistribution_ok"] = True
        if self.source_work_nsfw is not None:
            doc["source_work_nsfw"] = self.source_work_nsfw
        if self.content_tags:
            doc["content_tags"] = sorted(set(self.content_tags))
        if self.safety_tier is not None:
            doc["safety_tier"] = self.safety_tier
        if self.policy_version:
            doc["policy_version"] = self.policy_version
        if self.safety_observations:
            doc["safety_observations"] = [o.to_dict() for o in self.safety_observations]
        if self.safety_adjudications:
            doc["safety_adjudications"] = [a.to_dict() for a in self.safety_adjudications]
        if self.gt_known_defects:
            doc["gt_known_defects"] = sorted(set(self.gt_known_defects))
        if self.updated_at:
            doc["updated_at"] = self.updated_at
        return doc

    @staticmethod
    def from_dict(d: dict) -> CaseProvenance:
        return CaseProvenance(
            case_id=d["case_id"],
            corpus_id=d.get("corpus_id", ""),
            source_url=d.get("source_url", ""),
            source_board=d.get("source_board", ""),
            licence=d.get("licence", ""),
            web_redistribution_ok=bool(d.get("web_redistribution_ok", False)),
            source_work_nsfw=d.get("source_work_nsfw"),
            content_tags=[t for t in d.get("content_tags", []) if isinstance(t, str)],
            safety_tier=d.get("safety_tier"),
            policy_version=d.get("policy_version", ""),
            safety_observations=[
                SafetyObservation.from_dict(o) for o in d.get("safety_observations", [])
            ],
            safety_adjudications=[
                SafetyAdjudication.from_dict(a) for a in d.get("safety_adjudications", [])
            ],
            gt_known_defects=[
                t for t in d.get("gt_known_defects", []) if isinstance(t, str)
            ],
            updated_at=d.get("updated_at", ""),
        )


def load_provenance(path: str) -> dict[str, CaseProvenance]:
    if not os.path.exists(path):
        return {}
    with open(path) as fh:
        raw = json.load(fh)
    return {name: CaseProvenance.from_dict(entry) for name, entry in raw.items()}


def save_provenance(path: str, provenance: dict[str, CaseProvenance]) -> None:
    doc = {name: entry.to_dict() for name, entry in provenance.items()}
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w") as fh:
        json.dump(doc, fh, indent=2, sort_keys=True)
    os.replace(tmp_path, path)  # atomic -- a mid-write crash never corrupts the real file
