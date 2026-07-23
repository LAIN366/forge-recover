"""Bayesian evidence fusion for belief-aware manipulation task nodes."""

from dataclasses import dataclass


_EPSILON = 1e-6


def _probability(value):
    return min(1.0 - _EPSILON, max(_EPSILON, float(value)))


@dataclass(frozen=True)
class BeliefEvidence:
    source: str
    likelihood_if_valid: float
    likelihood_if_invalid: float

    def __post_init__(self):
        object.__setattr__(
            self, "likelihood_if_valid", _probability(self.likelihood_if_valid)
        )
        object.__setattr__(
            self, "likelihood_if_invalid", _probability(self.likelihood_if_invalid)
        )


class BayesianBeliefUpdater:
    """Fuse independent observations while retaining an auditable trace."""

    def __init__(self, prior=0.5, forgetting=0.0):
        self.prior = _probability(prior)
        self.forgetting = min(1.0, max(0.0, float(forgetting)))
        self.posterior = self.prior
        self.trace = []

    def reset(self, prior=None):
        if prior is not None:
            self.prior = _probability(prior)
        self.posterior = self.prior
        self.trace.clear()

    def update(self, evidence):
        prior = (
            (1.0 - self.forgetting) * self.posterior
            + self.forgetting * self.prior
        )
        numerator = prior * evidence.likelihood_if_valid
        denominator = numerator + (
            (1.0 - prior) * evidence.likelihood_if_invalid
        )
        self.posterior = _probability(numerator / max(denominator, _EPSILON))
        self.trace.append({
            "source": evidence.source,
            "prior": prior,
            "posterior": self.posterior,
            "likelihood_if_valid": evidence.likelihood_if_valid,
            "likelihood_if_invalid": evidence.likelihood_if_invalid,
        })
        return self.posterior

    def fuse(self, evidence_items):
        for evidence in evidence_items:
            self.update(evidence)
        return self.posterior


def confidence_evidence(confidence, source="rgbd", false_positive_rate=0.12):
    """Convert a calibrated detector confidence into Bayesian evidence."""
    return BeliefEvidence(
        source=source,
        likelihood_if_valid=_probability(confidence),
        likelihood_if_invalid=_probability(false_positive_rate),
    )


def binary_evidence(observed, source, true_positive=0.97, false_positive=0.08):
    """Create evidence from a contact, tracking, or constraint check."""
    if observed:
        return BeliefEvidence(source, true_positive, false_positive)
    return BeliefEvidence(source, 1.0 - true_positive, 1.0 - false_positive)
