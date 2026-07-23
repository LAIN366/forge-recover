"""Tests for Bayesian task-node belief updates."""

import unittest

from cruzr_sim.tasks.belief_update import (
    BayesianBeliefUpdater,
    binary_evidence,
    confidence_evidence,
)


class BayesianBeliefUpdaterTest(unittest.TestCase):
    def test_consistent_multimodal_evidence_raises_belief(self):
        updater = BayesianBeliefUpdater(prior=0.5)
        posterior = updater.fuse([
            confidence_evidence(0.9),
            binary_evidence(True, "depth_valid"),
            binary_evidence(True, "contact_consistent"),
        ])
        self.assertGreater(posterior, 0.98)
        self.assertEqual(len(updater.trace), 3)

    def test_negative_contact_evidence_reduces_belief(self):
        updater = BayesianBeliefUpdater(prior=0.8)
        posterior = updater.update(binary_evidence(False, "contact"))
        self.assertLess(posterior, 0.3)


if __name__ == "__main__":
    unittest.main()
