import unittest
import torch

from sparse_bench.digital_twin import DualDigitalTwin, TwinConfig, train_digital_twin


class DigitalTwinTests(unittest.TestCase):
    def test_transfer_shape_mismatch_is_not_silent(self):
        with self.assertRaises(ValueError):
            DualDigitalTwin(12, TwinConfig(dim=8), torch.zeros(12, 7))

    def test_state_is_entity_bound_versioned_and_branch_isolated(self):
        model = DualDigitalTwin(12, TwinConfig(dim=8, epochs=1))
        state = model.user.synchronize("u-1", [1, 2, 3], torch.device("cpu"))
        branch = state.branch()
        branch.belief.add_(1)
        self.assertEqual(state.entity_id, "u-1")
        self.assertEqual(state.observations, 3)
        self.assertEqual(state.version, 1)
        self.assertFalse(torch.equal(state.belief, branch.belief))

    def test_interventions_create_distinct_counterfactuals(self):
        model = DualDigitalTwin(12, TwinConfig(dim=8, rollout_horizon=2))
        state = model.user.synchronize("u", [1, 2], torch.device("cpu"))
        a = model.counterfactual_value(state.branch(), 3)
        b = model.counterfactual_value(state.branch(), 4)
        self.assertNotEqual(a, b)

    def test_training_and_reranking_contract(self):
        cfg = TwinConfig(dim=8, epochs=1, batch_size=2, rollout_horizon=2)
        model = train_digital_twin([[1, 2, 3], [2, 3, 4], [3, 5, 6]], 10, cfg, "cpu")
        ranked = model.rerank("u", [1, 2], [(3, 0.9), (4, 0.8), (5, 0.7)])
        self.assertEqual({x for x, _ in ranked}, {3, 4, 5})
        self.assertEqual(len(ranked), 3)

    def test_training_objective_reaches_deployed_assimilator(self):
        model = DualDigitalTwin(12, TwinConfig(dim=8))
        model.sequence_loss([1, 2, 3], torch.device("cpu")).backward()
        grads = [p.grad for p in model.user.assimilator.parameters()]
        self.assertTrue(all(g is not None for g in grads))
        self.assertGreater(sum(float(g.abs().sum()) for g in grads), 0.0)

    def test_training_calibrates_scale_and_deployed_uncertainty(self):
        model = DualDigitalTwin(12, TwinConfig(dim=8))
        model.sequence_loss([1, 2, 3], torch.device("cpu")).backward()
        self.assertIsNotNone(model.logit_scale.grad)
        self.assertGreater(float(model.logit_scale.grad.abs()), 0.0)
        grads = [p.grad for p in model.user.log_variance.parameters()]
        self.assertTrue(all(g is not None for g in grads))
        self.assertGreater(sum(float(g.abs().sum()) for g in grads), 0.0)


if __name__ == "__main__":
    unittest.main()
