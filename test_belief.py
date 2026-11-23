import tempfile
import unittest
from pathlib import Path
import numpy as np
from loki_memory import BeliefModel


class BeliefTests(unittest.TestCase):
    def test_ambiguous_consequence_survives_save_and_continued_learning(self):
        x, y, z = np.random.default_rng(18).random((3, 24), dtype="f4")
        model = BeliefModel()
        i, j = model.fit(x, 1, y)
        _, k = model.fit(x, 1, z)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model"
            model.save(path)
            size = path.stat().st_size
            restored = BeliefModel.load(path)
            np.testing.assert_array_equal(
                restored.probabilities(), model.probabilities()
            )
            np.testing.assert_allclose(
                restored.probabilities()[1, i, [j, k]], [0.5, 0.5]
            )
            for instance in (model, restored):
                instance.fit(x, 1, z)
            np.testing.assert_array_equal(restored.theta, model.theta)
            restored.save(path)
            self.assertEqual(path.stat().st_size, size)
            self.assertLessEqual(size, 256 * 1024)

    def test_inferred_reverse_rows_have_unit_mass(self):
        model = BeliefModel()
        model.n = 12
        for i in range(8):
            model.count[0, i, i + 1] = 1
            model.count[1, i + 1, i] = 1
        model.count[1, 1, 3] = 1
        model.count[1, 10, 9] = 1
        probabilities = model.probabilities()
        self.assertAlmostEqual(float(probabilities[0, 9, 10]), 1)
        np.testing.assert_allclose(probabilities.sum(2), 1, atol=1e-6)

    def test_invalid_command_does_not_change_memory(self):
        model = BeliefModel()
        x = np.zeros(24)
        with self.assertRaises(ValueError):
            model.fit(x, -1, x)
        self.assertEqual(model.n, 0)

    def test_imagination_is_read_only_and_survives_restart(self):
        rng = np.random.default_rng(921)
        model = BeliefModel()
        x = rng.random(24)
        for a in range(12):
            y = np.clip(x + 0.04, 0, 1)
            model.fit(x, a % 4, y)
            x = y
        previous = np.eye(4)[3]
        commands = [2, 0, 1, 2]
        before = {
            k: v.copy() for k, v in vars(model).items() if isinstance(v, np.ndarray)
        }
        trajectory = model.imagine(x, commands, previous)
        self.assertEqual(trajectory.shape, (4, 24))
        for k, v in before.items():
            np.testing.assert_array_equal(getattr(model, k), v)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state"
            model.save(path)
            np.testing.assert_array_equal(
                BeliefModel.load(path).imagine(x, commands, previous), trajectory
            )

    def test_compact_decision_including_ambiguous_dense_rows(self):
        import copy
        from importlib.util import find_spec

        if find_spec("numba") is None:
            self.skipTest("optional compact backend")
        rng = np.random.default_rng(997)
        model = BeliefModel(16)
        model.n = 20
        model.emission[:20] = rng.random((20, 24))
        model.emission[1] = model.emission[0]
        model.count[:, :20, :20] = rng.integers(4, size=(4, 20, 20))
        for step in range(8):
            belief = model.infer(model.emission[step])
            goal = model.emission[step + 4]
            reference = copy.deepcopy(model)
            q, t = reference.policy(goal)
            expected = reference.choose(belief, q, t)
            action, prior = model.compact_step(belief, goal)
            self.assertEqual(action, expected)
            np.testing.assert_allclose(prior, belief @ t[action], atol=1e-6)
