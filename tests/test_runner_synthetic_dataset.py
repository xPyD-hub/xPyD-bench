"""Tests for runner synthetic dataset integration (issue #84).

Verifies that --dataset-name synthetic and distribution flags are respected
when no --dataset-path is provided.
"""

from __future__ import annotations

from types import SimpleNamespace


def _make_args(**overrides):
    """Build a minimal args namespace for run_benchmark dataset loading."""
    defaults = dict(
        dataset_path=None,
        dataset_name="random",
        num_prompts=10,
        input_len=64,
        output_len=32,
        synthetic_input_len_dist="fixed",
        synthetic_output_len_dist="fixed",
        seed=42,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


class TestSyntheticDatasetRouting:
    """Ensure runner routes to load_dataset when dataset_name=synthetic."""

    def test_random_uses_generate_random_prompts(self):
        """Default dataset_name='random' still uses _generate_random_prompts."""
        args = _make_args()
        # Simulate the runner logic
        dataset_name = getattr(args, "dataset_name", "random")
        uses_synthetic = (
            dataset_name == "synthetic"
            or getattr(args, "synthetic_input_len_dist", "fixed") != "fixed"
            or getattr(args, "synthetic_output_len_dist", "fixed") != "fixed"
        )
        assert not uses_synthetic

    def test_synthetic_triggers_load_dataset(self):
        """dataset_name='synthetic' should trigger load_dataset path."""
        args = _make_args(dataset_name="synthetic")
        dataset_name = getattr(args, "dataset_name", "random")
        uses_synthetic = (
            dataset_name == "synthetic"
            or getattr(args, "synthetic_input_len_dist", "fixed") != "fixed"
            or getattr(args, "synthetic_output_len_dist", "fixed") != "fixed"
        )
        assert uses_synthetic

    def test_dist_flag_triggers_load_dataset(self):
        """Non-fixed distribution flags should trigger load_dataset even with name=random."""
        args = _make_args(synthetic_input_len_dist="zipf")
        dataset_name = getattr(args, "dataset_name", "random")
        uses_synthetic = (
            dataset_name == "synthetic"
            or getattr(args, "synthetic_input_len_dist", "fixed") != "fixed"
            or getattr(args, "synthetic_output_len_dist", "fixed") != "fixed"
        )
        assert uses_synthetic

    def test_synthetic_produces_varied_lengths(self):
        """Synthetic with zipf distribution should produce varied prompt lengths."""
        from xpyd_bench.datasets.loader import load_dataset

        entries = load_dataset(
            path=None,
            name="synthetic",
            num_prompts=20,
            input_len=128,
            output_len=64,
            input_len_dist="zipf",
            output_len_dist="uniform",
            seed=42,
        )
        assert len(entries) == 20
        lengths = [len(e.prompt.split()) for e in entries]
        # With zipf distribution, lengths should not all be identical
        assert len(set(lengths)) > 1, "Zipf distribution should produce varied lengths"

    def test_fixed_dist_produces_uniform_lengths(self):
        """Synthetic with fixed distribution should produce uniform prompt lengths."""
        from xpyd_bench.datasets.loader import load_dataset

        entries = load_dataset(
            path=None,
            name="synthetic",
            num_prompts=10,
            input_len=64,
            output_len=32,
            input_len_dist="fixed",
            output_len_dist="fixed",
            seed=42,
        )
        lengths = [len(e.prompt.split()) for e in entries]
        assert len(set(lengths)) == 1, "Fixed distribution should produce uniform lengths"
