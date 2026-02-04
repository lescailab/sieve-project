#!/usr/bin/env python3
"""
Tests for null baseline functionality.
"""

import tempfile
from pathlib import Path

import torch
import pytest


def test_permutation_preserves_label_distribution():
    """Verify that permutation preserves case/control counts."""
    # Import within test to avoid import errors if module not installed
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from scripts.create_null_baseline import create_single_permutation

    # Create mock data
    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = Path(tmpdir) / "input.pt"
        output_path = Path(tmpdir) / "output.pt"

        # Mock preprocessed data
        n_samples = 100
        labels = torch.cat([torch.ones(40), torch.zeros(60)]).long()
        mock_data = {
            'labels': labels,
            'features': torch.randn(n_samples, 10),
        }
        torch.save(mock_data, input_path)

        # Run permutation
        stats = create_single_permutation(str(input_path), str(output_path), seed=42)

        # Load and check
        permuted = torch.load(output_path, weights_only=False)

        assert (permuted['labels'] == 1).sum() == 40, "Case count changed"
        assert (permuted['labels'] == 0).sum() == 60, "Control count changed"
        assert stats['n_cases'] == 40
        assert stats['n_controls'] == 60


def test_permutation_is_reproducible():
    """Verify that same seed produces same permutation."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from scripts.create_null_baseline import create_single_permutation

    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = Path(tmpdir) / "input.pt"
        output1_path = Path(tmpdir) / "output1.pt"
        output2_path = Path(tmpdir) / "output2.pt"

        labels = torch.randint(0, 2, (50,))
        mock_data = {'labels': labels}
        torch.save(mock_data, input_path)

        create_single_permutation(str(input_path), str(output1_path), seed=123)
        create_single_permutation(str(input_path), str(output2_path), seed=123)

        perm1 = torch.load(output1_path, weights_only=False)['labels']
        perm2 = torch.load(output2_path, weights_only=False)['labels']

        assert torch.equal(perm1, perm2), "Same seed should produce same permutation"


def test_different_seeds_produce_different_permutations():
    """Verify that different seeds produce different permutations."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from scripts.create_null_baseline import create_single_permutation

    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = Path(tmpdir) / "input.pt"
        output1_path = Path(tmpdir) / "output1.pt"
        output2_path = Path(tmpdir) / "output2.pt"

        labels = torch.randint(0, 2, (100,))
        mock_data = {'labels': labels}
        torch.save(mock_data, input_path)

        create_single_permutation(str(input_path), str(output1_path), seed=1)
        create_single_permutation(str(input_path), str(output2_path), seed=2)

        perm1 = torch.load(output1_path, weights_only=False)['labels']
        perm2 = torch.load(output2_path, weights_only=False)['labels']

        assert not torch.equal(perm1, perm2), "Different seeds should produce different permutations"


def test_permutation_metadata_is_saved():
    """Verify that null baseline metadata is saved correctly."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from scripts.create_null_baseline import create_single_permutation

    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = Path(tmpdir) / "input.pt"
        output_path = Path(tmpdir) / "output.pt"

        labels = torch.randint(0, 2, (50,))
        mock_data = {'labels': labels}
        torch.save(mock_data, input_path)

        create_single_permutation(str(input_path), str(output_path), seed=999)

        # Load and check metadata
        permuted = torch.load(output_path, weights_only=False)

        assert '_null_baseline_metadata' in permuted
        metadata = permuted['_null_baseline_metadata']

        assert metadata['is_null_baseline'] is True
        assert metadata['permutation_seed'] == 999
        assert metadata['original_path'] == str(input_path)
        assert metadata['n_samples'] == 50


def test_permutation_works_with_sample_objects():
    """Test permutation with samples stored as list of dicts (realistic case)."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from scripts.create_null_baseline import create_single_permutation

    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = Path(tmpdir) / "input.pt"
        output_path = Path(tmpdir) / "output.pt"

        # Mock data with samples as list of dicts (more realistic)
        samples = [
            {'sample_id': f'S{i}', 'label': i % 2, 'variants': []}
            for i in range(20)
        ]
        mock_data = {'samples': samples}
        torch.save(mock_data, input_path)

        create_single_permutation(str(input_path), str(output_path), seed=42)

        # Load and check
        permuted = torch.load(output_path, weights_only=False)

        # Check that labels are still 0 or 1 and counts are preserved
        permuted_labels = [s['label'] for s in permuted['samples']]
        original_labels = [s['label'] for s in samples]

        assert sum(permuted_labels) == sum(original_labels), "Label counts changed"
        assert all(l in [0, 1] for l in permuted_labels), "Invalid labels"

        # Check that at least some labels changed positions
        n_changed = sum(1 for i, (p, o) in enumerate(zip(permuted_labels, original_labels)) if p != o)
        assert n_changed > 0, "No labels changed position (unlikely with n=20, seed=42)"


def test_multiple_permutations_creates_files():
    """Test that multiple permutations creates the expected files."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from scripts.create_null_baseline import create_multiple_permutations

    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = Path(tmpdir) / "input.pt"
        output_dir = Path(tmpdir) / "permutations"

        labels = torch.randint(0, 2, (30,))
        mock_data = {'labels': labels}
        torch.save(mock_data, input_path)

        n_perms = 3
        all_stats = create_multiple_permutations(
            str(input_path),
            str(output_dir),
            n_permutations=n_perms,
            base_seed=100
        )

        # Check that all files were created
        assert len(all_stats) == n_perms
        for i in range(n_perms):
            perm_file = output_dir / f"preprocessed_NULL_perm{i}.pt"
            assert perm_file.exists(), f"Permutation {i} file not created"

            # Check that seeds are different
            assert all_stats[i]['seed'] == 100 + i

        # Check summary file
        summary_file = output_dir / "permutation_summary.txt"
        assert summary_file.exists(), "Summary file not created"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
