"""
Unit tests for Phase 3 explainability components.
"""
import pytest
import torch
import numpy as np
from pathlib import Path

from src.encoding import VariantDataset, collate_samples, AnnotationLevel
from src.models.sieve import create_sieve_model
from src.explain.gradients import IntegratedGradientsExplainer
from src.explain.attention_analysis import AttentionAnalyzer
from src.explain.variant_ranking import VariantRanker
from src.explain.shap_epistasis import SHAPEpistasisDetector


@pytest.fixture
def test_data_dir():
    """Path to test data directory."""
    return Path(__file__).parent.parent / "test_data" / "small"


@pytest.fixture
def preprocessed_data(test_data_dir):
    """Load preprocessed test data."""
    data_path = test_data_dir / "preprocessed_test.pt"
    if not data_path.exists():
        pytest.skip(f"Test data not found: {data_path}")
    return torch.load(data_path, weights_only=False)


@pytest.fixture
def test_checkpoint(test_data_dir):
    """Load test model checkpoint."""
    checkpoint_path = test_data_dir / "test_model" / "L3_run" / "fold_0" / "best_model.pt"
    if not checkpoint_path.exists():
        pytest.skip(f"Test checkpoint not found: {checkpoint_path}")
    return torch.load(checkpoint_path, weights_only=False, map_location='cpu')


@pytest.fixture
def test_dataset(preprocessed_data):
    """Create test dataset."""
    samples = preprocessed_data['samples']
    return VariantDataset(samples, annotation_level=AnnotationLevel.L3)


@pytest.fixture
def test_model(test_checkpoint, test_dataset):
    """Create and load test model."""
    config = {
        'input_dim': 71,
        'latent_dim': 64,
        'hidden_dim': 32,
        'num_heads': 2,
        'num_attention_layers': 1,
        'dropout': 0.1,
    }
    model = create_sieve_model(config, num_genes=test_dataset.num_genes)
    model.load_state_dict(test_checkpoint['model_state_dict'])
    model.eval()
    return model


class TestIntegratedGradients:
    """Test integrated gradients explainer."""

    def test_initialization(self, test_model):
        """Test explainer initialization."""
        explainer = IntegratedGradientsExplainer(test_model, device='cpu', n_steps=5)
        assert explainer.model is test_model
        assert explainer.device == 'cpu'
        assert explainer.n_steps == 5

    def test_attribute_single_sample(self, test_model, test_dataset):
        """Test attribution for a single sample."""
        explainer = IntegratedGradientsExplainer(test_model, device='cpu', n_steps=5)

        # Get a sample
        sample = test_dataset[0]
        features = sample['features'].unsqueeze(0)
        positions = sample['positions'].unsqueeze(0)
        gene_ids = sample['gene_ids'].unsqueeze(0)
        mask = sample['mask'].unsqueeze(0)

        # Compute attributions
        attributions = explainer.attribute(features, positions, gene_ids, mask)

        # Check output shape
        assert attributions.shape == features.shape
        assert not torch.isnan(attributions).any()

    def test_attribute_batch(self, test_model, test_dataset):
        """Test batch attribution."""
        from torch.utils.data import DataLoader

        explainer = IntegratedGradientsExplainer(test_model, device='cpu', n_steps=5)
        dataloader = DataLoader(
            test_dataset,
            batch_size=4,
            collate_fn=collate_samples,
            shuffle=False
        )

        all_attrs, all_scores, all_meta = explainer.attribute_batch(
            dataloader, aggregate='l2'
        )

        # Check we got results for all samples
        assert len(all_attrs) == len(test_dataset)
        assert len(all_scores) == len(test_dataset)
        assert len(all_meta) == len(test_dataset)

        # Check shapes and values
        for attrs, scores in zip(all_attrs, all_scores):
            assert attrs.ndim == 2  # (n_variants, n_features)
            assert scores.ndim == 1  # (n_variants,)
            assert len(attrs) == len(scores)
            assert not np.isnan(attrs).any()
            assert not np.isnan(scores).any()


class TestAttentionAnalysis:
    """Test attention pattern analysis."""

    def test_initialization(self, test_model):
        """Test analyzer initialization."""
        analyzer = AttentionAnalyzer(test_model, device='cpu')
        assert analyzer.model is not None

    def test_extract_attention(self, test_model, test_dataset):
        """Test attention extraction."""
        analyzer = AttentionAnalyzer(test_model, device='cpu')

        # Get a sample
        sample = test_dataset[0]
        features = sample['features'].unsqueeze(0)
        positions = sample['positions'].unsqueeze(0)
        gene_ids = sample['gene_ids'].unsqueeze(0)
        mask = sample['mask'].unsqueeze(0)

        # Extract attention weights
        attention_weights = analyzer.extract_attention_weights(
            features, positions, gene_ids, mask
        )

        # Should return a list of attention weight tensors
        assert isinstance(attention_weights, list)
        if len(attention_weights) > 0:
            # Each element should be a tensor
            assert isinstance(attention_weights[0], torch.Tensor)
            # Find interactions
            interactions = analyzer.find_top_interactions(
                attention_weights, positions, gene_ids, mask, top_k=10
            )
            assert isinstance(interactions, list)


class TestVariantRanking:
    """Test variant ranking."""

    def test_rank_variants(self):
        """Test variant ranking."""
        ranker = VariantRanker()

        # Create dummy attribution data (chromosomes now required)
        all_scores = [
            np.array([0.5, 0.3, 0.8]),
            np.array([0.2, 0.9]),
        ]
        all_meta = [
            {
                'positions': np.array([100, 200, 300]),
                'gene_ids': np.array([0, 1, 0]),
                'chromosomes': np.array(['1', '1', '1']),
                'label': 1
            },
            {
                'positions': np.array([100, 400]),
                'gene_ids': np.array([0, 2]),
                'chromosomes': np.array(['1', '2']),
                'label': 0
            }
        ]

        rankings = ranker.rank_variants(all_scores, all_meta)

        # Check output structure (check only fields that are actually returned)
        assert 'position' in rankings.columns
        assert 'gene_id' in rankings.columns
        assert 'mean_attribution' in rankings.columns
        assert 'num_samples' in rankings.columns
        assert 'chromosome' in rankings.columns

        # Check we got all unique (chrom, pos, gene) combinations
        unique_keys = set()
        for meta in all_meta:
            for chrom, pos, gene in zip(meta['chromosomes'], meta['positions'], meta['gene_ids']):
                unique_keys.add((chrom, pos, gene))
        assert len(rankings) == len(unique_keys)

    def test_rank_variants_chromosome_collision_prevention(self):
        """Test that same position on different chromosomes creates separate entries."""
        ranker = VariantRanker()

        # Create data where position 100 with gene 0 exists on BOTH chr1 and chrX
        # This would have caused a collision with the old (pos, gene) key
        all_scores = [
            np.array([0.9, 0.1]),  # High score for chr1:100, low for chrX:100
        ]
        all_meta = [
            {
                'positions': np.array([100, 100]),  # Same position!
                'gene_ids': np.array([0, 0]),        # Same gene!
                'chromosomes': np.array(['1', 'X']), # Different chromosomes
                'label': 1
            },
        ]

        rankings = ranker.rank_variants(all_scores, all_meta)

        # CRITICAL: We should get TWO rows, not one
        assert len(rankings) == 2, (
            f"Expected 2 rows for same pos/gene on different chromosomes, got {len(rankings)}"
        )

        # Verify both chromosomes are present
        chroms = set(rankings['chromosome'].tolist())
        assert chroms == {'1', 'X'}, f"Expected chromosomes {{'1', 'X'}}, got {chroms}"

        # Verify the attribution scores are correct (not merged)
        chr1_row = rankings[rankings['chromosome'] == '1'].iloc[0]
        chrX_row = rankings[rankings['chromosome'] == 'X'].iloc[0]
        assert abs(chr1_row['mean_attribution'] - 0.9) < 0.01, "Chr1 attribution should be 0.9"
        assert abs(chrX_row['mean_attribution'] - 0.1) < 0.01, "ChrX attribution should be 0.1"

    def test_rank_variants_missing_chromosomes_raises_error(self):
        """Test that missing chromosomes in metadata raises ValueError."""
        ranker = VariantRanker()

        all_scores = [np.array([0.5])]
        all_meta = [
            {
                'positions': np.array([100]),
                'gene_ids': np.array([0]),
                # NO 'chromosomes' key - should raise error
                'label': 1
            },
        ]

        with pytest.raises(ValueError, match="chromosomes"):
            ranker.rank_variants(all_scores, all_meta)

    def test_rank_genes(self):
        """Test gene ranking."""
        ranker = VariantRanker()

        # Create dummy variant rankings
        import pandas as pd
        variant_rankings = pd.DataFrame({
            'position': [100, 200, 300, 400],
            'gene_id': [0, 1, 0, 2],
            'mean_attribution': [0.5, 0.3, 0.8, 0.2],
            'num_samples': [2, 1, 1, 1]
        })

        gene_rankings = ranker.rank_genes(variant_rankings)

        # Check output structure
        assert 'gene_id' in gene_rankings.columns
        assert 'num_variants' in gene_rankings.columns
        assert 'gene_score' in gene_rankings.columns
        assert 'top_variant_pos' in gene_rankings.columns

        # Check gene with 2 variants has higher count
        gene0 = gene_rankings[gene_rankings['gene_id'] == 0].iloc[0]
        assert gene0['num_variants'] == 2


class TestSHAPEpistasis:
    """Test SHAP epistasis detection."""

    def test_initialization(self, test_model):
        """Test detector initialization."""
        detector = SHAPEpistasisDetector(test_model, device='cpu')
        assert detector.model is test_model
        assert detector.device == 'cpu'

    def test_validate_interaction(self, test_model, test_dataset):
        """Test counterfactual perturbation."""
        detector = SHAPEpistasisDetector(test_model, device='cpu')

        # Get a sample with multiple variants
        sample = test_dataset[0]
        features = sample['features'].unsqueeze(0)
        positions = sample['positions'].unsqueeze(0)
        gene_ids = sample['gene_ids'].unsqueeze(0)
        mask = sample['mask'].unsqueeze(0)

        # Find two variants to test
        variant_indices = torch.where(mask[0])[0]
        if len(variant_indices) < 2:
            pytest.skip("Sample needs at least 2 variants")

        v1_idx = variant_indices[0].item()
        v2_idx = variant_indices[1].item()

        # Validate interaction
        result = detector.validate_interaction_with_perturbation(
            features, positions, gene_ids, mask, v1_idx, v2_idx
        )

        # Check result structure (use actual keys returned)
        assert 'pred_both' in result
        assert 'pred_variant1_only' in result
        assert 'pred_variant2_only' in result
        assert 'pred_neither' in result
        assert 'effect_variant1' in result
        assert 'effect_variant2' in result
        assert 'effect_combined' in result
        assert 'synergy' in result
        assert 'interaction_type' in result

        # Check interaction type is valid
        assert result['interaction_type'] in ['synergistic', 'antagonistic', 'independent']


if __name__ == '__main__':
    pytest.main([__file__, '-v'])


def test_validate_epistasis_empty_file():
    """Test validate_epistasis.py handles empty interaction files."""
    import subprocess
    import tempfile
    import os
    from pathlib import Path

    # Create temporary empty interactions file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write("")  # Empty file
        empty_file = f.name

    try:
        # Set PYTHONPATH for subprocess
        env = os.environ.copy()
        project_root = Path(__file__).parent.parent
        env['PYTHONPATH'] = str(project_root)

        # Run validate_epistasis.py script
        result = subprocess.run(
            ['python', 'scripts/validate_epistasis.py',
             '--interactions', empty_file,
             '--checkpoint', 'test_data/small/test_model/L3_run/fold_0/best_model.pt',
             '--config', 'test_data/small/test_model/L3_run/config.yaml',
             '--preprocessed-data', 'test_data/small/preprocessed_test.pt',
             '--output-dir', '/tmp/test_epistasis',
             '--device', 'cpu'],
            cwd=project_root,
            env=env,
            capture_output=True,
            text=True,
            timeout=30
        )
        
        # Should exit cleanly (not crash) with helpful error message
        assert 'ERROR: Interactions file is empty' in result.stdout
        assert 'No interactions were found' in result.stdout
        
    finally:
        # Clean up
        Path(empty_file).unlink(missing_ok=True)


def test_validate_epistasis_no_data():
    """Test validate_epistasis.py handles CSV with headers but no data."""
    import subprocess
    import tempfile
    import os
    from pathlib import Path

    # Create CSV with headers but no data
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write("pos1,pos2,gene1,gene2,weight\n")  # Header only
        header_only_file = f.name

    try:
        # Set PYTHONPATH for subprocess
        env = os.environ.copy()
        project_root = Path(__file__).parent.parent
        env['PYTHONPATH'] = str(project_root)

        result = subprocess.run(
            ['python', 'scripts/validate_epistasis.py',
             '--interactions', header_only_file,
             '--checkpoint', 'test_data/small/test_model/L3_run/fold_0/best_model.pt',
             '--config', 'test_data/small/test_model/L3_run/config.yaml',
             '--preprocessed-data', 'test_data/small/preprocessed_test.pt',
             '--output-dir', '/tmp/test_epistasis',
             '--device', 'cpu'],
            cwd=project_root,
            env=env,
            capture_output=True,
            text=True,
            timeout=30
        )
        
        # Should exit cleanly with helpful error message
        assert 'ERROR: Interactions file has no rows' in result.stdout
        assert 'No interactions available' in result.stdout
        
    finally:
        Path(header_only_file).unlink(missing_ok=True)


class TestLoadSampleAttributions:
    """Test the load_sample_attributions helper function."""

    def test_load_single_sample(self, tmp_path):
        """Test loading a single sample's attributions from per-sample directory."""
        from src.explain import load_sample_attributions

        # Create mock per-sample directory
        per_sample_dir = tmp_path / 'attributions_per_sample'
        per_sample_dir.mkdir()

        # Save mock data for 3 samples
        for i in range(3):
            n_variants = 10 + i * 5
            input_dim = 6
            attrs = np.random.randn(n_variants, input_dim).astype(np.float32)
            scores = np.linalg.norm(attrs, ord=2, axis=1)
            np.savez(per_sample_dir / f'sample_{i}.npz',
                     attributions=attrs, variant_scores=scores)

        # Load sample 1
        result = load_sample_attributions(per_sample_dir, 1)

        assert 'attributions' in result
        assert 'variant_scores' in result
        assert result['attributions'].shape == (15, 6)
        assert result['variant_scores'].shape == (15,)

    def test_load_nonexistent_sample_raises(self, tmp_path):
        """Test that loading a nonexistent sample raises FileNotFoundError."""
        from src.explain import load_sample_attributions

        per_sample_dir = tmp_path / 'attributions_per_sample'
        per_sample_dir.mkdir()

        with pytest.raises(FileNotFoundError):
            load_sample_attributions(per_sample_dir, 999)

    def test_scores_match_attributions(self, tmp_path):
        """Test that variant_scores are the L2 norm of raw attributions."""
        from src.explain import load_sample_attributions

        per_sample_dir = tmp_path / 'attributions_per_sample'
        per_sample_dir.mkdir()

        attrs = np.array([[1.0, 0.0], [3.0, 4.0], [0.0, 0.0]], dtype=np.float32)
        scores = np.linalg.norm(attrs, ord=2, axis=1)
        np.savez(per_sample_dir / 'sample_0.npz',
                 attributions=attrs, variant_scores=scores)

        result = load_sample_attributions(per_sample_dir, 0)
        np.testing.assert_allclose(result['variant_scores'], [1.0, 5.0, 0.0])
