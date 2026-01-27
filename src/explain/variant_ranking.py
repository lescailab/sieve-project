"""
Variant ranking and prioritization.

Combines attribution scores and attention patterns to rank variants
by their importance for disease prediction.

Author: Lescai Lab
"""

from typing import Dict, List, Optional
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.stats import rankdata


class VariantRanker:
    """
    Rank variants by importance scores.

    Combines multiple sources of evidence:
    - Integrated gradients attribution scores
    - Attention weights (how often variant is attended to)
    - Consistency across samples (how many samples show high attribution)

    Parameters
    ----------
    aggregation : str
        How to aggregate scores: 'mean', 'max', 'rank_average'

    Examples
    --------
    >>> ranker = VariantRanker(aggregation='rank_average')
    >>> rankings = ranker.rank_variants(
    ...     attributions=all_attributions,
    ...     metadata=all_metadata
    ... )
    """

    def __init__(self, aggregation: str = 'rank_average'):
        self.aggregation = aggregation
        if aggregation not in ['mean', 'max', 'rank_average']:
            raise ValueError(f"Unknown aggregation: {aggregation}")

    def rank_variants(
        self,
        attributions: List[np.ndarray],
        metadata: List[Dict],
        case_indices: Optional[List[int]] = None,
        control_indices: Optional[List[int]] = None
    ) -> pd.DataFrame:
        """
        Rank all variants across all samples.

        Parameters
        ----------
        attributions : List[np.ndarray]
            List of attribution arrays, one per sample
        metadata : List[Dict]
            List of metadata dicts with positions, genes
        case_indices : Optional[List[int]]
            Indices of case samples (for case-specific ranking)
        control_indices : Optional[List[int]]
            Indices of control samples (for control-specific ranking)

        Returns
        -------
        rankings : pd.DataFrame
            Ranked variants with columns:
            - position: genomic position
            - gene_id: gene ID
            - mean_attribution: mean absolute attribution
            - max_attribution: max absolute attribution
            - num_samples: number of samples with this variant
            - case_attribution: mean attribution in cases (if provided)
            - control_attribution: mean attribution in controls (if provided)
            - case_control_diff: difference (if both provided)
            - rank: final rank (1 = most important)
        """
        # Collect variant scores
        variant_scores = defaultdict(lambda: {
            'attributions': [],
            'case_attributions': [],
            'control_attributions': [],
            'samples': []
        })

        for sample_idx, (attr, meta) in enumerate(zip(attributions, metadata)):
            positions = meta['positions']
            genes = meta['gene_ids']

            # Get absolute attribution scores
            abs_attr = np.abs(attr)

            # Aggregate across features if needed
            if abs_attr.ndim > 1:
                abs_attr = np.linalg.norm(abs_attr, ord=2, axis=1)

            # Store per-variant scores
            for i, (pos, gene) in enumerate(zip(positions, genes)):
                key = (int(pos), int(gene))
                score = float(abs_attr[i])

                variant_scores[key]['attributions'].append(score)
                variant_scores[key]['samples'].append(sample_idx)

                # Separate by case/control
                if case_indices is not None and sample_idx in case_indices:
                    variant_scores[key]['case_attributions'].append(score)
                if control_indices is not None and sample_idx in control_indices:
                    variant_scores[key]['control_attributions'].append(score)

        # Build DataFrame
        records = []
        for (pos, gene), scores in variant_scores.items():
            attrs = np.array(scores['attributions'])

            record = {
                'position': pos,
                'gene_id': gene,
                'mean_attribution': float(np.mean(attrs)),
                'max_attribution': float(np.max(attrs)),
                'median_attribution': float(np.median(attrs)),
                'std_attribution': float(np.std(attrs)),
                'num_samples': len(attrs),
            }

            # Case/control analysis
            if scores['case_attributions']:
                record['case_attribution'] = float(np.mean(scores['case_attributions']))
                record['case_num_samples'] = len(scores['case_attributions'])

            if scores['control_attributions']:
                record['control_attribution'] = float(np.mean(scores['control_attributions']))
                record['control_num_samples'] = len(scores['control_attributions'])

            if scores['case_attributions'] and scores['control_attributions']:
                record['case_control_diff'] = (
                    record['case_attribution'] - record['control_attribution']
                )

            records.append(record)

        df = pd.DataFrame(records)

        # Compute final ranking
        if self.aggregation == 'mean':
            df['score'] = df['mean_attribution']
        elif self.aggregation == 'max':
            df['score'] = df['max_attribution']
        elif self.aggregation == 'rank_average':
            # Rank-based aggregation (more robust)
            df['rank_mean'] = rankdata(-df['mean_attribution'])
            df['rank_max'] = rankdata(-df['max_attribution'])
            df['rank_samples'] = rankdata(-df['num_samples'])
            df['score'] = (df['rank_mean'] + df['rank_max'] + df['rank_samples']) / 3

        df['rank'] = rankdata(df['score']).astype(int)
        df = df.sort_values('rank')

        return df

    def rank_genes(
        self,
        variant_rankings: pd.DataFrame,
        aggregation: str = 'max'
    ) -> pd.DataFrame:
        """
        Aggregate variant rankings to gene level.

        Parameters
        ----------
        variant_rankings : pd.DataFrame
            Output from rank_variants()
        aggregation : str
            How to aggregate variants to genes: 'max', 'mean', 'sum'

        Returns
        -------
        gene_rankings : pd.DataFrame
            Gene-level rankings with:
            - gene_id: gene ID
            - num_variants: number of variants in gene
            - gene_score: aggregated score
            - top_variant_pos: position of top variant
            - top_variant_score: score of top variant
            - gene_rank: gene ranking
        """
        if aggregation == 'max':
            gene_agg = variant_rankings.groupby('gene_id').agg({
                'mean_attribution': 'max',
                'num_samples': 'sum',
                'position': 'count'
            }).rename(columns={'position': 'num_variants'})
            gene_agg['gene_score'] = gene_agg['mean_attribution']

        elif aggregation == 'mean':
            gene_agg = variant_rankings.groupby('gene_id').agg({
                'mean_attribution': 'mean',
                'num_samples': 'sum',
                'position': 'count'
            }).rename(columns={'position': 'num_variants'})
            gene_agg['gene_score'] = gene_agg['mean_attribution']

        elif aggregation == 'sum':
            gene_agg = variant_rankings.groupby('gene_id').agg({
                'mean_attribution': 'sum',
                'num_samples': 'sum',
                'position': 'count'
            }).rename(columns={'position': 'num_variants'})
            gene_agg['gene_score'] = gene_agg['mean_attribution']

        else:
            raise ValueError(f"Unknown aggregation: {aggregation}")

        # Get top variant per gene
        top_variants = variant_rankings.loc[
            variant_rankings.groupby('gene_id')['mean_attribution'].idxmax()
        ][['gene_id', 'position', 'mean_attribution']].rename(columns={
            'position': 'top_variant_pos',
            'mean_attribution': 'top_variant_score'
        })

        gene_rankings = gene_agg.merge(top_variants, on='gene_id')
        gene_rankings['gene_rank'] = rankdata(-gene_rankings['gene_score']).astype(int)
        gene_rankings = gene_rankings.sort_values('gene_rank').reset_index(drop=True)

        return gene_rankings

    def get_case_enriched_variants(
        self,
        variant_rankings: pd.DataFrame,
        min_case_samples: int = 5,
        min_diff: float = 0.1,
        top_k: int = 100
    ) -> pd.DataFrame:
        """
        Get variants enriched in cases vs controls.

        Parameters
        ----------
        variant_rankings : pd.DataFrame
            Output from rank_variants() with case/control info
        min_case_samples : int
            Minimum number of cases with the variant
        min_diff : float
            Minimum case-control attribution difference
        top_k : int
            Number of top variants to return

        Returns
        -------
        case_enriched : pd.DataFrame
            Case-enriched variants
        """
        if 'case_control_diff' not in variant_rankings.columns:
            raise ValueError("Rankings must include case/control information")

        # Filter
        filtered = variant_rankings[
            (variant_rankings['case_num_samples'] >= min_case_samples) &
            (variant_rankings['case_control_diff'] > min_diff)
        ].copy()

        # Sort by case-control difference
        filtered['case_enrichment_rank'] = rankdata(
            -filtered['case_control_diff']
        ).astype(int)

        filtered = filtered.sort_values('case_enrichment_rank').head(top_k)

        return filtered

    def export_rankings(
        self,
        variant_rankings: pd.DataFrame,
        gene_rankings: pd.DataFrame,
        output_dir: str,
        prefix: str = 'sieve'
    ):
        """
        Export rankings to CSV files.

        Parameters
        ----------
        variant_rankings : pd.DataFrame
            Variant-level rankings
        gene_rankings : pd.DataFrame
            Gene-level rankings
        output_dir : str
            Output directory
        prefix : str
            File prefix
        """
        import os
        os.makedirs(output_dir, exist_ok=True)

        # Export variant rankings
        variant_path = os.path.join(output_dir, f'{prefix}_variant_rankings.csv')
        variant_rankings.to_csv(variant_path, index=False)
        print(f"Variant rankings saved to {variant_path}")

        # Export gene rankings
        gene_path = os.path.join(output_dir, f'{prefix}_gene_rankings.csv')
        gene_rankings.to_csv(gene_path, index=False)
        print(f"Gene rankings saved to {gene_path}")

        # Export top 100 variants
        top100_path = os.path.join(output_dir, f'{prefix}_top100_variants.csv')
        variant_rankings.head(100).to_csv(top100_path, index=False)
        print(f"Top 100 variants saved to {top100_path}")

        # Export top 50 genes
        top50_path = os.path.join(output_dir, f'{prefix}_top50_genes.csv')
        gene_rankings.head(50).to_csv(top50_path, index=False)
        print(f"Top 50 genes saved to {top50_path}")
