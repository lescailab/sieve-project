"""
Biological validation of discovered variants.

Validates SIEVE discoveries against known biological databases:
- ClinVar: Known pathogenic variants
- GWAS Catalog: Genome-wide association studies
- Gene Ontology: Functional enrichment
- KEGG/Reactome: Pathway enrichment

Author: Lescai Lab
"""

from typing import Dict, List, Optional, Set, Tuple
from collections import defaultdict, Counter
from pathlib import Path

import pandas as pd
import numpy as np
from scipy.stats import hypergeom, fisher_exact


class BiologicalValidator:
    """
    Validate discovered variants against biological databases.

    Parameters
    ----------
    reference_genome : str
        Reference genome build ('GRCh37' or 'GRCh38')

    Examples
    --------
    >>> validator = BiologicalValidator(reference_genome='GRCh37')
    >>> validation = validator.validate_variants(
    ...     variant_rankings,
    ...     clinvar_db='clinvar.vcf',
    ...     gwas_db='gwas_catalog.tsv'
    ... )
    """

    def __init__(self, reference_genome: str = 'GRCh37'):
        self.reference_genome = reference_genome
        self.clinvar_cache = None
        self.gwas_cache = None
        self.gene_info_cache = None

    def load_clinvar(self, clinvar_file: str) -> pd.DataFrame:
        """
        Load ClinVar database.

        Parameters
        ----------
        clinvar_file : str
            Path to ClinVar VCF or TSV file

        Returns
        -------
        clinvar_df : pd.DataFrame
            ClinVar variants with columns:
            - chrom, pos, ref, alt
            - gene, clinical_significance
        """
        # This is a placeholder - actual implementation would parse VCF
        # For now, assume user provides a preprocessed TSV
        if Path(clinvar_file).exists():
            try:
                clinvar_df = pd.read_csv(clinvar_file, sep='\t')
                self.clinvar_cache = clinvar_df
                print(f"Loaded {len(clinvar_df)} ClinVar variants")
                return clinvar_df
            except Exception as e:
                print(f"Warning: Could not load ClinVar: {e}")
                return pd.DataFrame()
        else:
            print(f"Warning: ClinVar file not found: {clinvar_file}")
            return pd.DataFrame()

    def load_gwas_catalog(self, gwas_file: str) -> pd.DataFrame:
        """
        Load GWAS Catalog.

        Parameters
        ----------
        gwas_file : str
            Path to GWAS Catalog file

        Returns
        -------
        gwas_df : pd.DataFrame
            GWAS associations
        """
        if Path(gwas_file).exists():
            try:
                gwas_df = pd.read_csv(gwas_file, sep='\t')
                self.gwas_cache = gwas_df
                print(f"Loaded {len(gwas_df)} GWAS associations")
                return gwas_df
            except Exception as e:
                print(f"Warning: Could not load GWAS Catalog: {e}")
                return pd.DataFrame()
        else:
            print(f"Warning: GWAS file not found: {gwas_file}")
            return pd.DataFrame()

    def validate_variants_against_clinvar(
        self,
        variant_rankings: pd.DataFrame,
        clinvar_df: pd.DataFrame,
        top_k: int = 100
    ) -> pd.DataFrame:
        """
        Check if top variants are in ClinVar.

        Parameters
        ----------
        variant_rankings : pd.DataFrame
            Ranked variants from SIEVE
        clinvar_df : pd.DataFrame
            ClinVar database
        top_k : int
            Number of top variants to check

        Returns
        -------
        validation : pd.DataFrame
            Top variants with ClinVar annotations
        """
        if clinvar_df.empty:
            print("ClinVar data not available - skipping validation")
            return variant_rankings.head(top_k).copy()

        # Get top variants
        top_variants = variant_rankings.head(top_k).copy()

        # Match against ClinVar (assuming position-based matching)
        # In practice, you'd match chr:pos:ref:alt
        top_variants['in_clinvar'] = False
        top_variants['clinvar_significance'] = None

        if 'position' in top_variants.columns and 'pos' in clinvar_df.columns:
            clinvar_positions = set(clinvar_df['pos'])
            top_variants['in_clinvar'] = top_variants['position'].isin(clinvar_positions)

            # Add significance
            for idx, row in top_variants[top_variants['in_clinvar']].iterrows():
                matches = clinvar_df[clinvar_df['pos'] == row['position']]
                if len(matches) > 0:
                    top_variants.at[idx, 'clinvar_significance'] = matches.iloc[0].get(
                        'clinical_significance', 'Unknown'
                    )

        n_in_clinvar = top_variants['in_clinvar'].sum()
        print(f"Found {n_in_clinvar}/{top_k} top variants in ClinVar")

        return top_variants

    def validate_genes_against_gwas(
        self,
        gene_rankings: pd.DataFrame,
        gwas_df: pd.DataFrame,
        disease_terms: Optional[List[str]] = None,
        top_k: int = 50
    ) -> pd.DataFrame:
        """
        Check if top genes are in GWAS Catalog.

        Parameters
        ----------
        gene_rankings : pd.DataFrame
            Ranked genes from SIEVE
        gwas_df : pd.DataFrame
            GWAS Catalog
        disease_terms : Optional[List[str]]
            Disease terms to filter GWAS (e.g., ['diabetes', 'obesity'])
        top_k : int
            Number of top genes to check

        Returns
        -------
        validation : pd.DataFrame
            Top genes with GWAS annotations
        """
        if gwas_df.empty:
            print("GWAS data not available - skipping validation")
            return gene_rankings.head(top_k).copy()

        # Get top genes
        top_genes = gene_rankings.head(top_k).copy()

        # Match against GWAS
        top_genes['in_gwas'] = False
        top_genes['gwas_traits'] = None
        top_genes['gwas_studies'] = 0

        if 'gene_id' in top_genes.columns and 'gene' in gwas_df.columns:
            # Filter GWAS by disease if provided
            if disease_terms:
                gwas_filtered = gwas_df[
                    gwas_df['trait'].str.contains('|'.join(disease_terms), case=False, na=False)
                ]
                print(f"Filtered GWAS to {len(gwas_filtered)} disease-relevant associations")
            else:
                gwas_filtered = gwas_df

            # Count GWAS hits per gene
            gwas_gene_counts = gwas_filtered.groupby('gene').size().to_dict()
            gwas_gene_traits = gwas_filtered.groupby('gene')['trait'].apply(
                lambda x: '; '.join(x.unique()[:5])  # Top 5 traits
            ).to_dict()

            for idx, row in top_genes.iterrows():
                gene_id = row['gene_id']
                if gene_id in gwas_gene_counts:
                    top_genes.at[idx, 'in_gwas'] = True
                    top_genes.at[idx, 'gwas_studies'] = gwas_gene_counts[gene_id]
                    top_genes.at[idx, 'gwas_traits'] = gwas_gene_traits[gene_id]

        n_in_gwas = top_genes['in_gwas'].sum()
        print(f"Found {n_in_gwas}/{top_k} top genes in GWAS Catalog")

        return top_genes

    def compute_enrichment(
        self,
        discovered_genes: List[int],
        database_genes: Set[int],
        total_genes: int
    ) -> Dict:
        """
        Compute enrichment statistics.

        Uses hypergeometric test to determine if discovered genes
        are enriched in a database (e.g., disease genes).

        Parameters
        ----------
        discovered_genes : List[int]
            Gene IDs from SIEVE
        database_genes : Set[int]
            Gene IDs in database (e.g., GWAS genes)
        total_genes : int
            Total number of genes in genome

        Returns
        -------
        enrichment : Dict
            Enrichment statistics:
            - overlap: number of overlapping genes
            - p_value: hypergeometric p-value
            - odds_ratio: odds ratio
            - enrichment_fold: fold enrichment
        """
        n_discovered = len(discovered_genes)
        n_database = len(database_genes)
        overlap = len(set(discovered_genes) & database_genes)

        # Hypergeometric test
        # P(X >= overlap | n_discovered, n_database, total_genes)
        p_value = hypergeom.sf(
            overlap - 1,  # -1 because sf is P(X >= x), we want P(X >= overlap)
            total_genes,
            n_database,
            n_discovered
        )

        # Odds ratio
        # (overlap / (n_discovered - overlap)) / ((n_database - overlap) / (total_genes - n_database - n_discovered + overlap))
        contingency = [
            [overlap, n_discovered - overlap],
            [n_database - overlap, total_genes - n_database - n_discovered + overlap]
        ]
        odds_ratio, fisher_p = fisher_exact(contingency)

        # Fold enrichment
        expected = (n_discovered * n_database) / total_genes
        fold_enrichment = overlap / expected if expected > 0 else 0

        return {
            'n_discovered': n_discovered,
            'n_database': n_database,
            'overlap': overlap,
            'p_value': p_value,
            'fisher_p_value': fisher_p,
            'odds_ratio': odds_ratio,
            'fold_enrichment': fold_enrichment,
            'is_significant': p_value < 0.05
        }

    def perform_go_enrichment(
        self,
        gene_list: List[int],
        gene_to_go: Dict[int, List[str]],
        background_genes: Optional[List[int]] = None,
        min_genes_per_term: int = 3,
        max_genes_per_term: int = 500
    ) -> pd.DataFrame:
        """
        Perform Gene Ontology enrichment analysis.

        Parameters
        ----------
        gene_list : List[int]
            Genes to test for enrichment
        gene_to_go : Dict[int, List[str]]
            Mapping of gene IDs to GO terms
        background_genes : Optional[List[int]]
            Background gene set (default: all genes in gene_to_go)
        min_genes_per_term : int
            Minimum genes per term to test
        max_genes_per_term : int
            Maximum genes per term to test

        Returns
        -------
        enrichment : pd.DataFrame
            GO terms with enrichment statistics
        """
        if background_genes is None:
            background_genes = list(gene_to_go.keys())

        # Create term-to-genes mapping
        term_to_genes = defaultdict(set)
        for gene, terms in gene_to_go.items():
            if gene in background_genes:
                for term in terms:
                    term_to_genes[term].add(gene)

        # Filter terms by size
        term_to_genes = {
            term: genes
            for term, genes in term_to_genes.items()
            if min_genes_per_term <= len(genes) <= max_genes_per_term
        }

        # Test enrichment for each term
        results = []
        gene_set = set(gene_list)
        total_genes = len(background_genes)

        for term, term_genes in term_to_genes.items():
            overlap = len(gene_set & term_genes)
            if overlap > 0:
                enrichment = self.compute_enrichment(
                    discovered_genes=list(gene_set),
                    database_genes=term_genes,
                    total_genes=total_genes
                )

                results.append({
                    'go_term': term,
                    'overlap': overlap,
                    'n_term_genes': len(term_genes),
                    'p_value': enrichment['p_value'],
                    'fold_enrichment': enrichment['fold_enrichment'],
                    'overlapping_genes': ','.join(map(str, gene_set & term_genes))
                })

        if not results:
            return pd.DataFrame()

        # Create DataFrame and sort by p-value
        df = pd.DataFrame(results)
        df = df.sort_values('p_value')

        # FDR correction (Benjamini-Hochberg)
        from scipy.stats import false_discovery_control
        df['fdr'] = false_discovery_control(df['p_value'].values)

        return df

    def create_validation_summary(
        self,
        variant_rankings: pd.DataFrame,
        gene_rankings: pd.DataFrame,
        clinvar_validation: Optional[pd.DataFrame] = None,
        gwas_validation: Optional[pd.DataFrame] = None,
        go_enrichment: Optional[pd.DataFrame] = None
    ) -> Dict:
        """
        Create comprehensive validation summary.

        Parameters
        ----------
        variant_rankings : pd.DataFrame
            All variant rankings
        gene_rankings : pd.DataFrame
            All gene rankings
        clinvar_validation : Optional[pd.DataFrame]
            ClinVar validation results
        gwas_validation : Optional[pd.DataFrame]
            GWAS validation results
        go_enrichment : Optional[pd.DataFrame]
            GO enrichment results

        Returns
        -------
        summary : Dict
            Validation summary statistics
        """
        summary = {
            'total_variants': len(variant_rankings),
            'total_genes': len(gene_rankings),
        }

        # ClinVar summary
        if clinvar_validation is not None and 'in_clinvar' in clinvar_validation.columns:
            n_in_clinvar = clinvar_validation['in_clinvar'].sum()
            n_pathogenic = 0
            if 'clinvar_significance' in clinvar_validation.columns:
                n_pathogenic = clinvar_validation['clinvar_significance'].str.contains(
                    'Pathogenic', case=False, na=False
                ).sum()

            summary['clinvar'] = {
                'n_variants_checked': len(clinvar_validation),
                'n_in_clinvar': int(n_in_clinvar),
                'n_pathogenic': int(n_pathogenic),
                'pct_in_clinvar': float(n_in_clinvar / len(clinvar_validation) * 100) if len(clinvar_validation) > 0 else 0
            }

        # GWAS summary
        if gwas_validation is not None and 'in_gwas' in gwas_validation.columns:
            n_in_gwas = gwas_validation['in_gwas'].sum()
            summary['gwas'] = {
                'n_genes_checked': len(gwas_validation),
                'n_in_gwas': int(n_in_gwas),
                'pct_in_gwas': float(n_in_gwas / len(gwas_validation) * 100) if len(gwas_validation) > 0 else 0,
                'total_studies': int(gwas_validation['gwas_studies'].sum()) if 'gwas_studies' in gwas_validation.columns else 0
            }

        # GO enrichment summary
        if go_enrichment is not None and len(go_enrichment) > 0:
            n_significant = (go_enrichment['fdr'] < 0.05).sum()
            summary['go_enrichment'] = {
                'n_terms_tested': len(go_enrichment),
                'n_significant': int(n_significant),
                'top_term': go_enrichment.iloc[0]['go_term'] if len(go_enrichment) > 0 else None,
                'top_term_pvalue': float(go_enrichment.iloc[0]['p_value']) if len(go_enrichment) > 0 else None
            }

        return summary
