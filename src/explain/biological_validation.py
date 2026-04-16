"""
Biological validation of discovered variants.

Validates SIEVE discoveries against known biological databases:
- ClinVar: Known pathogenic variants
- GWAS Catalog: Genome-wide association studies
- Gene Ontology: Functional enrichment
- KEGG/Reactome: Pathway enrichment

Author: Francesco Lescai
"""

from typing import Dict, List, Optional, Set
from collections import defaultdict
import logging
from pathlib import Path

import pandas as pd
from scipy.stats import hypergeom, fisher_exact

logger = logging.getLogger(__name__)


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
                chrom_candidates = ('chromosome', 'chrom', 'CHROM')
                pos_candidates = ('position', 'pos', 'POS')

                chrom_col = next((c for c in chrom_candidates if c in clinvar_df.columns), None)
                pos_col = next((c for c in pos_candidates if c in clinvar_df.columns), None)
                if chrom_col is None or pos_col is None:
                    raise ValueError(
                        "ClinVar file must contain chromosome and position columns. "
                        f"Expected one of {chrom_candidates} and one of {pos_candidates}; "
                        f"got {list(clinvar_df.columns)}"
                    )

                clinvar_df = clinvar_df.rename(
                    columns={chrom_col: 'chrom', pos_col: 'pos'}
                ).copy()
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

    def match_variants_to_clinvar(
        self,
        variant_rankings: pd.DataFrame,
        clinvar_df: pd.DataFrame,
        top_k: int = 100
    ) -> pd.DataFrame:
        """
        Check if top variants are in ClinVar.

        Matching requires both chromosome and position.  Chromosome prefixes
        ('chr1' vs '1') are normalised automatically before comparison.
        Position-only matching is not supported because the same genomic
        position exists on multiple chromosomes.

        Parameters
        ----------
        variant_rankings : pd.DataFrame
            Ranked variants from SIEVE (must contain 'chromosome' and 'position')
        clinvar_df : pd.DataFrame
            ClinVar database (must contain 'chrom' and 'pos')
        top_k : int
            Number of top variants to check

        Returns
        -------
        validation : pd.DataFrame
            Top variants with ClinVar annotations

        Raises
        ------
        ValueError
            If variant_rankings is missing 'chromosome' or 'position', or
            if clinvar_df is missing 'chrom' or 'pos'.
        """
        if clinvar_df.empty:
            print("ClinVar data not available - skipping validation")
            return variant_rankings.head(top_k).copy()

        top_variants = variant_rankings.head(top_k).copy()

        if 'chromosome' not in top_variants.columns or 'position' not in top_variants.columns:
            raise ValueError(
                "match_variants_to_clinvar requires 'chromosome' and 'position' "
                f"columns in variant_rankings; got {list(top_variants.columns)}"
            )
        if 'chrom' not in clinvar_df.columns or 'pos' not in clinvar_df.columns:
            raise ValueError(
                "match_variants_to_clinvar requires 'chrom' and 'pos' columns "
                f"in clinvar_df; got {list(clinvar_df.columns)}"
            )

        def _strip_chr(series: pd.Series) -> pd.Series:
            return series.astype(str).str.lower().str.removeprefix('chr')

        clinvar_working = clinvar_df.copy()
        top_working = top_variants.copy()
        clinvar_working['_chrom_norm'] = _strip_chr(clinvar_working['chrom'])
        top_working['_chrom_norm'] = _strip_chr(top_working['chromosome'])

        # Deduplicate ClinVar on (chrom, pos) before merging to prevent row
        # duplication when multiple ClinVar entries share the same coordinate.
        # Keep the first entry; aggregate significance into a semicolon-delimited
        # string so no information is silently discarded.
        if clinvar_working.duplicated(subset=['_chrom_norm', 'pos']).any():
            if 'clinical_significance' in clinvar_working.columns:
                clinvar_working = (
                    clinvar_working
                    .groupby(['_chrom_norm', 'pos'], as_index=False)
                    .agg({'clinical_significance': lambda x: '; '.join(x.dropna().unique()),
                          **{c: 'first' for c in clinvar_working.columns
                             if c not in ('_chrom_norm', 'pos', 'clinical_significance')}})
                )
            else:
                clinvar_working = clinvar_working.drop_duplicates(
                    subset=['_chrom_norm', 'pos'], keep='first'
                )

        logger.info(
            "ClinVar matching: input_variants=%d clinvar_rows=%d",
            len(top_working), len(clinvar_working),
        )

        merged = top_working.merge(
            clinvar_working,
            left_on=['_chrom_norm', 'position'],
            right_on=['_chrom_norm', 'pos'],
            how='left',
            suffixes=('', '_clinvar'),
        )
        merged['in_clinvar'] = merged['pos'].notna()
        merged['clinvar_significance'] = merged.get('clinical_significance', 'Unknown')
        merged.loc[~merged['in_clinvar'], 'clinvar_significance'] = None

        n_matched = int(merged['in_clinvar'].sum())
        logger.info(
            "ClinVar matching: matched_variants=%d/%d", n_matched, len(merged)
        )

        keep_columns = list(top_variants.columns) + ['in_clinvar', 'clinvar_significance']
        return merged[keep_columns]

    def validate_variants_against_clinvar(
        self,
        variant_rankings: pd.DataFrame,
        clinvar_df: pd.DataFrame,
        top_k: int = 100
    ) -> pd.DataFrame:
        """Backward-compatible wrapper around chromosome+position ClinVar matching."""
        return self.match_variants_to_clinvar(
            variant_rankings=variant_rankings,
            clinvar_df=clinvar_df,
            top_k=top_k,
        )

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

        # Match against GWAS using gene_name (gene symbol)
        top_genes['in_gwas'] = False
        top_genes['gwas_traits'] = None
        top_genes['gwas_studies'] = 0

        # Check column names in GWAS file (could be 'gene' or 'gene_name')
        gwas_gene_col = None
        if 'gene' in gwas_df.columns:
            gwas_gene_col = 'gene'
        elif 'gene_name' in gwas_df.columns:
            gwas_gene_col = 'gene_name'

        # Check if we have gene_name in rankings
        has_gene_name = 'gene_name' in top_genes.columns

        if has_gene_name and gwas_gene_col:
            # Filter GWAS by disease if provided
            if disease_terms:
                # GWAS trait column could be 'disease_trait' or 'trait'
                trait_col = 'disease_trait' if 'disease_trait' in gwas_df.columns else 'trait'
                if trait_col in gwas_df.columns:
                    gwas_filtered = gwas_df[
                        gwas_df[trait_col].astype(str).str.contains('|'.join(disease_terms), case=False, na=False)
                    ].copy()
                    print(f"Filtered GWAS to {len(gwas_filtered)} disease-relevant associations")
                else:
                    gwas_filtered = gwas_df.copy()
            else:
                gwas_filtered = gwas_df.copy()

            # Count GWAS hits per gene (case-insensitive matching)
            gwas_filtered[gwas_gene_col] = gwas_filtered[gwas_gene_col].astype(str).str.upper()
            gwas_gene_counts = gwas_filtered.groupby(gwas_gene_col).size().to_dict()

            # Get traits per gene
            trait_col = 'disease_trait' if 'disease_trait' in gwas_filtered.columns else 'trait'
            if trait_col in gwas_filtered.columns:
                gwas_gene_traits = gwas_filtered.groupby(gwas_gene_col)[trait_col].apply(
                    lambda x: '; '.join(x.dropna().unique()[:5])  # Top 5 traits
                ).to_dict()
            else:
                gwas_gene_traits = {}

            # Match genes
            for idx, row in top_genes.iterrows():
                gene_name = str(row['gene_name']).upper()
                if gene_name in gwas_gene_counts:
                    top_genes.at[idx, 'in_gwas'] = True
                    top_genes.at[idx, 'gwas_studies'] = gwas_gene_counts[gene_name]
                    top_genes.at[idx, 'gwas_traits'] = gwas_gene_traits.get(gene_name, '')
        else:
            print(f"ERROR: Required columns missing for GWAS matching")
            print(f"  Rankings has gene_name: {has_gene_name}")
            print(f"  GWAS has gene column: {gwas_gene_col}")

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
        gene_list: List[str],
        gene_to_go: Dict[str, List[str]],
        background_genes: Optional[List[str]] = None,
        min_genes_per_term: int = 3,
        max_genes_per_term: int = 500
    ) -> pd.DataFrame:
        """
        Perform Gene Ontology enrichment analysis.

        Parameters
        ----------
        gene_list : List[str]
            Gene symbols to test for enrichment
        gene_to_go : Dict[str, List[str]]
            Mapping of gene symbols to GO term IDs
        background_genes : Optional[List[str]]
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

        # Convert to uppercase for case-insensitive matching
        gene_list = [str(g).upper() for g in gene_list]
        gene_to_go_upper = {str(k).upper(): v for k, v in gene_to_go.items()}
        background_genes_upper = [str(g).upper() for g in background_genes]

        # Create term-to-genes mapping
        term_to_genes = defaultdict(set)
        for gene, terms in gene_to_go_upper.items():
            if gene in background_genes_upper:
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
        total_genes = len(background_genes_upper)

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
