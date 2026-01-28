"""
Unit tests for database download utilities.
"""
import pytest
import json
import pandas as pd
from pathlib import Path
import tempfile

from utilities.download_clinvar import parse_clinvar_vcf, parse_info_field
from utilities.download_gwas_catalog import parse_gwas_catalog
from utilities.download_gene_ontology import parse_biomart_file, parse_gaf_file, GO_ASPECTS


class TestClinVarParser:
    """Test ClinVar VCF parsing."""

    def test_parse_info_field(self):
        """Test INFO field extraction."""
        info = "CLNSIG=Pathogenic;GENEINFO=BRCA1:672;AF=0.001"

        assert parse_info_field(info, 'CLNSIG') == 'Pathogenic'
        assert parse_info_field(info, 'GENEINFO') == 'BRCA1:672'
        assert parse_info_field(info, 'AF') == '0.001'
        assert parse_info_field(info, 'MISSING') is None

    def test_parse_vcf(self, tmp_path):
        """Test VCF parsing end-to-end."""
        # Create test VCF
        vcf_content = """##fileformat=VCFv4.1
##INFO=<ID=CLNSIG,Number=.,Type=String,Description="Clinical significance">
##INFO=<ID=GENEINFO,Number=1,Type=String,Description="Gene information">
#CHROM	POS	ID	REF	ALT	QUAL	FILTER	INFO
1	12345	rs123	A	G	.	.	CLNSIG=Pathogenic;GENEINFO=BRCA1:672
1	23456	rs456	C	T	.	.	CLNSIG=Benign;GENEINFO=TP53:7157
2	34567	rs789	G	A	.	.	CLNSIG=Likely_pathogenic;GENEINFO=CFTR:1080
"""
        vcf_path = tmp_path / "test.vcf"
        vcf_path.write_text(vcf_content)

        output_path = tmp_path / "output.tsv"
        parse_clinvar_vcf(vcf_path, output_path)

        # Check output
        df = pd.read_csv(output_path, sep='\t')
        assert len(df) == 2  # Only pathogenic variants
        assert set(df['gene']) == {'BRCA1', 'CFTR'}
        assert 'chrom' in df.columns
        assert 'pos' in df.columns
        assert 'ref' in df.columns
        assert 'alt' in df.columns
        assert 'clinical_significance' in df.columns

    def test_max_variants_limit(self, tmp_path):
        """Test max_variants parameter."""
        # Create VCF with multiple pathogenic variants
        vcf_content = """##fileformat=VCFv4.1
##INFO=<ID=CLNSIG,Number=.,Type=String>
##INFO=<ID=GENEINFO,Number=1,Type=String>
#CHROM	POS	ID	REF	ALT	QUAL	FILTER	INFO
1	100	.	A	G	.	.	CLNSIG=Pathogenic;GENEINFO=GENE1:1
1	200	.	C	T	.	.	CLNSIG=Pathogenic;GENEINFO=GENE2:2
1	300	.	G	A	.	.	CLNSIG=Pathogenic;GENEINFO=GENE3:3
"""
        vcf_path = tmp_path / "test.vcf"
        vcf_path.write_text(vcf_content)

        output_path = tmp_path / "output.tsv"
        parse_clinvar_vcf(vcf_path, output_path, max_variants=2)

        df = pd.read_csv(output_path, sep='\t')
        assert len(df) == 2  # Limited to 2


class TestGWASParser:
    """Test GWAS Catalog parsing."""

    def test_parse_gwas_catalog(self, tmp_path):
        """Test GWAS catalog parsing."""
        # Create test GWAS file
        gwas_content = """DISEASE/TRAIT	CHR_ID	CHR_POS	MAPPED_GENE	SNPS	P-VALUE	STRONGEST SNP-RISK ALLELE
Type 2 diabetes	1	100000	TCF7L2	rs7903146	1e-12	rs7903146-A
Alzheimer's disease	19	45411941	APOE	rs429358	5e-100	rs429358-C
Hypertension	3	50000	GENE1	rs111111	1e-6	rs111111-T
"""
        gwas_path = tmp_path / "gwas.tsv"
        gwas_path.write_text(gwas_content)

        output_path = tmp_path / "output.tsv"
        parse_gwas_catalog(gwas_path, output_path, min_pvalue=1e-8)

        # Check output
        df = pd.read_csv(output_path, sep='\t')
        assert len(df) == 2  # Only p < 1e-8
        assert set(df['gene']) == {'TCF7L2', 'APOE'}
        assert 'disease_trait' in df.columns
        assert 'p_value' in df.columns

    def test_pvalue_filtering(self, tmp_path):
        """Test p-value threshold filtering."""
        gwas_content = """DISEASE/TRAIT	MAPPED_GENE	P-VALUE
Disease1	GENE1	1e-5
Disease2	GENE2	1e-10
Disease3	GENE3	1e-15
"""
        gwas_path = tmp_path / "gwas.tsv"
        gwas_path.write_text(gwas_content)

        output_path = tmp_path / "output.tsv"
        parse_gwas_catalog(gwas_path, output_path, min_pvalue=1e-8)

        df = pd.read_csv(output_path, sep='\t')
        assert len(df) == 2  # Only 1e-10 and 1e-15
        assert all(df['p_value'] < 1e-8)

    def test_multiple_genes(self, tmp_path):
        """Test handling of multiple genes per association."""
        gwas_content = """DISEASE/TRAIT	MAPPED_GENE	P-VALUE
Disease1	GENE1, GENE2	1e-10
"""
        gwas_path = tmp_path / "gwas.tsv"
        gwas_path.write_text(gwas_content)

        output_path = tmp_path / "output.tsv"
        parse_gwas_catalog(gwas_path, output_path, min_pvalue=1e-8)

        df = pd.read_csv(output_path, sep='\t')
        assert len(df) == 2  # Split into 2 rows
        assert set(df['gene']) == {'GENE1', 'GENE2'}


class TestGOParser:
    """Test Gene Ontology parsing."""

    def test_parse_biomart_file(self, tmp_path):
        """Test BioMart file parsing."""
        # Create test BioMart file
        biomart_content = """Gene name	GO term accession	GO term name	GO domain
TP53	GO:0006355	regulation of transcription	P
TP53	GO:0003677	DNA binding	F
BRCA1	GO:0006281	DNA repair	P
BRCA1	GO:0005634	nucleus	C
"""
        biomart_path = tmp_path / "biomart.tsv"
        biomart_path.write_text(biomart_content)

        # Test all aspects
        gene_to_go = parse_biomart_file(biomart_path, GO_ASPECTS['all'])
        assert len(gene_to_go) == 2
        assert 'TP53' in gene_to_go
        assert 'BRCA1' in gene_to_go
        assert len(gene_to_go['TP53']) == 2
        assert 'GO:0006355' in gene_to_go['TP53']

    def test_aspect_filtering(self, tmp_path):
        """Test GO aspect filtering."""
        biomart_content = """Gene name	GO term accession	GO term name	GO domain
TP53	GO:0006355	regulation of transcription	P
TP53	GO:0003677	DNA binding	F
TP53	GO:0005634	nucleus	C
"""
        biomart_path = tmp_path / "biomart.tsv"
        biomart_path.write_text(biomart_content)

        # Test biological_process only
        gene_to_go = parse_biomart_file(biomart_path, GO_ASPECTS['biological_process'])
        assert len(gene_to_go['TP53']) == 1
        assert 'GO:0006355' in gene_to_go['TP53']

        # Test molecular_function only
        gene_to_go = parse_biomart_file(biomart_path, GO_ASPECTS['molecular_function'])
        assert len(gene_to_go['TP53']) == 1
        assert 'GO:0003677' in gene_to_go['TP53']

    def test_json_output_format(self, tmp_path):
        """Test JSON output format."""
        biomart_content = """Gene name	GO term accession	GO term name	GO domain
TP53	GO:0006355	regulation of transcription	P
BRCA1	GO:0006281	DNA repair	P
"""
        biomart_path = tmp_path / "biomart.tsv"
        biomart_path.write_text(biomart_content)

        gene_to_go = parse_biomart_file(biomart_path, GO_ASPECTS['all'])

        # Convert to JSON format (as done in main script)
        gene_to_go_list = {gene: sorted(list(go_terms))
                           for gene, go_terms in gene_to_go.items()}

        output_path = tmp_path / "output.json"
        with open(output_path, 'w') as f:
            json.dump(gene_to_go_list, f, indent=2)

        # Verify can be loaded
        with open(output_path, 'r') as f:
            loaded = json.load(f)

        assert isinstance(loaded, dict)
        assert 'TP53' in loaded
        assert isinstance(loaded['TP53'], list)
        assert 'GO:0006355' in loaded['TP53']


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
