"""
Genome build definitions for SIEVE.

This module centralises all reference-genome-dependent constants:
- Pseudoautosomal region (PAR) coordinates
- Sex chromosome contig identifiers
- Autosomal chromosome lists
- Contig harmonisation rules

All build-specific logic throughout the pipeline must import from this
module rather than hardcoding coordinates.

Supported builds: GRCh37, GRCh38
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple

SUPPORTED_BUILDS = ('GRCh37', 'GRCh38')


@dataclass(frozen=True)
class GenomeBuild:
    """Immutable container for genome build parameters."""

    name: str
    par_regions: Dict[str, List[Tuple[int, int]]]
    sex_chroms: Tuple[str, ...]
    autosomal_chroms: Tuple[str, ...]
    x_contig_aliases: Tuple[str, ...]
    y_contig_aliases: Tuple[str, ...]


GRCH37 = GenomeBuild(
    name='GRCh37',
    par_regions={
        'X': [(60001, 2699520), (154931044, 155260560)],
        'Y': [(10001, 2649520), (59034050, 59363566)],
    },
    sex_chroms=('X', 'Y'),
    autosomal_chroms=tuple(str(c) for c in range(1, 23)),
    x_contig_aliases=('X', 'chrX', '23', 'chr23'),
    y_contig_aliases=('Y', 'chrY', '24', 'chr24'),
)

GRCH38 = GenomeBuild(
    name='GRCh38',
    par_regions={
        'X': [(10001, 2781479), (155701383, 156030895)],
        'Y': [(10001, 2781479), (56887903, 57217415)],
    },
    sex_chroms=('X', 'Y'),
    autosomal_chroms=tuple(str(c) for c in range(1, 23)),
    x_contig_aliases=('X', 'chrX', '23', 'chr23'),
    y_contig_aliases=('Y', 'chrY', '24', 'chr24'),
)

BUILDS = {
    'GRCh37': GRCH37,
    'GRCh38': GRCH38,
}


def get_genome_build(name: str) -> GenomeBuild:
    """
    Retrieve GenomeBuild by name.

    Accepts case-insensitive input and common aliases
    (hg19 -> GRCh37, hg38 -> GRCh38).

    Parameters
    ----------
    name : str
        Build name or alias.

    Returns
    -------
    GenomeBuild
        The corresponding build object.

    Raises
    ------
    ValueError
        If the build name is not recognised.
    """
    aliases = {
        'hg19': 'GRCh37', 'grch37': 'GRCh37', 'b37': 'GRCh37',
        'hg38': 'GRCh38', 'grch38': 'GRCh38', 'b38': 'GRCh38',
    }
    normalised = aliases.get(name.lower(), name)
    if normalised not in BUILDS:
        raise ValueError(
            f"Unsupported genome build '{name}'. "
            f"Supported: {', '.join(SUPPORTED_BUILDS)} "
            f"(aliases: hg19, hg38, b37, b38)"
        )
    return BUILDS[normalised]


def is_in_par(pos: int, chrom: str, build: GenomeBuild) -> bool:
    """
    Check if a position falls within a pseudoautosomal region.

    Parameters
    ----------
    pos : int
        Genomic position (1-based).
    chrom : str
        Harmonised chromosome name (e.g. 'X', 'Y').
    build : GenomeBuild
        Genome build to use for PAR coordinates.

    Returns
    -------
    bool
        True if the position is inside a PAR.
    """
    regions = build.par_regions.get(chrom, [])
    return any(start <= pos <= end for start, end in regions)


def is_sex_chrom(chrom: str, build: GenomeBuild) -> bool:
    """
    Check if a harmonised contig name is a sex chromosome.

    Parameters
    ----------
    chrom : str
        Harmonised chromosome name.
    build : GenomeBuild
        Genome build.

    Returns
    -------
    bool
        True if the chromosome is X or Y.
    """
    return chrom in build.sex_chroms


def is_autosomal(chrom: str, build: GenomeBuild) -> bool:
    """
    Check if a harmonised contig name is autosomal.

    Parameters
    ----------
    chrom : str
        Harmonised chromosome name.
    build : GenomeBuild
        Genome build.

    Returns
    -------
    bool
        True if the chromosome is autosomal (1-22).
    """
    return chrom in build.autosomal_chroms


def normalise_chrom(contig: str, build: GenomeBuild) -> str:
    """
    Normalise a contig name to the canonical form for this build.

    Extends the existing ``harmonize_contig()`` logic to also handle
    numeric sex chromosome aliases (23 -> X, 24 -> Y).

    Parameters
    ----------
    contig : str
        Raw contig name from VCF (e.g. 'chr1', 'chrX', '23').
    build : GenomeBuild
        Genome build (used for future build-specific contig rules).

    Returns
    -------
    str
        Canonical contig name (e.g. '1', 'X').
    """
    # Strip chr prefix first (existing logic)
    stripped = contig[3:] if contig.startswith('chr') else contig
    # Handle numeric aliases
    if stripped == '23':
        return 'X'
    if stripped == '24':
        return 'Y'
    return stripped
