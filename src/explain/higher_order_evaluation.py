"""
Evaluation of higher-order interaction discovery against a known architecture.

Kept apart from :mod:`src.explain.higher_order` so that the discovery stages
cannot reach a truth file: candidate generation stays a function of the model
and the cohort alone, and only this module reads what the generating
architecture was.

Reported quantities:

- the rank at which each true set first appears among the emitted candidates,
- the proportion of true sets recovered at each order,
- the false discovery rate among the candidates surviving the counterfactual.

A cohort simulated without interactions fixes the false-positive rate, since
every set it emits is spurious.

Author: Francesco Lescai
"""


from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd

VariantKey = Tuple[int, int]


def load_truth_sets(
    path: str | Path,
    gene_index: Optional[Dict[str, int]] = None,
) -> List[Tuple[VariantKey, ...]]:
    """
    Read the generating architecture as sets of variant keys.

    Parameters
    ----------
    path : str or Path
        CSV in long form with one row per locus and columns ``set_id`` and
        ``pos``, plus either ``gene_id`` (the run's internal gene index) or
        ``gene`` (a gene symbol resolved through ``gene_index``).
    gene_index : Optional[Dict[str, int]]
        Symbol to index mapping of the run, required when the file carries
        symbols rather than indices.

    Returns
    -------
    List[Tuple[VariantKey, ...]]
        One sorted tuple of ``(pos, gene_id)`` per true set.

    Raises
    ------
    ValueError
        When the required columns are absent, or symbols are supplied with no
        gene index to resolve them.
    """
    frame = pd.read_csv(path)

    missing = {'set_id', 'pos'} - set(frame.columns)
    if missing:
        raise ValueError(
            f"Truth file is missing required column(s): {', '.join(sorted(missing))}"
        )

    if 'gene_id' in frame.columns:
        gene_ids = frame['gene_id'].astype(int)
    elif 'gene' in frame.columns:
        if gene_index is None:
            raise ValueError(
                "Truth file gives gene symbols but no gene index was supplied "
                "to resolve them."
            )
        unknown = sorted(set(frame['gene']) - set(gene_index))
        if unknown:
            raise ValueError(
                f"Truth file names gene(s) absent from the run's gene index: "
                f"{', '.join(unknown[:5])}"
            )
        gene_ids = frame['gene'].map(gene_index).astype(int)
    else:
        raise ValueError("Truth file needs either a 'gene_id' or a 'gene' column.")

    frame = frame.assign(_gene_id=gene_ids)

    truth_sets: List[Tuple[VariantKey, ...]] = []
    for _set_id, group in frame.groupby('set_id', sort=True):
        members = tuple(sorted(
            (int(pos), int(gene)) for pos, gene in
            zip(group['pos'], group['_gene_id'])
        ))
        if len(members) >= 2:
            truth_sets.append(members)

    return truth_sets


def parse_candidate_members(members: str) -> Tuple[VariantKey, ...]:
    """
    Parse the ``pos:gene`` member string written for each candidate set.

    Parameters
    ----------
    members : str
        Semicolon-separated ``pos:gene_id`` entries.

    Returns
    -------
    Tuple[VariantKey, ...]
        Sorted variant keys.
    """
    keys = []
    for token in str(members).split(';'):
        token = token.strip()
        if not token:
            continue
        pos, gene = token.split(':')
        keys.append((int(pos), int(gene)))
    return tuple(sorted(keys))


def evaluate_recovery(
    candidates: pd.DataFrame,
    truth_sets: Sequence[Tuple[VariantKey, ...]],
    rank_column: str = 'density_gap',
) -> pd.DataFrame:
    """
    Rank at which each true set first appears among the emitted candidates.

    A true set is credited to a candidate when the candidate's members are
    exactly the true set, and separately when the candidate merely contains
    it, so that partial recovery at a higher order is visible rather than
    counted as a miss.

    Parameters
    ----------
    candidates : pd.DataFrame
        Emitted candidates with a ``members`` column and ``rank_column``.
    truth_sets : Sequence[Tuple[VariantKey, ...]]
        The generating architecture.
    rank_column : str
        Column the candidate list is ranked by, decreasing.

    Returns
    -------
    pd.DataFrame
        One row per true set: ``order``, ``exact_rank``, ``containing_rank``,
        ``recovered_exact``, ``recovered_containing``, ``synergy`` and
        ``is_significant`` of the matching candidate where one exists.
        Ranks are one-based; a miss is recorded as a missing rank.
    """
    ranked = candidates.sort_values(rank_column, ascending=False).reset_index(drop=True)
    parsed = [parse_candidate_members(m) for m in ranked['members']]
    parsed_as_sets = [set(p) for p in parsed]

    rows = []
    for truth in truth_sets:
        truth_set = set(truth)
        exact_rank = None
        containing_rank = None
        matched_row = None

        for rank, (members, member_set) in enumerate(zip(parsed, parsed_as_sets), start=1):
            if containing_rank is None and truth_set <= member_set:
                containing_rank = rank
            if exact_rank is None and members == truth:
                exact_rank = rank
                matched_row = ranked.iloc[rank - 1]
            if exact_rank is not None and containing_rank is not None:
                break

        row = {
            'truth_set': ';'.join(f"{pos}:{gene}" for pos, gene in truth),
            'order': len(truth),
            'exact_rank': exact_rank,
            'containing_rank': containing_rank,
            'recovered_exact': exact_rank is not None,
            'recovered_containing': containing_rank is not None,
        }
        if matched_row is not None:
            row['synergy'] = matched_row.get('synergy')
            row['is_significant'] = bool(matched_row.get('is_significant', False))
        else:
            row['synergy'] = float('nan')
            row['is_significant'] = False

        rows.append(row)

    return pd.DataFrame(rows)


def recovery_by_order(recovery: pd.DataFrame) -> pd.DataFrame:
    """
    Proportion of true sets recovered at each order.

    Parameters
    ----------
    recovery : pd.DataFrame
        Output of :func:`evaluate_recovery`.

    Returns
    -------
    pd.DataFrame
        Per order: ``n_true``, ``n_recovered_exact``, ``n_recovered_containing``,
        the corresponding proportions, and the median exact rank of the sets
        that were recovered.
    """
    if recovery.empty:
        return pd.DataFrame(columns=[
            'order', 'n_true', 'n_recovered_exact', 'n_recovered_containing',
            'proportion_exact', 'proportion_containing', 'median_exact_rank',
        ])

    rows = []
    for order, group in recovery.groupby('order', sort=True):
        n_true = len(group)
        n_exact = int(group['recovered_exact'].sum())
        n_containing = int(group['recovered_containing'].sum())
        recovered_ranks = group.loc[group['recovered_exact'], 'exact_rank']
        rows.append({
            'order': int(order),
            'n_true': n_true,
            'n_recovered_exact': n_exact,
            'n_recovered_containing': n_containing,
            'proportion_exact': n_exact / n_true,
            'proportion_containing': n_containing / n_true,
            'median_exact_rank': float(recovered_ranks.median()) if n_exact else float('nan'),
        })

    return pd.DataFrame(rows)


def false_discovery_rate(
    candidates: pd.DataFrame,
    truth_sets: Sequence[Tuple[VariantKey, ...]],
) -> Dict[str, float]:
    """
    False discovery rate among the candidates surviving the counterfactual.

    A surviving candidate is a true discovery when its members are exactly a
    true set. With no truth sets supplied the cohort carries no interactions,
    every survivor is spurious, and the rate returned is the false-positive
    rate of the procedure.

    Parameters
    ----------
    candidates : pd.DataFrame
        Emitted candidates with ``is_significant``, ``status`` and ``members``.
    truth_sets : Sequence[Tuple[VariantKey, ...]]
        The generating architecture, possibly empty.

    Returns
    -------
    Dict[str, float]
        ``n_tested``, ``n_surviving``, ``n_true_positive``, ``n_false_positive``
        and ``false_discovery_rate`` (NaN when nothing survives).
    """
    truth_lookup = {tuple(sorted(t)) for t in truth_sets}

    tested = candidates[candidates.get('status', 'tested') == 'tested']
    surviving = tested[tested['is_significant'].astype(bool)]

    n_true_positive = 0
    for members in surviving['members']:
        if parse_candidate_members(members) in truth_lookup:
            n_true_positive += 1

    n_surviving = len(surviving)
    n_false_positive = n_surviving - n_true_positive

    return {
        'n_tested': int(len(tested)),
        'n_surviving': int(n_surviving),
        'n_true_positive': int(n_true_positive),
        'n_false_positive': int(n_false_positive),
        'false_discovery_rate': (
            float(n_false_positive / n_surviving) if n_surviving else float('nan')
        ),
    }


def evaluate(
    candidates: pd.DataFrame,
    truth_sets: Sequence[Tuple[VariantKey, ...]],
    rank_column: str = 'density_gap',
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float]]:
    """
    Run the whole of Stage 5 over one emitted candidate table.

    Parameters
    ----------
    candidates : pd.DataFrame
        Emitted candidates.
    truth_sets : Sequence[Tuple[VariantKey, ...]]
        The generating architecture, possibly empty.
    rank_column : str
        Column the candidate list is ranked by.

    Returns
    -------
    recovery : pd.DataFrame
        Per true set, from :func:`evaluate_recovery`.
    by_order : pd.DataFrame
        Per order, from :func:`recovery_by_order`.
    discovery_rates : Dict[str, float]
        From :func:`false_discovery_rate`.
    """
    recovery = evaluate_recovery(candidates, truth_sets, rank_column=rank_column)
    by_order = recovery_by_order(recovery)
    rates = false_discovery_rate(candidates, truth_sets)
    return recovery, by_order, rates
