"""Topsbm analysis helpers."""
import numpy as np
import pandas as pd
import toolz.curried as t
from research_daps.flows.topsbm.sbmtm import Sbmtm


@t.curry
def group_topsbm_by_sector(
    sector: pd.DataFrame, L: int, SIC_level: int, model: Sbmtm, cluster_type: str
) -> pd.DataFrame:
    """Group TopSBM cluster by SIC code.

    Args:
      sector: SIC lookup for Glass orgs
      L: TopSBM model level
      SIC_level: Number of digits of SIC code to aggregate at
      model: TopSBM model
      cluster_type: "document" (i.e. clusters of orgs) or "word" (i.e. topics)

    Returns:
      Rows: sector
      Columns: cluster index
      Values: prob. of a description in sector X being in cluster Y
    """
    if cluster_type not in {"document", "word"}:
        raise ValueError("`cluster_type` must be 'document' or 'word'")
    _, _, dict_groups = model.get_groups(L)
    p_tX_d = dict_groups["p_td_d" if cluster_type == "document" else "p_tw_d"]

    cluster_prob = pd.DataFrame(  # Rows: organisations | Columns: clusters | Values:
        p_tX_d.T, index=list(map(int, model.documents))
    )
    return (
        sector.merge(
            cluster_prob,
            left_on="org_id",
            right_index=True,
        )
        .melt(  # Columns: `SIC5_code`, `variable` (cluster number), `value` (cluster probability)
            id_vars=["SIC5_code"],
            value_vars=np.arange(p_tX_d.shape[0]),
        )
        # Slice SIC code to `SIC_level`
        .assign(sector=lambda x: x.SIC5_code.str.slice(0, SIC_level))
        .drop(["SIC5_code"], axis=1)
        .pipe(
            lambda x: pd.pivot_table(
                x,
                index="sector",
                columns="variable",
                values="value",
                aggfunc=np.mean,
            )
        )
    )
