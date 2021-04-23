"""Generate heatmaps showing cosine similarity between SIC divisions.

Similarity calculated according to the document clusters generated by two
TopSBM model pipelines.
"""
from itertools import product

import altair as alt
import numpy as np
import pandas as pd
import toolz.curried as t
from IPython.display import display
from research_daps.flows.topsbm.sbmtm import sbmtm

Sbmtm = sbmtm
from scipy.spatial.distance import pdist, squareform

from industrial_taxonomy.getters.sic import division_lookup
from industrial_taxonomy.getters.topsbm import get_topsbm_v1, get_topsbm_v2
from industrial_taxonomy.queries.sector import get_glass_SIC_sectors
from industrial_taxonomy.utils.altair_s3 import export_chart


@t.curry
def get_sic_similarity(df: pd.DataFrame, metric: str = "cosine") -> pd.DataFrame:
    """Compute pairwise similarity between rows of `df`, output in long form."""

    return (
        pd.DataFrame(
            1 - squareform(pdist(df, metric=metric)), index=df.index, columns=df.index
        )
        .melt(ignore_index=False)
        .rename_axis(index="sector_")
        .reset_index()
    )


@t.curry
def group_topsbm_by_sector(
    sector: pd.DataFrame, L: int, SIC_level: int, model: Sbmtm
) -> pd.DataFrame:
    """Group TopSBM cluster by SIC code.

    Args:
      sector: SIC lookup for Glass orgs
      L: TopSBM model level
      SIC_level: Number of digits of SIC code to aggregate at

    Returns:
      Rows: sector
      Columns: cluster index
      Values: prob. of a description in sector X being in cluster Y
    """
    # _, _, dict_groups = model.get_groups(L)
    dict_groups = model.get_groups(L)
    p_td_d = dict_groups["p_td_d"]

    cluster_prob = pd.DataFrame(  # Rows: organisations | Columns: clusters | Values:
        p_td_d.T, index=list(map(int, model.documents))
    )
    return (
        sector.merge(
            cluster_prob,
            left_on="org_id",
            right_index=True,
        )
        .melt(  # Columns: `SIC5_code`, `variable` (cluster number), `value` (cluster probability)
            id_vars=["SIC5_code"],
            value_vars=np.arange(p_td_d.shape[0]),
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


@t.curry
def plot_sic_similarity(sic_similarity: pd.DataFrame) -> alt.Chart:
    """Heatmap of similarity between SIC codes."""
    label_size = 200
    divs = division_lookup()

    has_negatives = sic_similarity.value.min() < 0

    return (
        alt.Chart(
            sic_similarity.assign(
                sector_name=lambda x: x.sector + ": " + x.sector.map(divs),
                sector_name_=lambda x: x.sector_ + ": " + x.sector_.map(divs),
            )
        )
        .mark_rect()
        .encode(
            x=alt.X(
                "sector:O",
                title="SIC division",
            ),
            y=alt.Y(
                "sector_name_:O",
                axis=alt.Axis(
                    labelAlign="left", labelPadding=label_size, labelLimit=label_size
                ),
                title="SIC division",
            ),
            color=alt.Color(
                "value:Q",
                scale=alt.Scale(
                    scheme="spectral" if has_negatives else "yellowgreenblue"
                ),
            ),
            tooltip=[
                alt.Tooltip("sector_name_", title="sector name (row)"),
                alt.Tooltip("sector_name", title="sector name (col)"),
                alt.Tooltip("value", title="Cosine similarity"),
            ],
        )
        .properties(height=800, width=800)
    )


if __name__ == "__main__":
    alt.data_transformers.disable_max_rows()

    model_levels = 3
    SIC_level = 2
    sector = get_glass_SIC_sectors()
    models = {"spacy": get_topsbm_v2(), "simple": get_topsbm_v1()}

    similarities = {}
    for L, model_name in product(range(model_levels), models.keys()):
        similarities[(L, model_name)] = t.pipe(
            models[model_name],
            group_topsbm_by_sector(sector, L, SIC_level),
            get_sic_similarity(metric="cosine"),
        )

        t.pipe(
            similarities[(L, model_name)],
            plot_sic_similarity,
            t.curry(
                export_chart,
                key=f"topsbm/SIC{SIC_level}_similarity_{model_name}-model_L{L}",
            ),
        )

    # Differences
    for L in range(model_levels):
        t.pipe(
            (
                similarities[(L, "spacy")].set_index(["sector_", "sector"])
                - similarities[(L, "simple")].set_index(["sector_", "sector"])
            ).reset_index(),
            plot_sic_similarity,
            t.curry(
                export_chart,
                key=f"topsbm/SIC{SIC_level}_similarity_difference_L{L}",
            ),
        )
