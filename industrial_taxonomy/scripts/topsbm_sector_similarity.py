"""Generate heatmaps showing cosine similarity between SIC divisions.

Similarity calculated according to the document clusters generated by two
TopSBM model pipelines.
"""
import logging

import altair as alt
import pandas as pd
import toolz.curried as t
from scipy.spatial.distance import pdist, squareform

from industrial_taxonomy.getters.sic import level_lookup
from industrial_taxonomy.getters.topsbm import get_topsbm_v2
from industrial_taxonomy.queries.sector import get_glass_SIC_sectors
from industrial_taxonomy.utils.altair_s3 import export_chart
from topsbm_utils import group_topsbm_by_sector as group_topsbm_by_sector_

group_topsbm_by_sector = group_topsbm_by_sector_(cluster_type="word")
logger = logging.getLogger(__name__)


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
def plot_sic_similarity(sic_similarity: pd.DataFrame) -> alt.Chart:
    """Heatmap of similarity between SIC codes."""
    label_size = 200
    divs = level_lookup(len(sic_similarity.sector.iloc[0]))

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
    group_topsbm_by_sector = group_topsbm_by_sector_(cluster_type="document")

    model_levels = [1]
    SIC_level = 2
    sector = get_glass_SIC_sectors()
    model = get_topsbm_v2()
    model_name = "spacy"

    similarities = {}
    for L in model_levels:
        logger.info(f"{model_name} level {L}")

        similarities[(L, model_name)] = t.pipe(
            model,
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
