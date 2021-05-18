"""Look at heterogeneity of SIC codes using TopSBM topics."""
import logging
from typing import Dict, Optional

import altair as alt
import pandas as pd
import toolz.curried as t
import tqdm
from scipy.stats import entropy

from industrial_taxonomy.getters.sic import level_lookup
from industrial_taxonomy.getters.topsbm import get_topsbm
from industrial_taxonomy.queries.sector import get_glass_SIC_sectors
from industrial_taxonomy.utils.altair_s3 import export_chart
from industrial_taxonomy.utils.econ_geo import location_quotient
from industrial_taxonomy.scripts.topsbm_utils import (
    group_topsbm_by_sector as group_topsbm_by_sector_,
)


alt.data_transformers.disable_max_rows()
logger = logging.getLogger(__name__)

SIC_LEVEL = 4

TopWords = Dict[str, str]


def _melt(data, SIC_level):
    """Melt and format data indexed by sector across variables."""
    return (
        data.melt(ignore_index=False)  # Melt for altair
        .rename_axis(index="sector")
        .reset_index()
        # Labels
        .assign(
            sector_name=lambda x: x.sector
            + ": "
            + x.sector.map(level_lookup(SIC_level))
        )
        .dropna()  # Currently 98000 and 99999 codes
    )


def plot_sector_topic_activity(
    topic_activity_by_sector: pd.DataFrame,
    sectors: pd.DataFrame,
    top_words: Optional[TopWords] = None,
    value_name: str = "Topic activity",
) -> alt.Chart:
    """Bar chart of topic activity in `sectors`.

    Args:
        TODO

    Returns:
        Chart
    """
    L = len(sectors[0])
    return (
        alt.Chart(
            topic_activity_by_sector.loc[sectors]
            .melt(ignore_index=False, value_name=value_name)
            .pipe(
                lambda df: df
                if not top_words
                else df.assign(top_words=lambda x: x.variable.map(top_words))
            )
            .reset_index()
            .assign(
                sector_name=lambda x: x.sector + ": " + x.sector.map(level_lookup(L))
            )
            .dropna()
        )
        .mark_bar()
        .encode(
            y=alt.Y(value_name, title=value_name),
            x=alt.X(
                "variable:N",
                title="Topic number",
                sort="-y",
            ),
            color=alt.Color(
                "sector_name:N",
                title="Sector name",
                scale=alt.Scale(scheme="category20b"),
            ),
            tooltip=[
                alt.Tooltip("variable", title="Topic number"),
                alt.Tooltip(value_name, title="Topic activity"),
                alt.Tooltip("sector_name:N", title="Sector"),
                alt.Tooltip("top_words:N", title="Top words"),
            ],
        )
    )


def plot_sector_entropy_interactive(
    sector_entropy: pd.DataFrame, top_words: Optional[TopWords] = None
) -> alt.Chart:
    """Minimap bar chart of SIC entropy.

    Args:
        sector_entropy: Index is SIC codes, columns are entropy across model levels.
        top_words: Keys are SIC codes, values are string list of top words.

    Returns:
        Chart
    """
    SIC_level = len(sector_entropy.index[0])

    label_size = 200
    brush = alt.selection_interval(encodings=["y"])
    base = alt.Chart(
        sector_entropy.pipe(_melt, SIC_level).pipe(
            lambda df: df
            if not top_words
            else df.assign(top_words=lambda x: x.sector.map(top_words))
        )
    ).mark_bar()
    sort = alt.EncodingSortField(field="value", op="max", order="descending")

    minimap = (
        base.add_selection(brush)
        .encode(
            x=alt.X("value", axis=None), y=alt.Y("sector_name", sort=sort, axis=None)
        )
        .properties(height=200, width=50, title=["Minimap -- click", " & drag to zoom"])
    )

    chart = (
        base.encode(
            y=alt.Y(
                "sector_name",
                sort=sort,
                axis=alt.Axis(
                    labelOverlap="greedy",
                    title=None,
                    labelAlign="left",
                    labelPadding=label_size,
                    labelLimit=label_size,
                ),
            ),
            x=alt.X("value", title="Entropy"),
            color=alt.Color("variable", title="Model hierarchy level"),
            tooltip=[
                alt.Tooltip("sector_name", title="Sector"),
                alt.Tooltip("variable", title="Model hierarchy level"),
                alt.Tooltip("value", title="entropy"),
                alt.Tooltip(
                    "top_words", title="Top words of most differentiated topic"
                ),
            ],
        )
        .transform_filter(brush)
        .add_selection(alt.selection_single())
        .properties(height=600)
    )

    return chart | minimap


def plot_sector_entropy_static(sector_entropy: pd.DataFrame, n: int = 10) -> alt.Chart:
    """Bar chart of SIC entropy for top `n` and bottom `n` sectors.

    Args:
        sector_entropy: Index is SIC codes, columns are entropy across model levels.
        n: How many top or bottom sectors to show.

    Returns:
        Chart
    """

    label_size = 200
    SIC_level = len(sector_entropy.index[0])

    top_data = (
        sector_entropy.assign(agg_value=lambda x: x.max(axis=1))
        .sort_values("agg_value", ascending=False)
        .head(n)
        .drop(["agg_value"], axis=1)
        .pipe(_melt, SIC_level)
    )
    bottom_data = (
        sector_entropy.assign(agg_value=lambda x: x.max(axis=1))
        .sort_values("agg_value", ascending=False)
        .tail(n)
        .drop(["agg_value"], axis=1)
        .pipe(_melt, SIC_level)
    )

    sort = alt.EncodingSortField(field="value", op="max", order="descending")

    chart = (
        alt.Chart()
        .mark_bar()
        .encode(
            y=alt.Y(
                "sector_name",
                sort=sort,
                axis=alt.Axis(
                    labelAlign="left",
                    labelPadding=label_size,
                    labelLimit=label_size,
                    title=None,
                ),
            ),
            x=alt.X("value", title="Entropy"),
            color=alt.Color("variable", title="Model hierarchy level"),
            tooltip=[
                alt.Tooltip("sector_name", title="Sector"),
                alt.Tooltip("variable", title="Model hierarchy level"),
                alt.Tooltip("value", title="entropy"),
            ],
        )
    )

    return (
        chart.properties(data=top_data, title=f"Top {n} entropy sectors")
        & chart.properties(data=bottom_data, title=f"Bottom {n} entropy sectors")
    ).resolve_axis(x="shared")


if __name__ == "__main__":
    group_topsbm_by_sector = group_topsbm_by_sector_(cluster_type="word")

    model = get_topsbm()
    sector = get_glass_SIC_sectors()

    model_levels = [0, 1, 2]
    model_level = 1

    # %%
    # Pre-calc topic activity and entropy
    topic_activity_by_sector: dict = {
        L: group_topsbm_by_sector(sector, L, SIC_LEVEL, model)
        for L in tqdm.tqdm(model_levels)
    }
    sector_entropy = pd.concat(
        [
            data.apply(entropy, axis=1).to_frame(f"Level {level}")
            for level, data in topic_activity_by_sector.items()
        ],
        axis=1,
    )
    # %%
    # Top words of topic that most differentiated sector - model level `model_level`
    n_words = 20
    top_words = {
        k: ", ".join(map(t.first, v))
        for k, v in model.topics(model_level, n_words).items()
    }
    top_words_of_sector = (
        topic_activity_by_sector[model_level]
        .pipe(location_quotient)
        .idxmax(axis=1)
        .map(top_words)
        .pipe(dict)
    )

    # %%
    # Plot entropy of SIC codes across `model_levels`
    export_chart(
        plot_sector_entropy_interactive(sector_entropy, top_words_of_sector),
        static_alt_chart=plot_sector_entropy_static(sector_entropy),
        key=f"topsbm/SIC{SIC_LEVEL}_entropy",
    )

    # %%
    # Look at the topics of the most heterogeneous sectors
    n_top_sectors = 22  # 9999 and 7499 don't count

    top_sectors = (
        sector_entropy.assign(
            agg_value=lambda x: x.max(axis=1),  # How we agg across model levels
            sector=lambda x: x.index.str.slice(0, SIC_LEVEL),
        )
        .sort_values("agg_value", ascending=False)
        .head(n_top_sectors)
        .drop(["agg_value"], axis=1)
        .sector.values
    )

    value_name = "Topic activity (fraction of mean)"

    chart = (
        plot_sector_topic_activity(
            topic_activity_by_sector[model_level].pipe(location_quotient),
            top_sectors,
            top_words,
            value_name,
        )
        # .transform_joinaggregate(lq=f"sum({value_name})", groupby=["variable"])
        # .transform_filter((alt.datum.lq > 15))
        .properties(title="Most heterogeneous", width=800)
        .resolve_legend("independent")
        .resolve_scale(color="independent")
    )
    export_chart(chart, key=f"topsbm/SIC{SIC_LEVEL}_hetero_topicdist_L{model_level}")
    # %%
