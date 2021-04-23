# %%
"""Look at heterogeneity of SIC codes using TopSBM topics.

TODO: LOTS OF REFACTORING - this is essentially a notebook
"""
from collections import defaultdict
from itertools import product

import altair as alt
import numpy as np
import pandas as pd
import toolz.curried as t
import tqdm
from IPython.display import display
from research_daps.flows.topsbm.sbmtm import sbmtm as Sbmtm

from scipy.stats import entropy

from industrial_taxonomy.getters.sic import level_lookup
from industrial_taxonomy.getters.topsbm import get_topsbm_v2
from industrial_taxonomy.queries.sector import get_glass_SIC_sectors
from industrial_taxonomy.utils.altair_s3 import export_chart

# %%


@t.curry
def group_topsbm_by_sector(
    sector: pd.DataFrame, L: int, SIC_level: int, model: Sbmtm
) -> pd.DataFrame:
    """Group TopSBM topic by SIC code.

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
    p_td_d = dict_groups["p_tw_d"]

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


alt.data_transformers.disable_max_rows()
# %%

SIC_levels = [2, 3, 4]

model = get_topsbm_v2()
sector = get_glass_SIC_sectors()
sic_entropy: dict = defaultdict(lambda: defaultdict(lambda: {}))
topic_activity_by_sector: dict = defaultdict(lambda: defaultdict(lambda: {}))

# %%

for L, SIC_level in tqdm.tqdm(product(range(0, 3), SIC_levels)):
    topic_activity_by_sector[SIC_level][L] = group_topsbm_by_sector(
        sector, L, SIC_level, model
    )
    sic_entropy[SIC_level][L] = topic_activity_by_sector[SIC_level][L].apply(
        entropy, axis=1
    )

# %%


def plot_interactive(data):
    """Minimap bar chart of SIC entropy."""

    label_size = 200
    brush = alt.selection_interval(encodings=["y"])
    base = alt.Chart(data.pipe(_melt)).mark_bar()
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
            ],
        )
        .transform_filter(brush)
        .add_selection(alt.selection_single())
        .properties(
            # title=f"SIC level {SIC_level}",
            height=600
        )
    )

    return chart | minimap


def _melt(data):
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


def plot_static(data, n=10):
    """Minimap bar chart of SIC entropy."""

    label_size = 200

    top_data = (
        data.assign(agg_value=lambda x: x.max(axis=1))
        .sort_values("agg_value", ascending=False)
        .head(n)
        .drop(["agg_value"], axis=1)
        .pipe(_melt)
    )
    bottom_data = (
        data.assign(agg_value=lambda x: x.max(axis=1))
        .sort_values("agg_value", ascending=False)
        .tail(n)
        .drop(["agg_value"], axis=1)
        .pipe(_melt)
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


for SIC_level in tqdm.tqdm(SIC_levels):
    data = pd.concat(
        [
            data.to_frame(f"Level {level}")
            for level, data in sic_entropy[SIC_level].items()
        ],
        axis=1,
    )

    export_chart(
        plot_interactive(data),
        static_alt_chart=plot_static(data),
        key=f"topsbm/SIC{SIC_level}_entropy",
    )

# %%


def sector_topic_activity(topic_activity_by_sector, sectors):
    """Bar chart of topic activity in `sectors`."""
    L = len(sectors[0])
    return (
        alt.Chart(
            topic_activity_by_sector.loc[sectors]
            .melt(ignore_index=False)
            .reset_index()
            .assign(
                sector_name=lambda x: x.sector + ": " + x.sector.map(level_lookup(L))
            )
            .dropna()
        )
        .mark_bar()
        .encode(
            y=alt.Y("value:Q", title="Topic activity"),
            x=alt.X("variable:N", title="Topic number"),
            color="sector_name:N",
            tooltip=[
                alt.Tooltip("variable", title="Topic number"),
                alt.Tooltip("value", title="Topic activity"),
                alt.Tooltip("sector_name:N", title="Sector"),
            ],
        )
    )


n = 10
SIC_level = 2
L = 1

top_sectors = (
    data.assign(
        agg_value=lambda x: x.max(axis=1),
        sector=lambda x: x.index.str.slice(0, SIC_level),
    )
    .sort_values("agg_value", ascending=False)
    .head(n)
    .drop(["agg_value"], axis=1)
    .sector.values
)
bottom_sectors = (
    data.assign(
        agg_value=lambda x: x.max(axis=1),
        sector=lambda x: x.index.str.slice(0, SIC_level),
    )
    .sort_values("agg_value", ascending=False)
    .tail(n)
    .drop(["agg_value"], axis=1)
    .sector.values
)


chart = (
    (
        sector_topic_activity(
            topic_activity_by_sector[SIC_level][L], top_sectors
        ).properties(title="Most heterogeneous", width=800)
        & sector_topic_activity(
            topic_activity_by_sector[SIC_level][L], bottom_sectors
        ).properties(title="Most homogeneous", width=800)
    )
    .resolve_legend("independent")
    .resolve_scale(color="independent")
)
display(chart)

export_chart(chart, key=f"topsbm/SIC{SIC_level}_entropy_topicdist_L{L}")
# %%
