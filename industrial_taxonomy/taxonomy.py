# Functions to build co-occurrence networks from company descriptions

from itertools import chain, combinations
import pandas as pd
import numpy as np
import altair as alt
import networkx as nx


def filter_keywords(
    kw_df: pd.DataFrame, length=1, to_remove=[], unspsc_filter=None, min_freq=5
):
    """Filters keyword / keyphrase frequency table
    Args:
        kw_df: kw frequencies and sources
        length: minimum length of a kw
        to_remove: list of terms to remove
        unspsc_filter: filter by UNSPSC
        min_freq: minimum occurrence of terms
    Returns:
        A filtered dataframe and a list of dropped kws
    """
    short = {word for word in kw_df["kw"] if len(str(word)) <= length}
    stop = {word for word in kw_df["kw"] if any(x in word for x in to_remove)}

    drop_words = short.union(stop)

    filtered = kw_df.loc[~kw_df["kw"].isin(drop_words)]

    if unspsc_filter is not None:
        filtered = filtered.query(f"in_unspsc=={unspsc_filter}")

    filtered = filtered.query(f"freqs>={min_freq}").reset_index(drop=True)

    return filtered, drop_words


def get_extraction_report(
    kw_df: pd.DataFrame, top_words=10, sector="sic4"
) -> pd.DataFrame:
    """Creates an extraction report (top kws) by sector and extraction method
    Args:dataframe with keyword frequencies
        top_words: top kws to report
        sector: variable with the sector
    """

    grouped = (
        kw_df.groupby([sector, "method"])
        .apply(lambda df: df.sort_values(by="freqs", ascending=False)[:top_words])
        .reset_index(drop=True)
    )
    return grouped


def plot_word_frequencies(kw_df: pd.DataFrame, height=700, sector="sic4") -> alt.Chart:
    """Plot word frequencies
    Args:
        kw_df: keyword frequencies
        height: chart height
        sector: industry variable
    """

    chart = (
        alt.Chart(kw_df)
        .mark_bar()
        .encode(
            y=alt.Y(
                "kw", sort=alt.EncodingSortField("freqs", "sum", order="descending")
            ),
            x="freqs",
            color="method",
            column=sector,
        )
        .properties(width=150, height=height)
    )

    return chart


def not_stop_kws(co_occ: list, thres=0.1) -> list:
    """Return keywords below a description occupation threshold
    Args:
        co_occ: co-occurrence list
        thres: maximum occurrence rate
        corpus_n: size of the corpus
    """
    freq = pd.Series(list(chain(*co_occ))).value_counts() / len(co_occ)

    to_keep = freq.loc[freq < thres].index.tolist()
    return to_keep


def make_network_from_coocc(co_occ: list, thres=0.1, spanning=True) -> nx.Network:
    """Create a network from a list of co-occurring terms
    Args
        co_occ: each element is a list of co-occurring entities
        thres: maximum occurrence rate
        spanning: filter the network with a maximum spanning tree
    """

    pairs = list(chain(*[sorted(list(combinations(x, 2))) for x in co_occ]))
    pairs = [x for x in pairs if len(x) > 0]

    edge_list = pd.DataFrame(pairs, columns=["source", "target"])

    edge_list["weight"] = 1

    edge_list_weighted = (
        edge_list.groupby(["source", "target"])["weight"].sum().reset_index(drop=False)
    )

    net = nx.from_pandas_edgelist(edge_list_weighted, edge_attr=True)

    to_keep = not_stop_kws(co_occ, thres=thres)

    sg = net.subgraph(to_keep)

    if spanning is True:
        return nx.maximum_spanning_tree(sg)
    else:
        return sg
