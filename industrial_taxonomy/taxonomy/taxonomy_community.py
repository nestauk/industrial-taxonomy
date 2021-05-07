"""Functions to build networks and extract commumities
"""
import numpy as np
import logging
from itertools import chain, combinations
import pandas as pd
import networkx as nx
from cdlib import ensemble, evaluation


def not_stop_kws(co_occ: list, thres: str = 0.1) -> list:
    """Return keywords below a description occupation threshold
    Args:
        co_occ: co-occurrence list
        thres: maximum occurrence rate
        corpus_n: size of the corpus
    Return:
        list with terms to keep
    """
    freq = pd.Series(list(chain(*co_occ))).value_counts() / len(co_occ)

    to_keep = freq.loc[freq < thres].index.tolist()
    return to_keep


def make_network_from_coocc(
    co_occ: list, thres: float = 0.1, extra_links: int = 200, spanning: bool = True
) -> nx.Graph:
    """Create a network from a list of co-occurring terms
    Args
        co_occ: each element is a list of co-occurring entities
        thres: maximum occurrence rate
        weight_thres: extra edges to add
        spanning: filter the network with a maximum spanning tree
    """

    # Make weighted edge list
    pairs = list(chain(*[sorted(list(combinations(x, 2))) for x in co_occ]))
    pairs = [x for x in pairs if len(x) > 0]

    edge_list = pd.DataFrame(pairs, columns=["source", "target"])

    edge_list["weight"] = 1

    edge_list_weighted = (
        edge_list.groupby(["source", "target"])["weight"].sum().reset_index(drop=False)
    )

    # Make and post-process network
    net = nx.from_pandas_edgelist(edge_list_weighted, edge_attr=True)

    to_keep = not_stop_kws(co_occ, thres=thres)

    net_filt = net.subgraph(to_keep)

    if spanning is True:
        msp = nx.maximum_spanning_tree(net_filt)
        msp_plus = make_msp_plus(net_filt, msp, thres=extra_links)
        return msp_plus

    else:
        return net_filt


def make_msp_plus(net: nx.Graph, msp: nx.Graph, thres: int = 200) -> nx.Graph:
    """Create a network combining maximum spanning tree and top edges
    Args:
        net: original network
        msp: maximum spanning tree of the original network
        thres: extra edges to aadd
    Returns:
        A network
    """

    msp_ed = set(msp.edges())

    top_edges_net = nx.Graph(
        [
            x
            for x in sorted(
                net.edges(data=True),
                key=lambda x: x[2]["weight"],
                reverse=True,
            )
            if (x[0], x[1]) not in msp_ed
        ][:thres]
    )

    # Combines them
    united_graph = nx.Graph(
        list(msp.edges(data=True)) + list(top_edges_net.edges(data=True))
    )
    return united_graph


def community_grid_search(
    net: nx.Graph,
    method_dict: dict,
    qual=evaluation.erdos_renyi_modularity,
    aggregate=max,
):
    """Grid search community detection algorithms over a graph
    Args:
        net: graph
        method_dict: dictionary where keys are algorithms and values are parameters
        qual: evaluation metric
        aggregate: criterion for selecting the evaluation metric
    Returns:
        A table of results and a dict with the best performing community
    """

    algos = list(method_dict.keys())
    pars = list(method_dict.values())

    results = ensemble.pool_grid_filter(net, algos, pars, qual, aggregate=aggregate)

    results_container = []

    comm_assignments_container = {}

    for comm, score in results:
        try:
            logging.info(comm.method_name)
            out = [
                comm.method_name,
                len(comm.communities),
                comm.method_parameters,
                score.score,
                comm,
            ]
            results_container.append(out)
            comm_assignments_container[comm.method_name] = comm.communities
        except:
            logging.warning(f"error with algorithm {logging.info(comm.method_name)}")
            pass

    results_df = pd.DataFrame(
        results_container,
        columns=["method", "comm_n", "parametres", "score", "instance"],
    )
    return results_df, comm_assignments_container


def evaluate_similarity(result_df: pd.DataFrame, top: int = 3) -> pd.DataFrame:
    """Evaaluates similarity in community partitions for top algorithms
    Args:
        result_df: a summary table
        top: top algorithms to consider
    Returns:
        a df with comparisons
    """

    name_instance_lu = result_df.set_index("method")["instance"].to_dict()

    top_algos = result_df.sort_values("score", ascending=False)["method"].tolist()[
        : top + 1
    ]

    prod = combinations(top_algos, 2)

    comparison_container = []

    for pair in prod:

        insts = [name_instance_lu[meth] for meth in pair]
        try:
            comp = evaluation.normalized_mutual_information(*insts)
            comparison_container.append([pair[0], pair[1], comp.score])
        except ValueError:
            # TODO write script to remove not overlapping tokens
            logging.warning("misalignment in number of communities betwween algorithms")
            comparison_container.append([pair[0], pair[1], np.nan])

    comp_df = pd.DataFrame(
        comparison_container, columns=["method_1", "method_2", "score"]
    )
    return comp_df


def extract_sector_communities(glass_sector: pd.DataFrame, sic: str, method_pars: dict):
    """Extracts sector communities
    Args:
        glass_sector: glass companies
        sic: sic4 to focus on
        method_pars: dictionary with algorithms and parameters
    Returns:
        best communities, a table of similarities for best performing algorithms
        and a summary table of results
    """

    obs = glass_sector.query(f"sic4=='{sic}'").reset_index(drop=True)

    co_occ = make_network_from_coocc(
        obs["token_filtered"], thres=0.2, extra_links=20, spanning=True
    )

    summary_table, comms = community_grid_search(co_occ, method_pars)

    similarity_table = evaluate_similarity(summary_table)

    top_method = summary_table.loc[summary_table["score"].idxmax(), "method"]

    # sort_values("score", ascending=False)["method"][0]

    selected_comms = {
        f"{sic}_{str(n)}": set(comm) for n, comm in enumerate(comms[top_method])
    }

    return (
        co_occ,
        selected_comms,
        similarity_table.assign(sic4=sic),
        summary_table.assign(sic4=sic),
    )


def merge_container(comm_container: list) -> dict:
    """Flattens a list of dictionaries into a dictionary"""
    merged_communities = {}
    for _dict in comm_container:
        for k, v in _dict.items():
            merged_communities[k] = list(v)
    return merged_communities


def extract_communities(
    glass_sector: pd.DataFrame, sector_list: list, method_pars: dict
):
    """Extracts community outputs for a list of sectors
    Args:
        glass_sector: glass companies
        sector_list: list of sectors to extract communities for
        method_pars: methods to use
    Returns: a container of networks and communities and tables with summaries
    """

    network_container = []
    comm_container = []
    similarity_container = []
    summary_container = []

    for sic in sector_list:
        logging.info(sic)

        cocc, comms, similarity, summary = extract_sector_communities(
            glass_sector, sic, method_pars
        )

        comm_container.append(comms)
        similarity_container.append(similarity)
        summary_container.append(summary)

    merged_comms = merge_container(comm_container)
    similarity_df = pd.concat(similarity_container).reset_index(drop=True)
    summary_df = pd.concat(summary_container).reset_index(drop=True)

    return network_container, merged_comms, similarity_df, summary_df
