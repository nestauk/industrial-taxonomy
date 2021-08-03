# final report results of the taxonomy analysis

import logging
import os
import pickle
from itertools import combinations, chain

import graph_tool.all as gt
import networkx as nx
import numpy as np
import pandas as pd
from cdlib import algorithms
from numpy.random import choice
from scipy.stats import zscore


from industrial_taxonomy import project_dir
from industrial_taxonomy.altair_network import plot_altair_network
from industrial_taxonomy.getters.glass import get_organisation_description
from industrial_taxonomy.getters.processing import get_firm_embeddings
from industrial_taxonomy.scripts.extract_communities import get_glass_tokenised
from industrial_taxonomy.taxonomy.taxonomy_community import make_network_from_coocc
import altair as alt
from industrial_taxonomy.getters.processing import (
    get_company_sector_lookup,
    get_sector_name_lookup,
    get_sector_reassignment_outputs,
)

from industrial_taxonomy.utils.altair_save_utils import (
    google_chrome_driver_setup,
    altair_text_resize,
    save_altair,
)
from industrial_taxonomy.taxonomy.taxonomy_filtering import (
    make_doc_term_matrix,
    make_tfidf_mat,
    get_promo_terms,
)
from industrial_taxonomy.utils.sic_utils import (
    extract_sic_code_description,
    load_sic_taxonomy,
    section_code_lookup,
)


# Set up
alt.data_transformers.disable_max_rows()


def make_sic_lookups_extra():
    """Creates lookups between section numbers, names and divisions"""
    section_name_lookup = {
        k: k + ": " + v
        for k, v in extract_sic_code_description(load_sic_taxonomy(), "SECTION").items()
    }
    division_section_names = {
        k: section_name_lookup[v.split(" :")[0]]
        for k, v in section_code_lookup().items()
    }

    sic_4_names = {
        k: k + ": " + v
        for k, v in extract_sic_code_description(load_sic_taxonomy(), "Class").items()
    }

    sic_4_names["9999"] = "9999: Not classified"
    sic_4_names["7499"] = "7499: Non-trading company"

    return section_name_lookup, division_section_names, sic_4_names


def clean_table_variables(table, variables, lookup):
    """Cleans variable values in a table"""

    t = table.copy()
    for v in variables:
        t[v] = t[v].map(lookup)
    return t


def get_topic_mdl():

    return pd.read_csv(
        f"{project_dir}/data/processed/topsbm_mdl.csv", dtype={"sector": str}
    ).rename(columns={"model": "mdl"})


def get_transitions(_list):

    same_code = _list[0] == _list[1]
    same_sic = _list[0][:4] == _list[1][:4]
    same_division = _list[0][:2] == _list[1][:2]

    return same_code, same_sic, same_division


def get_homogeneities():
    return pd.read_csv(f"{project_dir}/data/processed/topsbm_homog.csv")


def make_gt_network(net: nx.Graph) -> list:
    """Converts co-occurrence network to graph-tool netwotk"""
    nodes = {name: n for n, name in enumerate(net.nodes())}
    index_to_name = {v: k for k, v in nodes.items()}
    edges = list(net.edges(data=True))

    g_net = gt.Graph(directed=False)
    g_net.add_vertex(len(net.nodes))

    eprop = g_net.new_edge_property("int")
    g_net.edge_properties["weight"] = eprop

    for edg in edges:
        n1 = nodes[edg[0]]
        n2 = nodes[edg[1]]

        e = g_net.add_edge(g_net.vertex(n1), g_net.vertex(n2))
        g_net.ep["weight"][e] = edg[2]["weight"]

    return g_net, index_to_name


def get_community_names(partition, index_to_name, level=1):
    """Create node - community lookup"""

    b = partition.get_bs()

    b_lookup = {n: b[level][n] for n in sorted(set(b[0]))}

    names = {index_to_name[n]: int(b_lookup[c]) for n, c in enumerate(b[0])}

    return names


def make_partition(g_net, comms, distr_type):
    """Make community partition"""
    logging.info(distr_type)
    state = gt.minimize_nested_blockmodel_dl(
        g_net,
        B_min=comms,
        deg_corr=True,
        state_args=dict(recs=[g_net.ep.weight], rec_types=[distr_type]),
    )

    state = state.copy(bs=state.get_bs() + [np.zeros(1)] * 4, sampling=True)

    for i in range(100):
        ret = state.multiflip_mcmc_sweep(niter=10, beta=np.inf)

    return state


def make_sector_vocab(
    glass, sector, tokens="tokens_clean", drop=150, words=10, min_occ=100
):
    """Creates a sector vocabulary given a corpus"""

    dtm = make_doc_term_matrix(
        glass, sector=sector, tokens=tokens, min_occurrence=min_occ
    )
    drop_high_freq = dtm.sum().sort_values(ascending=False)[:drop].index.tolist()

    marketing = list(get_promo_terms(dtm))

    tfidf = make_tfidf_mat(dtm.drop(axis=1, labels=drop_high_freq + marketing))

    label_lu = tfidf.apply(
        lambda x: " ".join(x.sort_values(ascending=False).index[:words]), axis=1
    ).to_dict()
    return label_lu


def decompose_sector_chart(
    sector, sector_transitions_df, division_section_names, sector_names_merged
):
    """Decomnposes a sector into its sources"""

    target = (
        sector_transitions_df.loc[[sector in x for x in sector_transitions_df["last"]]]
        .groupby("last")["first"]
        .value_counts(normalize=False)
        .reset_index(name="n")
        .assign(
            section=lambda df: df["first"].apply(
                lambda x: division_section_names[x[:2]]
            )
        )
        .assign(last_name=lambda df: df["last"].map(sector_names_merged))
        .assign(first_name=lambda df: df["first"].map(sector_names_merged))
        .assign(last_short=lambda df: df["last"].apply(lambda x: x.split("_")[-1]))
    )

    composition = (
        alt.Chart(target)
        .mark_bar(stroke="darkgrey", strokeWidth=0.1)
        .encode(
            x=alt.X(
                "last_short",
                sort=alt.EncodingSortField("n", order="descending"),
                title="Text sector",
            ),
            y=alt.Y("n", title="Number of companies"),
            tooltip=["last", "last_name"],
            color=alt.Color(
                "section",
                title="SIC Section",
                legend=alt.Legend(orient="bottom", columns=6),
            ),
        )
    ).properties(width=1050, height=300)

    return composition, target


if __name__ == "__main__":

    logging.info("Setting up")
    driv = google_chrome_driver_setup()

    os.makedirs(f"{project_dir}/figures/cnei/png", exist_ok=True)
    os.makedirs(f"{project_dir}/figures/cnei/html", exist_ok=True)

    fig_path = f"{project_dir}/figures/cnei"

    logging.info("Reading data")
    company_label_lookup = get_company_sector_lookup()
    label_name_lookup = get_sector_name_lookup()

    section_name_lookup, division_section_names, sic_4_names = make_sic_lookups_extra()

    clean_variable_lookup = {}

    sector_labels = [
        (k.split("_")[0], int(k.split("_")[1])) for k in label_name_lookup.keys()
    ]

    sic_4_sectors = set([x[0] for x in sector_labels])

    logging.info("Check topic model outputs")

    sector_label_counts = (
        pd.DataFrame(
            [
                [v, len([x[1] for x in sector_labels if x[0] == v])]
                for v in sic_4_sectors
            ],
            columns=["sic4", "communities"],
        )
        .assign(
            section=lambda df: df["sic4"]
            .apply(lambda x: x[:2])
            .map(division_section_names)
        )
        .assign(nec=lambda df: ["9" in x[2:] for x in df["sic4"]])
    )
    sector_label_counts = sector_label_counts.loc[
        sector_label_counts["communities"] > 1
    ]

    mdl = get_topic_mdl()

    topmodel_output = (
        sector_label_counts.merge(mdl, left_on="sic4", right_on="sector")
        .assign(mdl_scaled=lambda df: zscore(df["mdl"]))
        .assign(communities_scaled=lambda df: zscore(df["communities"]))
        .assign(sector_name=lambda df: df["sic4"].map(sic_4_names))
    )

    topmodel_base = alt.Chart(topmodel_output).encode(
        y=alt.Y("sector", title="SIC4"),
        x=alt.X("mdl_scaled", title=["Minimum description", "length"]),
    )

    topmodel_output_chart_2 = (
        topmodel_base.mark_point(filled=True, stroke="black", strokeWidth=0.5).encode(
            color=alt.Color("section", title="SIC section"),
            size=alt.Size("communities", title="Number of text sectors"),
        )
    ).properties(height=700, width=500)

    topmodel_lines = topmodel_base.mark_line(stroke="black", strokeDash=[1, 1]).encode(
        detail="section"
    )

    top_model_results = altair_text_resize(topmodel_output_chart_2 + topmodel_lines)
    save_altair(top_model_results, "top_model_results", driver=driv, path=fig_path)

    logging.info("Analysis of sector reassignment")

    sector_reassignment_output = get_sector_reassignment_outputs()
    dist_vect = sector_reassignment_output["distance_container"]

    dist_summary = [(np.mean(x), np.std(x)) for x in dist_vect]

    dist_summary_df = pd.DataFrame(dist_summary, columns=["mean", "std"])
    dist_summary_df["low"], dist_summary_df["high"] = [
        dist_summary_df["mean"] + 1 * v * dist_summary_df["std"] for v in [-1, 1]
    ]
    dist_summary_df["iteration"] = range(len(dist_summary_df))

    # Plot evolution of distance to centroid
    dist_base = alt.Chart(dist_summary_df).encode(x="iteration")

    dist_line = dist_base.mark_line(point=True).encode(
        y=alt.Y(
            "mean",
            scale=alt.Scale(zero=False),
            title=["Mean distance", "to closest sector"],
        )
    )

    save_altair(dist_line, "mean_distance_sector", driver=driv, path=fig_path)

    # Stability in assignments
    # We will count how many times are companies assigned to
    # the same sector / SIC4 / division
    transitions = sector_reassignment_output["transition_container"]

    transition_comparison = [
        [get_transitions(el) for el in iteration.values()] for iteration in transitions
    ]

    transition_summary = [
        [np.mean([el[n] for el in org]) for org in transition_comparison]
        for n in [0, 1, 2]
    ]

    transition_df = pd.DataFrame(transition_summary).T
    transition_df.columns = ["same_community", "same_sic_code", "same_division"]
    transition_df["iteration"] = range(len(transition_df))

    transition_df_long = transition_df.melt(id_vars="iteration")

    clean_var_lookup = {
        "same_community": "Same text sector",
        "same_division": "Same SIC division",
        "same_sic_code": "Same SIC4 code",
    }
    transition_df_long = clean_table_variables(
        transition_df_long, variables=["variable"], lookup=clean_var_lookup
    )

    transition_chart = (
        alt.Chart(transition_df_long)
        .mark_line(point=True)
        .encode(
            x="iteration",
            y=alt.Y("value", title="% of reassignments", axis=alt.Axis(format="%")),
            color=alt.X("variable", title="Transition"),
        )
    ).properties(width=600, height=400)

    save_altair(
        altair_text_resize(transition_chart),
        "transition_shares",
        driver=driv,
        path=fig_path,
    )

    # Plot matrix of transition frequencies
    first_sector = [trans[0] for trans in transitions[0].values()]
    last_sector = [trans[1] for trans in transitions[-1].values()]

    transition_pairs_df = pd.DataFrame(
        [[x, y] for x, y in zip(first_sector, last_sector)],
        columns=["initial_sector", "final_sector"],
    )

    selected = choice(
        list(set(transition_pairs_df["initial_sector"])), 1600, replace=False
    )

    transition_freqs = (
        transition_pairs_df.groupby("initial_sector")["final_sector"]
        .value_counts(normalize=True)
        .reset_index(name="share")
    )

    transition_freqs_filt = transition_freqs.loc[
        (transition_freqs["initial_sector"].isin(selected))
        & (transition_freqs["final_sector"].isin(selected))
    ]

    transition_heatmat = (
        alt.Chart(transition_freqs_filt)
        .mark_rect()
        .encode(
            y=alt.Y(
                "initial_sector",
                axis=alt.Axis(labels=False, ticks=False, title="Initial sector"),
            ),
            x=alt.X(
                "final_sector",
                axis=alt.Axis(labels=False, ticks=False),
                title="Final sector",
            ),
            color=alt.Color(
                "share",
                scale=alt.Scale(type="log"),
                title="share",
                legend=alt.Legend(format="%"),
            ),
            tooltip=["initial_sector", "final_sector", "share"],
        )
    ).properties(width=400, height=400)

    save_altair(
        altair_text_resize(transition_heatmat),
        "transition_heatmap",
        driver=driv,
        path=fig_path,
    )

    # And at the SIC4 level

    transition_pairs_df["initial_sic4"], transition_pairs_df["final_sic4"] = [
        [x[:4] for x in df]
        for df in [
            transition_pairs_df["initial_sector"],
            transition_pairs_df["final_sector"],
        ]
    ]

    transition_sic_freqs = (
        transition_pairs_df.groupby("initial_sic4")["final_sic4"]
        .value_counts(normalize=True)
        .reset_index(name="share")
    )

    sic_name_descr = extract_sic_code_description(load_sic_taxonomy(), "Class")

    transition_sic_freqs["initial_name"], transition_sic_freqs["final_name"] = [
        transition_sic_freqs[var].map(sic_name_descr)
        for var in ["initial_sic4", "final_sic4"]
    ]

    transition_heatmat_sic = (
        alt.Chart(transition_sic_freqs)
        .mark_rect()
        .encode(
            x=alt.X("final_sic4", axis=alt.Axis(labels=False, ticks=False)),
            y=alt.Y("initial_sic4", axis=alt.Axis(labels=False, ticks=False)),
            color=alt.Color("share", scale=alt.Scale(type="log")),
            tooltip=["initial_name", "final_name", "share"],
        )
    ).properties(width=400, height=400)

    save_altair(
        transition_heatmat_sic, "transition_heatmap_sic", driver=driv, path=fig_path
    )

    # What do different sectors get reallocated to?
    transition_first_last = [
        get_transitions([f, l]) for f, l in zip(first_sector, last_sector)
    ]

    assessment_cols = ["same_community", "same_sic_code", "same_division"]

    transition_assessment_df = pd.concat(
        [
            transition_pairs_df,
            pd.DataFrame(
                transition_first_last,
                columns=["same_community", "same_sic_code", "same_division"],
            ),
        ],
        axis=1,
    ).assign(
        section=lambda df: df["initial_sector"].apply(
            lambda x: division_section_names[x[:2]]
        )
    )

    transition_shares_df = (
        transition_assessment_df.groupby(["initial_sector", "section"])[assessment_cols]
        .mean()
        .reset_index(drop=False)
        .melt(id_vars=["initial_sector", "section"])
        .reset_index(drop=False)
        .assign(comm_name=lambda df: df["initial_sector"].map(label_name_lookup))
    )

    reassignment_stat_chart = (
        alt.Chart(transition_shares_df)
        .mark_point(filled=True, opacity=0.5, stroke="black", strokeWidth=0.2)
        .encode(
            y=alt.Y("initial_sector", axis=alt.Axis(labels=False, ticks=False)),
            x=alt.X("value", axis=alt.Axis(format="%")),
            color="section",
            tooltip=["initial_sector", "comm_name", "value"],
            facet=alt.Facet("variable"),
        )
    ).properties(width=200, height=500)

    # Do less homogeneous communities lose / attract companies
    hom = get_homogeneities()
    dist_lookup = hom.set_index("cluster")["hom"].to_dict()

    tsector_freqs = pd.concat(
        [
            pd.Series(
                sector_reassignment_output["org_sector_container"][n]
            ).value_counts(normalize=True)
            for n in [0, -1]
        ],
        axis=1,
    )
    tsector_freqs.columns = ["freq_t0", "freq_t1"]
    tsector_freqs["ratio"] = tsector_freqs["freq_t1"] / tsector_freqs["freq_t0"]
    tsector_freqs["heterogeneity"] = tsector_freqs.index.map(dist_lookup)
    tsector_freqs = tsector_freqs.reset_index(drop=False)
    tsector_freqs["section"] = (
        tsector_freqs["index"].apply(lambda x: x[:2]).map(division_section_names)
    )
    tsector_freqs["name"] = tsector_freqs["index"].map(label_name_lookup)

    het_chart = (
        alt.Chart(tsector_freqs)
        .mark_point(filled=True, stroke="black", strokeWidth=0.15, opacity=0.7)
        .encode(
            x=alt.X("heterogeneity", title="initial heterogeneity"),
            color=alt.Color(
                "section", scale=alt.Scale(scheme="Spectral"), title="SIC section"
            ),
            tooltip=["name"],
            y=alt.Y("ratio", scale=alt.Scale(type="log"), title="Reassignment ratio"),
        )
    ).properties(height=400, width=600)

    save_altair(
        altair_text_resize(het_chart), "heterogeneity_ratio", driver=driv, path=fig_path
    )

    # %%
    save_altair(
        altair_text_resize(
            alt.vconcat(transition_chart, het_chart).resolve_scale(color="independent")
        ),
        "reassignment_combined",
        driver=driv,
        path=fig_path,
    )

    # Name / rename communities
    glass_tok = get_glass_tokenised()

    final_community = {k: v[1] for k, v in transitions[-1].items()}

    glass_tok["comm_assigned"] = glass_tok["org_id"].map(final_community)
    glass_tok_filtered = glass_tok.dropna(axis=0, subset=["comm_assigned"]).reset_index(
        drop=True
    )

    dtm = make_doc_term_matrix(
        glass_tok_filtered, sector="comm_assigned", tokens="tokens_clean"
    )

    drop_high_freq = dtm.sum().sort_values(ascending=False)[:150].index.tolist()

    tfidf = make_tfidf_mat(dtm.drop(axis=1, labels=drop_high_freq))
    sector_label2_lookup = tfidf.apply(
        lambda x: " ".join(x.sort_values(ascending=False).index[:10]), axis=1
    ).to_dict()

    logging.info("Make bottom up taxonomy")
    sectors_cooccs, sector_distances = (
        sector_reassignment_output["sector_cooccurrences"],
        sector_reassignment_output["sector_distances"],
    )

    all_sectors = list(set(chain(*sectors_cooccs)))

    # %%
    dist = {s: [] for s in all_sectors}

    # %%
    # %%time
    for n, sect in enumerate(all_sectors):
        if n % 100 == 0:
            logging.info(n)
        for s, d in zip(sectors_cooccs, sector_distances):
            for els, eld in zip(s, d):
                if els == sect:
                    dist[sect].append(eld)

    # %%
    dist_stats_dict = {k: (np.mean(v), np.std(v)) for k, v in dist.items()}

    dist_stats = pd.DataFrame(dist_stats_dict).T
    dist_stats.columns = ["mean", "std"]

    dist_stats["low"], dist_stats["high"] = [
        dist_stats["mean"] + 1 * v * dist_stats["std"] for v in [-1, 1]
    ]
    dist_stats = dist_stats.reset_index(drop=False).assign(
        section=lambda df: df["index"].apply(lambda x: division_section_names[x[:2]])
    )

    # %%
    dist_chart = (
        alt.Chart(dist_stats)
        .mark_point(filled=True, opacity=0.5, stroke="black", strokeWidth=0.2)
        .encode(
            y=alt.Y("index", axis=alt.Axis(labels=False, ticks=False)),
            x=alt.X("mean"),
            color="section",
        )
    ).properties(width=200, height=450)

    p = 0
    dist_thres = {k: v[0] - p * v[1] for k, v in dist_stats_dict.items()}

    sector_coocc_filtered = []
    for s, d in zip(sectors_cooccs, sector_distances):
        occurrences = []
        for els, eld in zip(s, d):
            if eld < dist_thres[els]:
                occurrences.append(els)
        sector_coocc_filtered.append(occurrences)

    sector_coocc_filtered_2 = [x for x in sector_coocc_filtered if len(x) > 0]

    net = make_network_from_coocc(
        sector_coocc_filtered_2, spanning=True, extra_links=500
    )
    g_net, index_name_lu = make_gt_network(net)

    p = make_partition(g_net, 400, "discrete-poisson")

    p.draw(
        output=f"{project_dir}/figures/industry_network.png",
        eorder=g_net.ep.weight
        # edge_pen_width=gt.prop_to_size(g_net.ep.weight, 0.5, 1, power=1)
    )

    bs = p.get_bs()

    chains = []
    chains_named = []

    for n, el in enumerate(bs[0]):
        links = [el]
        links_named = [index_name_lu[n]]
        for it in range(1, 4):
            new_el = [x for m, x in enumerate(bs[it]) if m == links[-1]]
            links.append(new_el[0])
            links_named.append(f"l{str(it)}_{str(new_el[0])}")

        chains.append(links)
        chains_named.append(links_named)
    level_dict = {l[0]: l[1:] for l in chains_named}

    for n in range(0, 3):
        level_dict_sub = {k: v[n] for k, v in level_dict.items()}
        glass_tok[f"sector_level_{str(n+1)}"] = glass_tok["comm_assigned"].map(
            level_dict_sub
        )

    sector_names = [
        make_sector_vocab(
            glass_tok,
            sector=sector_name,
            tokens="tokens_clean",
            drop=250,
            words=20,
            min_occ=min_occ,
        )
        for sector_name, min_occ in zip(
            ["comm_assigned", "sector_level_1", "sector_level_2", "sector_level_3"],
            [100, 150, 200, 300],
        )
    ]

    sector_names_merged = {
        k: k + ": " + v
        for k, v in {k: v for d in sector_names for k, v in d.items()}.items()
    }

    # Visualise sector network

    sector_col_names = [
        "comm_assigned",
        "sector_level_1",
        "sector_level_2",
        "sector_level_3",
    ]

    sector_array = np.array(glass_tok[sector_col_names].dropna())

    sector_pairs = list(
        chain(*[[[x[n], x[n + 1]] for x in sector_array] for n in range(3)])
    )

    sector_net = make_network_from_coocc(sector_pairs)

    edges_original = [(e[0], e[1], {"weight": 1}) for e in sector_net.edges()]
    final_layer = [
        (c[0], c[1], {"weight": 1})
        for c in combinations(list(set(sector_array[:, 3])), 2)
    ]

    sector_net_2 = nx.Graph(chain(*[edges_original, final_layer]))

    sect_pos = (
        pd.DataFrame(nx.nx_agraph.graphviz_layout(sector_net_2))
        .T.reset_index(drop=False)
        .rename(columns={0: "x", 1: "y", "index": "node"})
    )
    sector_freqs = (
        pd.concat(
            [
                glass_tok[s].value_counts().reset_index(name="sector")
                for s in sector_col_names
            ]
        )
        .set_index("index")["sector"]
        .to_dict()
    )

    sector_comms = {
        el: str(n)
        for n, comm in enumerate(algorithms.louvain(sector_net).communities)
        for el in comm
    }
    sector_net_df = (
        sect_pos.assign(node_size=lambda df: df["node"].map(sector_freqs))
        .assign(node_color=lambda df: df["node"].map(sector_comms))
        .assign(node_name=lambda df: df["node"].map(sector_names_merged))
    )

    # Finally - plot network

    sector_net_chart = plot_altair_network(
        sector_net_df,
        sector_net,
        show_neighbours=True,
        node_label="node_name",
        node_size="node_size",
        node_color="node_color",
        **{
            "node_size_title": "Number of companies",
            "node_color_title": "Community",
            "edge_weight_title": "Number of links",
            "title": "",
        },
    )

    taxonomy_tree = sector_net_chart.properties(height=600, width=500)

    save_altair(
        altair_text_resize(taxonomy_tree), "taxonomy_tree", driver=driv, path=fig_path
    )

    logging.info("Sector analysis")
    embs = get_firm_embeddings()

    sector_transitions_df = (
        pd.concat(
            [
                pd.DataFrame(
                    sector_reassignment_output["org_sector_container"][n].values(),
                    index=sector_reassignment_output["org_sector_container"][n].keys(),
                )
                for n in [0, -1]
            ],
            axis=1,
        )
    ).dropna()
    sector_transitions_df.columns = ["first", "last"]

    dec_7490 = decompose_sector_chart(
        "7490", sector_transitions_df, division_section_names, sector_names_merged
    )

    save_altair(
        altair_text_resize(dec_7490[0]), "7490_decomposed", driver=driv, path=fig_path
    )

    logging.info("Cross-sector analysis")

    descriptions = get_organisation_description()

    # %%
    env = [
        "environmental",
        "renewable",
        "solar",
        "sustainability",
        "energy",
        "emission",
        "sustainable",
        "green_energy",
    ]

    env_names = {
        k: v for k, v in sector_names_merged.items() if sum(e in v for e in env) > 2
    }

    env_firms = glass_tok.loc[glass_tok["comm_assigned"].isin(set(env_names.keys()))]

    env_composition_df = (
        env_firms.groupby("comm_assigned")["sic4"]
        .value_counts()
        .reset_index(name="n")
        .assign(
            section=lambda df: df["sic4"].apply(lambda x: division_section_names[x[:2]])
        )
        .assign(name=lambda df: df["comm_assigned"].map(sector_names_merged))
    )

    env_composition_ch = (
        alt.Chart(env_composition_df)
        .mark_bar(stroke="darkgrey", strokeWidth=0.1)
        .encode(
            y=alt.X("comm_assigned", title="Text sector"),
            x=alt.X("n", title="Number of companies"),
            tooltip=["comm_assigned", "name"],
            color=alt.Color(
                "section",
                title="SIC section",
                legend=alt.Legend(orient="bottom", columns=3),
            ),
        )
    ).properties(width=500, height=600)

    save_altair(
        altair_text_resize(env_composition_ch),
        "env_comp_chart",
        driver=driv,
        path=fig_path,
    )

    with open(
        f"{project_dir}/data/processed/glass_tokenised_processed.p", "wb"
    ) as outfile:
        pickle.dump(glass_tok, outfile)
