# Functions to produce final report. TO BE REFACTORED
import altair as alt
import numpy as np
import re
import pandas as pd
import logging
import networkx as nx
import matplotlib.pyplot as plt
from random import sample, seed
from cdlib import algorithms
from itertools import chain, combinations
from wordcloud import WordCloud
from industrial_taxonomy.taxonomy.taxonomy_filtering import (
    make_glass_for_taxonomy,
    get_glass_tokenised,
    drop_ners,
    make_doc_term_matrix,
    extract_salient_terms,
    filter_salient_terms,
    get_promo_terms,
)
from industrial_taxonomy.getters.glass import get_organisation_description
from industrial_taxonomy.scripts.extract_communities import get_summaries
from industrial_taxonomy.scripts.tag_companies import get_lookup
from industrial_taxonomy.taxonomy.post_processing_functions import (
    make_co_occurrence_table,
    find_duplicated_communities,
    make_distance_matrix,
)
from industrial_taxonomy.taxonomy.taxonomy_community import make_network_from_coocc
from industrial_taxonomy.getters.processing import get_table
import industrial_taxonomy
from industrial_taxonomy.utils.sic_utils import (
    load_sic_taxonomy,
    extract_sic_code_description,
    section_code_lookup,
)
from industrial_taxonomy.utils.altair_save_utils import (
    google_chrome_driver_setup,
    save_altair,
)
from industrial_taxonomy.altair_network import plot_altair_network

alt.data_transformers.disable_max_rows()
project_dir = industrial_taxonomy.project_dir

REP_PATH = f"{project_dir}/reports/tables_figures"
driv = google_chrome_driver_setup()


def save_table_md(tab, name):
    tab.to_markdown(REP_PATH + f"/{name}.md", index=False, floatfmt=".3f")


def label_decile(vector):

    out = []

    for el in vector:
        for n, r in enumerate(np.arange(0, 1.1, 0.1)):
            if el <= r:
                out.append(n)
                break
    return out


def jaccard_similarity(list1, list2):
    intersection = len(list(set(list1).intersection(list2)))
    union = (len(list1) + len(list2)) - intersection
    return float(intersection) / union


def rename_merged(name, comm_names):

    cs = [comm_names[x].split(", ") for x in name.split(" ")[1].split("__")]
    cs = list(set(chain(*cs)))
    return ", ".join(cs)


def make_token_network(
    sector, glass_sector, token_comms, comm_names, coh_lookup, top_comms=20, sample=500
):
    logging.info(f"Making network {sector}")
    sector_comms = {k: v for k, v in token_comms.items() if sector in k}
    comms_affiliations = {}
    logging.info("Making lookup")
    for k, v in sector_comms.items():
        for tok in list(v):
            comms_affiliations[tok] = k

    sector_tokens = glass_sector.query(f"sic4=='{sector}'").sample(sample)[
        "token_filtered"
    ]

    logging.info("Extracting network")
    sector_network = nx.maximum_spanning_tree(
        make_network_from_coocc(sector_tokens.tolist(), extra_links=0)
    )

    logging.info("Making layout")
    pos = nx.kamada_kawai_layout(sector_network)
    pos_df = (
        pd.DataFrame(pos)
        .T.reset_index(drop=False)
        .rename(columns={0: "x", 1: "y", "index": "node"})
    )
    degree_distr = dict(sector_network.degree)

    node_df = (
        pos_df.assign(node_size=lambda df: df["node"].map(degree_distr))
        .assign(node_name=lambda df: df["node"])
        .assign(
            node_color=lambda df: df["node"].map(comms_affiliations).map(comm_names)
        )
    )
    top_20_comms = node_df["node_color"].value_counts().index[:top_comms]

    node_df["node_color"] = [
        x if x in top_20_comms else "Other" for x in node_df["node_color"]
    ]
    node_df["node_opacity"] = node_df["node"].map(comms_affiliations).map(coh_lookup)

    logging.info("Plotting")
    alt_net = plot_altair_network(
        node_df,
        sector_network,
        show_neighbours=True,
        node_label="node_name",
        node_size="node_size",
        node_color="node_color",
        node_opacity="node_opacity",
        **{
            "edge_weight_title": "co-occurrences",
            "title": f"{sector}: Term network",
            "node_opacity_title": "community coherence",
            "node_size_title": "connections",
            "node_color_title": "communities",
        },
    ).properties(width=600, height=400)

    return alt_net


if __name__ == "__main__":
    g = make_glass_for_taxonomy().assign(
        tokens_clean=lambda df: df["tokens"].apply(drop_ners)
    )

    sic4 = extract_sic_code_description(load_sic_taxonomy(), "Class")

    # Term filtering
    dtm = make_doc_term_matrix(g)
    sal = extract_salient_terms(dtm)

    not_sal = filter_salient_terms(sal, thres=0.6)
    sal_2 = {k: set([w for w in v if w not in not_sal]) for k, v in sal.items()}

    # Salient example table
    salient_example_container = []

    for s in ["0123", "6201", "7490", "9999"]:

        out = {
            "   SIC4   ": f"    {s}    ",
            "Filter 1": ", ".join(sample(list(sal[s]), 20)),
            "Filter 2": ", ".join(sample(list(sal_2[s]), 20)),
        }
        salient_example_container.append(out)

    salient_example_table = pd.DataFrame(salient_example_container)
    save_table_md(salient_example_table, "salient_example")

    promo_terms = get_promo_terms(dtm, thres=0.5)

    promo_term_counts = [
        [x for x in tok if x in promo_terms] for tok in g["tokens_clean"]
    ]

    promo_term_freqs = pd.Series(chain(*promo_term_counts)).value_counts().to_dict()

    wc = WordCloud(
        colormap="Reds",
        background_color="white",
        height=250,
        width=400,
        font_path="/Users/jmateosgarcia/Library/Fonts/FiraSans-Regular.ttf",
    )

    my_wc = wc.generate_from_frequencies(promo_term_freqs)

    fig, ax = plt.subplots(figsize=(10, 10))

    ax.imshow(my_wc)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(f"{REP_PATH}/wordcloud.png")

    #
    div_names = extract_sic_code_description(load_sic_taxonomy(), "Division")

    # Promo terms chart
    g2 = g.copy()
    g2["promo_terms"] = [len(x) for x in promo_term_counts]

    sector_promo_counts = (
        g2.assign(div=lambda df: [x[:2] for x in df["sic4"]])
        .groupby("div")["promo_terms"]
        .mean()
        .reset_index(drop=False)
    )

    high_pol_sectors = sector_promo_counts.sort_values("promo_terms", ascending=False)[
        :5
    ]["div"].tolist()

    sector_promo_counts["div_2"] = [
        x if x in high_pol_sectors else " " for x in sector_promo_counts["div"]
    ]

    promo_chart = (
        alt.Chart(sector_promo_counts)
        .mark_bar()
        .encode(
            x=alt.X("div", axis=alt.Axis(labels=False, ticks=False), title="Division"),
            y=alt.Y("promo_terms", title=["Mean n promotional", "terms"]),
            tooltip=["div"],
            color=alt.Color(
                "promo_terms",
                legend=None,
                scale=alt.Scale(scheme="Spectral"),
                sort="descending",
            ),
        )
    ).properties(width=500, height=150)

    text = (
        alt.Chart(sector_promo_counts)
        .mark_text(xOffset=4, yOffset=-5)
        .encode(
            x=alt.X("div", axis=alt.Axis(labels=False, ticks=False), title="Division"),
            y=alt.Y("promo_terms", title=["Mean n promotional", "terms"]),
            text=alt.Text("div_2:N"),
        )
    )

    out = promo_chart + text

    save_altair(out, "promotional_terms", driver=driv, path=REP_PATH)

    # Token length chart
    # Token length

    other_sics = [
        k
        for k, v in extract_sic_code_description(load_sic_taxonomy(), "Class").items()
        if any(name in v for name in ["n.e.c", "Other"])
    ]
    g["token_filtered"] = [
        [x for x in row["tokens_clean"] if x in sal_2[row["sic4"]]]
        for _id, row in g.iterrows()
    ]
    g["Initial"], g["Filtered"] = [
        [len(x) for x in g[var]] for var in ["tokens_clean", "token_filtered"]
    ]

    sect_lu = section_code_lookup()
    sect_name = {
        k.strip(): v
        for k, v in extract_sic_code_description(load_sic_taxonomy(), "SECTION").items()
    }

    sect_size = g["sic4"].value_counts()

    g_tokens = (
        g[["sic4", "Initial", "Filtered"]]
        .melt(id_vars=["sic4"])
        .reset_index(drop=False)
        .groupby(["sic4", "variable"])["value"]
        .mean()
        .reset_index(name="mean_terms")
        .assign(
            sect=lambda df: [
                sect_lu[x[:2]] + ": " + sect_name[sect_lu[x[:2]]] for x in df["sic4"]
            ]
        )
        .assign(
            is_other=lambda df: [True if x in other_sics else False for x in df["sic4"]]
        )
        .assign(size=lambda df: df["sic4"].map(sect_size))
    )

    tok_ch = (
        alt.Chart(g_tokens)
        .mark_point(opacity=0.5, filled=True, stroke="black", strokeWidth=0.1)
        .encode(
            y=alt.Y("sect", title="Sector"),
            x=alt.X(
                "mean_terms", scale=alt.Scale(type="log"), title="Mean number of terms"
            ),
            size=alt.X("size", title="organisations in sector"),
            shape=alt.Shape(
                "is_other",
                scale=alt.Scale(range=["circle", "cross"]),
                title="NEC category",
            ),
            color=alt.Color("variable", title="Term count variable"),
            tooltip=["sic4"],
        )
    )
    save_altair(tok_ch, "term_frequencies", driv, REP_PATH)

    reduction_effect = (
        g[["sic4", "Initial", "Filtered"]]
        .reset_index(drop=False)
        .assign(change=lambda df: df["Filtered"] / df["Initial"])
        .assign(
            is_other=lambda df: [True if x in other_sics else False for x in df["sic4"]]
        )
    )

    logging.info(f"reduction_effect={np.mean(reduction_effect['change'])}")

    # Community extraction

    # Summary chart

    sic_names = extract_sic_code_description(load_sic_taxonomy(), "Class")
    sic_names["9999"] = "Non-classifiable establishments"
    sic_names["7499"] = "Non trading company"

    sic_names_2 = {k: k + ": " + v for k, v in sic_names.items()}

    summary_table = get_summaries("partition_summaries").assign(
        sic_label=lambda df: [x + ": " + sic_names[x] for x in df["sic4"]]
    )

    method_comparison = (
        alt.Chart(
            summary_table.drop(axis=1, labels=["instance"]).query(
                "method!='SBM_nested'"
            )
        )
        .mark_point(filled=True, stroke="black", strokeWidth=0.3, shape="circle")
        .encode(
            y=alt.Y(
                "sic_label",
                sort=alt.EncodingSortField("score", op="max", order="descending"),
                title=None,
            ),
            x=alt.X("score", title="Modularity score", scale=alt.Scale(zero=False)),
            size=alt.Size("comm_n", title=["Number of", "communities"]),
            color=alt.Color(
                "method",
                scale=alt.Scale(scheme="accent"),
                title="Algorithm",
                sort=alt.EncodingSortField("score", op="mean", order="descending"),
            ),
        )
    )

    method_chart = (
        method_comparison.configure_legend(columns=1)
        .properties(height=1200, width=800)
        .configure_axis(labelFontSize=12)
    )

    save_altair(method_chart, "modularity_comparison", path=REP_PATH, driver=driv)

    # Similarity chart

    similarity_df = (
        get_summaries("partition_similarities")
        .assign(sic_label=lambda df: df["sic4"].map(sic_names_2))
        .assign(is_other=lambda df: [x in other_sics for x in df["sic4"]])
    )
    sort_sectors = (
        similarity_df.groupby("sic_label")["score"]
        .median()
        .sort_values(ascending=False)
        .index.tolist()
    )

    similarity_box = (
        alt.Chart(similarity_df)
        .mark_boxplot(size=10)
        .encode(
            y=alt.Y("sic_label", sort=sort_sectors, title=None),
            color=alt.X("is_other", title="Other and N.E.C. sector"),
            x=alt.X(
                "score",
                scale=alt.Scale(zero=False),
                title="Normalised mutual information",
            ),
        )
    )
    similarity_boxplot = (
        similarity_box.properties(height=350, width=200)
        .properties(height=1200, width=800)
        .configure_axis(labelFontSize=12)
    )

    save_altair(similarity_boxplot, "partition_similarity", driver=driv, path=REP_PATH)

    # Postprocessing
    # Community length
    comm_terms = get_lookup("sector_communities")

    size_distr = pd.DataFrame(
        [[k.split("_")[0], len(v)] for k, v in comm_terms.items()],
        columns=["sic4", "terms"],
    )

    term_hist = (
        alt.Chart(size_distr)
        .mark_line(point=True)
        .encode(
            x=alt.X("terms", bin=True, title="Number of terms"),
            tooltip=["sic4"],
            y=alt.Y("count(terms)", title="Count of communities"),
            color=alt.Color(
                "sic4",
                sort=alt.EncodingSortField("count(terms)", order="descending"),
                scale=alt.Scale(scheme="Oranges"),
                legend=None,
            ),
        )
    ).properties(width=500, height=200)

    save_altair(term_hist, "term_histogram", driver=driv, path=REP_PATH)

    # Coherence
    labelled = get_table("glass_comm_labelled")

    labelled_no_unl = labelled.query("label!='unlabelled'").reset_index(drop=True)

    coh = (
        get_table("title_homog")
        .assign(
            category=lambda df: [
                "high" if x > 0.75 else "low" if x < 0.5 else "medium"
                for x in df["median_similarity"]
            ]
        )
        .dropna()
    )

    coh_table = []

    for cat in ["high", "medium", "low"]:
        res = coh.query(f"category == '{cat}'")
        rand = (
            res.sample(10)
            .reset_index(drop=True)
            .sort_values("median_similarity", ascending=False)
        )
        rand["salient_terms"] = [
            re.sub("_", "\\_", x[:75] + "...") for x in rand["salient_terms"]
        ]
        rand["category"] = cat
        coh_table.append(rand)

    coh_assessment = (
        pd.concat(coh_table)
        .reset_index(drop=True)
        .reset_index(drop=False)
        .rename(
            columns={
                "level_0": "n",
                "index": "Community",
                "salient_terms": "Salient terms",
                "category": "Ranking",
                "median_similarity": "Median Similarity",
            }
        )
    )
    save_table_md(coh_assessment, "coherence_examples")

    # Coherence figure
    coh["decimal"] = label_decile(coh["median_similarity"])
    coh["sic4"] = [x.split("_")[0] for x in coh["index"]]

    coherence_shares = (
        coh.query("sic4!='unlabelled'")
        .groupby("sic4")["decimal"]
        .value_counts(normalize=True)
        .reset_index(name="share")
        .assign(sic_label=lambda df: [sic_names_2[x] for x in df["sic4"]])
    )

    sectors_sorted = (
        coherence_shares.query("decimal>7")
        .groupby("sic_label")["share"]
        .sum()
        .sort_values(ascending=False)
        .index.tolist()
    )

    coherence_ch = (
        alt.Chart(coherence_shares)
        .mark_bar()
        .encode(
            y=alt.Y("sic_label", sort=sectors_sorted),
            x=alt.X(
                "share",
                scale=alt.Scale(domain=[0, 1]),
                title="Share of activity",
                axis=alt.Axis(format="%"),
            ),
            color=alt.Color(
                "decimal",
                title="Coherence ranking",
                sort="ascending",
                scale=alt.Scale(scheme="Spectral"),
            ),
        )
        .properties(height=1200, width=800)
        .configure_axis(labelFontSize=12)
    )
    save_altair(coherence_ch, "coherence_table", driver=driv, path=REP_PATH)

    # Visualise networks
    sectors = ["8299", "7490", "6201", "4334"]
    tokens_to_comms = get_lookup("sector_communities")
    comm_names = get_lookup("comm_terms_lookup")
    g_tok = get_glass_tokenised()

    coh_lookup = coh.set_index("index")["median_similarity"].to_dict()

    example_networks = [
        make_token_network(
            s, g_tok, tokens_to_comms, comm_names, coh_lookup=coh_lookup, sample=1000
        )
        for s in sectors
    ]

    net_1 = alt.vconcat(example_networks[0], example_networks[1]).resolve_scale(
        color="independent"
    )

    save_altair(net_1, "network_1_examples", driver=driv, path=REP_PATH)

    net_2 = alt.vconcat(example_networks[2], example_networks[3]).resolve_scale(
        color="independent", opacity="shared"
    )

    save_altair(net_2, "network_2_examples", driver=driv, path=REP_PATH)

    # Deduplication

    merged_comm = get_lookup("merged_communities")
    seed(5)
    dupe_comms = []

    for v in merged_comm.values():

        comms = v.split(" ")[1].split("__")
        dupe_comms.append(
            {"Community 1": comm_names[comms[0]], "Community 2": comm_names[comms[1]]}
        )

    dupe_df = pd.DataFrame(dupe_comms)
    save_table_md(dupe_df.sample(5), "duplicated_examples")

    # Expanded examples table
    desc = get_organisation_description()

    merged_comm_names = get_lookup("merged_comm_names")
    comm_names_merged_2 = {
        k: v if "merged" not in k else rename_merged(k, comm_names)
        for k, v in merged_comm_names.items()
    }

    labelled_v2 = get_table("unlabelled_closest_sector_V2")

    labelled_v2 = labelled_v2.assign(
        description=lambda df: df["org_id"].map(
            desc.set_index("org_id")["description"].to_dict()
        )
    ).assign(comm_terms=lambda df: df["sector"].map(comm_names_merged_2))
    expanded_examples = (
        labelled_v2.query("similarity>0.9")
        .dropna()
        .sample(20)
        .assign(comm_short=lambda df: [x[:100] + "..." for x in df["comm_terms"]])
        .assign(descr_short=lambda df: [x[:150] + "..." for x in df["description"]])[
            ["sector", "comm_terms", "descr_short"]
        ]
    ).rename(
        columns={
            "sector": "community",
            "comm_terms": "community name",
            "descr_short": "description",
        }
    )

    save_table_md(expanded_examples, "expanded_examples")

    energy_terms = ["energy", "renewable", "environmental"]

    high_coh = coh.loc[coh["median_similarity"] > 0.5]["index"].tolist()

    energy_comms = pd.DataFrame(
        [
            [k, v[:150] + ".."]
            for k, v in comm_names.items()
            if any(en in v for en in energy_terms) & (k in high_coh)
        ],
        columns=["community name", "salient_terms"],
    )

    save_table_md(energy_comms, "energy_examples")

    # Conclusion
    comm_coocc = get_lookup("co_occurrence")
    co_occ_table = make_co_occurrence_table(comm_coocc)

    sim_df = make_distance_matrix(co_occ_table.T, similarity=True, return_df=True)
    merged_comms, merged_comms_names = find_duplicated_communities(
        sim_df[high_coh], comm_names
    )

    # Removed duplicates based on names
    comm_pairs = list(combinations([x for x in comm_names.keys() if x in high_coh], 2))

    salient_overlaps = []

    for c in comm_pairs:

        jacc = jaccard_similarity(
            comm_names[c[0]].split(" "), comm_names[c[1]].split(" ")
        )
        salient_overlaps.append([c[0], c[1], jacc])

    salient_df = pd.DataFrame(
        salient_overlaps, columns=["comm1", "comm2", "jacc"]
    ).sort_values("jacc", ascending=False)

    salient_df["comm1_names"], salient_df["comm2_names"] = [
        salient_df[x].map(comm_names) for x in ["comm1", "comm2"]
    ]

    merged_df = salient_df.loc[salient_df["jacc"] > 0.25]
    merged_df["merged_name"] = [
        r["comm1"] + "__" + r["comm2"] for _, r in merged_df.iterrows()
    ]
    merged_df["merged_terms"] = [
        sample([r["comm1_names"], r["comm2_names"]], 1)[0]
        for _, r in merged_df.iterrows()
    ]

    merged_to_lookup = merged_df[
        ["merged_name", "merged_terms", "comm1", "comm2"]
    ].melt(id_vars=["merged_name", "merged_terms"])
    merged_name_lu = merged_to_lookup.set_index("value")["merged_name"].to_dict()
    merged_terms_lu = merged_to_lookup.set_index("value")["merged_terms"].to_dict()

    med_coh = coh.loc[coh["median_similarity"] > 0.3]["index"].tolist()
    co_occ = [x for x in comm_coocc.values() if len(x) > 1]
    co_occ = [[x for x in el if x in med_coh] for el in co_occ]
    co_occ = [x for x in co_occ if len(x) > 1]
    co_occ = [
        [x if x not in merged_name_lu.keys() else merged_name_lu[x] for x in el]
        for el in co_occ
    ]

    comm_names_updated = {}

    for k, v in comm_names.items():
        if k not in merged_terms_lu.keys():
            comm_names_updated[k] = v
        else:
            comm_names_updated[merged_name_lu[k]] = merged_terms_lu[k]

    # Create taxonomy network
    net_1 = make_network_from_coocc(co_occ, extra_links=50)

    comps = list(nx.connected_components(net_1))

    comms = algorithms.chinesewhispers(net_1)

    comms_lookup = {
        v: f"Community {n}" for n, el in enumerate(comms.communities) for v in el
    }

    net = nx.subgraph(net_1, comps[0])

    pos = nx.spring_layout(net)

    pos_df = (
        pd.DataFrame(pos)
        .T.reset_index(drop=False)
        .rename(columns={0: "x", 1: "y", "index": "node"})
    )

    degree_distr = dict(net.degree)
    sect_name_2 = {k: k + ": " + v for k, v in sect_name.items()}

    comm_names_lu = {comm: sect_name_2[sect_lu[comm[:2]]] for comm in list(net.nodes())}

    node_df = (
        pos_df.assign(node_size=lambda df: df["node"].map(degree_distr))
        .assign(node_name=lambda df: df["node"])
        .assign(node_color=lambda df: df["node"].map(comms_lookup))
        .assign(node_name=lambda df: df["node"].map(comm_names_updated))
    )

    ind_plot = plot_altair_network(
        node_df,
        net,
        show_neighbours=True,
        node_label="node_name",
        node_size="node_size",
        node_color="node_color",
        node_opacity="node_opacity",
        edge_opacity=0.03,
        **{
            "edge_weight_title": "co-occurrences",
            "title": "Sector network",
            "node_opacity_title": "community coherence",
            "node_size_title": "connections",
            "node_color_title": "Industry communities",
        },
    ).properties(width=700, height=400)

    save_altair(ind_plot, "industry_taxon", driver=driv, path=REP_PATH)

    # Taxonomy share chart
    sampled_comms = sample(set(comms_lookup.values()), 30)
    sampled_comms_n = [int(x.split(" ")[1]) for x in sampled_comms]

    taxon_comms_df = (
        pd.Series(comms_lookup)
        .reset_index(name="comm_name")
        .assign(sic4=lambda df: df["index"].apply(lambda x: x[:3]))
        .assign(sect=lambda df: [sect_name_2[sect_lu[x[:2]]] for x in df["sic4"]])
    )
    taxons_shares = (
        taxon_comms_df.groupby("comm_name")["sect"]
        .value_counts(normalize=True)
        .reset_index(name="share_community")
    )
    taxons_shares["sort"] = taxons_shares["comm_name"].apply(
        lambda x: int(x.split(" ")[1])
    )
    taxons_shares = taxons_shares.sort_values("sort", ascending=True).reset_index(
        drop=False
    )
    taxons_shares = taxons_shares.loc[taxons_shares["comm_name"].isin(sampled_comms)]

    sorted_communities = taxons_shares["comm_name"].tolist()

    top_sectors_comm = (
        pd.Series(degree_distr)
        .reset_index(name="degree")
        .assign(comm=lambda df: df["index"].map(comms_lookup))
        .assign(comm_name=lambda df: df["index"].map(comm_names_updated))
    )

    community_table = (
        top_sectors_comm.groupby("comm")
        .apply(
            lambda df: re.sub(
                "_",
                "\\_",
                "___".join(df.sort_values("degree", ascending=False)["comm_name"][:3]),
            )
        )
        .reset_index(name="top_names")
        .assign(comm=lambda df: df["comm"].apply(lambda x: int(x.split(" ")[1])))
        .sort_values("comm", ascending=True)
        .rename(columns={"comm": "community", "top_names": "High degree sectors"})
    )

    save_table_md(
        community_table.loc[community_table["community"].isin(sampled_comms_n)],
        "comm_examples",
    )
    taxon_bar = (
        alt.Chart(taxons_shares)
        .mark_bar()
        .encode(
            y=alt.Y("comm_name", title="Community", sort=sorted_communities),
            x=alt.X(
                "share_community",
                title="Share",
                axis=alt.Axis(format="%"),
                sort="descending",
            ),
            color=alt.Color(
                "sect", scale=alt.Scale(scheme="tableau20"), title="Section"
            ),
        )
    )
    save_altair(taxon_bar, "community_composition", driver=driv, path=REP_PATH)
