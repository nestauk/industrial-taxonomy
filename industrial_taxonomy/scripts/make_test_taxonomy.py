# Script to filter test data and build prototype keyworc networks

import logging
import json
import pandas as pd
import networkx as nx
from nltk.stem import PorterStemmer
from communities.algorithms import louvain_method

# Read companies
from industrial_taxonomy.scripts.extract_description_keywords import sample_companies
from industrial_taxonomy.queries.sector import get_glass_descriptions_SIC_sectors
from industrial_taxonomy.utils.altair_save_utils import (
    google_chrome_driver_setup,
    save_altair,
)


from industrial_taxonomy.taxonomy import (
    filter_keywords,
    get_extraction_report,
    plot_word_frequencies,
    make_network_from_coocc,
    get_adjacency,
    label_comms,
    plot_extraction_performance,
)


import industrial_taxonomy

project_dir = industrial_taxonomy.project_dir


FIG_PATH = f"{project_dir}/figures"
NET_PARAMS = industrial_taxonomy.config["taxonomy_test"]
KW_PATH = NET_PARAMS["test_file"]
with open(f"{project_dir}/data/aux/stop_words.txt", "r") as infile:
    STOP_WORDS = [x.strip() for x in infile.readlines()]
DRIVER = google_chrome_driver_setup()
STEMMER = PorterStemmer()


def get_kw_freqs():
    return pd.read_csv(
        f"{project_dir}/data/processed/{KW_PATH}.csv", dtype={"sic4": str}
    )


def sample_descriptions(comps: pd.DataFrame, sic: str) -> list:
    """Samples company descriptions in a SIC code
    Args:
        comps: company table
        sic: sic code
    """
    return sample_companies(comps, sic)


def make_kws_report():
    """Filters keywords and saves some frequency chart"""
    kws_freq = get_kw_freqs()
    kws_filtered = filter_keywords(kws_freq, to_remove=STOP_WORDS)[0]
    ch_kws = plot_word_frequencies(
        get_extraction_report(kws_filtered, top_words=10)
    ).properties(title="KW frequencies")
    save_altair(ch_kws, "kw_freqs", driver=DRIVER)

    kws_filtered_unspsc = filter_keywords(kws_filtered, unspsc_filter=False)[0]
    ch_kws_unspsc = plot_word_frequencies(
        get_extraction_report(kws_filtered_unspsc, top_words=10)
    ).properties(title="KW frequencies: UNSPSC Filter")
    save_altair(ch_kws_unspsc, "kw_freqs_unspsc", driver=DRIVER)

    # We save KW descriptive statistics
    stats = kws_freq.groupby(["sic4", "method"]).describe()
    stats.columns = stats.columns.droplevel()
    (
        stats.reset_index(drop=False).to_markdown(
            f"{project_dir}/data/processed/kw_freq_stats.md", index=False
        )
    )

    # We return the filtered KWS without UNSPSC filter
    return kws_filtered


def kw_in_descr(desc: str, kw: str) -> bool:
    """True if a kw is in a description"""
    return ' '+kw+' ' in desc


def get_edges(description: str, terms: list) -> list:
    """Returns co-occurring keywords
    Args:
        description: company description
        terms: keywords
    """

    return [
        x.strip()
        for x in list(filter(lambda kw: kw_in_descr(description.lower(), kw), terms))
    ]


def get_sector_inputs(comps: pd.DataFrame, kws_df: pd.DataFrame, sector: str):
    """Returns company descriptions and keywords in a sector
    Args:
        comps: company df
        kws: list of keyword freqencies
        sector: industry to select
    """

    selected_comps = sample_descriptions(comps, sector)
    selected_kws = set(kws_df.query(f"sic4=='{sector}'")["kw"])

    return selected_comps, selected_kws


def make_comm_table(comms: list, name: str, length=300) -> pd.DataFrame:
    """Takes community outputs and turns them into a markdown table
    Args:
        comms: list of communities
        name: name of the sector
        length: max length of community terms
    """
    table = [" ".join(x)[:length] + "..." for x in comms]

    df = pd.DataFrame(
        {
            "community": ["**" + str(n) + "**" for n in range(len(table))],
            "keywords": table,
        }
    )
    return df


def make_sector_communities(
    comps: pd.DataFrame, kws_df: pd.DataFrame, sector: str
) -> nx.Graph:
    """Builds a kw co-occurrence network & decomposes it into communities of kws
    Args:
        decrs: company descriptions
        kws_df: selected keywords
        sector: industry
    """
    logging.info(f"Extracting communities for sector {sector}")
    logging.info("Making company inputs")
    descrs, kws = get_sector_inputs(comps, kws_df, sector)

    logging.info("Processing keywords")
    kws_stemmed = {kw if " " in kw else STEMMER.stem(kw) for kw in kws}
    # Pad short words to avoid spurious extraction
    kws_padded = {" " + kw + " " if " " not in kw else kw for kw in kws_stemmed}

    logging.info("Making network")
    edge_list = list(map(lambda d: get_edges(d, kws_padded), descrs))
    net = make_network_from_coocc(edge_list, thres=0.05, spanning=True)

    logging.info("Making communities")
    comms = louvain_method(get_adjacency(net))
    comms_labelled = label_comms([list(c) for c in comms[0]], net)
    return comms_labelled


if __name__ == "__main__":

    save_altair(plot_extraction_performance(), "extr_perf", DRIVER)

    comps = get_glass_descriptions_SIC_sectors()
    filtered_kw = make_kws_report()

    for sic in set(filtered_kw["sic4"]):
        comm = make_sector_communities(comps, filtered_kw, sector=sic)

        with open(
            f"{project_dir}/data/processed/communities_{sic}.json", "w"
        ) as outfile:
            json.dump(comm, outfile)

        # Save the table
        comm_table = make_comm_table(comm, sic)
        comm_table.to_markdown(
            f"{project_dir}/data/processed/communities_table_{sic}.md", index=False
        )
