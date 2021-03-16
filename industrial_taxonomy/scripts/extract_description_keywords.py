import time
import logging
from itertools import chain
import pandas as pd
import industrial_taxonomy
from industrial_taxonomy import config
from industrial_taxonomy.getters.unspsc import get_unspsc
from industrial_taxonomy.queries.sector import get_glass_descriptions_SIC_sectors
from industrial_taxonomy.extraction.fit_rake import get_rake_phrases
from industrial_taxonomy.extraction.fit_yake import get_yake_phrases
from industrial_taxonomy.extraction.fit_keybert import get_keybert_phrases

project_dir = industrial_taxonomy.project_dir

# Get parametres
params = config["extraction"]

KB_PARAMS = params["keybert"]
SAMPLE_PARAMS = params["sample"]
INDUSTRY_PARAMS = params["industries"]

# We use this to label some tables below
METHOD_NAMES = ["rake", "yake", "keybert"]

# We use this to build a collection of UNSPSC terms
UNSPSC_CATS = ["segment_title", "family_title", "class_title", "commodity_title"]


def sample_companies(
    comps, code, category="sic4", sample_size=SAMPLE_PARAMS["sample_size"]
) -> list:
    """Samples company descriptions from a division
    Args:
        comps (pd.DataFrame): df with sectors and descriptions
        code (str): value for the industry code
        category (str): sector name
        div (str): division
        sample_size (n): number of companies to sample. Defaults to 5,000

    Returns a sampled list of descriptions
    """

    comp_sampled = (
        comps.loc[comps[category] == str(code)]
        .sample(n=sample_size)["description"]
        .reset_index(drop=True)
        .tolist()
    )
    return comp_sampled


def build_unspsc_terms():
    """Concatenate UNSPSC terms into a big word to filter keywords"""
    unspsc_words = " ".join(
        (
            [
                x.lower()
                for x in chain(
                    *[
                        unspsc[var].dropna().drop_duplicates().tolist()
                        for var in UNSPSC_CATS
                    ]
                )
            ]
        )
    )
    return unspsc_words


if __name__ == "__main__":
    comps = get_glass_descriptions_SIC_sectors()
    unspsc = get_unspsc()

    # We combine all UNSPCC into a big word
    unspsc_words = build_unspsc_terms()

    # We will loop over divisions and methods to extract keywords
    kws_container = []
    perf_container = []
    # For each division we sample descriptions and extract phrases
    for ind in INDUSTRY_PARAMS:
        category = ind["category"]
        code = ind["code"]

        logging.info(f"Extracting keywords for {category}: {code}")
        sample_descrs = sample_companies(comps, code=code, category=category)

        ind_kw_container = []
        ind_perf_container = []
        # For each method, extract phrases
        for n, method in enumerate(
            [get_rake_phrases, get_yake_phrases, get_keybert_phrases]
        ):
            t0 = time.time()

            kws = (
                method(sample_descrs)
                .reset_index(name="freqs")
                .rename(columns={"index": "kw"})
                .assign(method=METHOD_NAMES[n])
            )

            # Check if words are in UNSPSC, if so label them
            kws["in_unspsc"] = [x in unspsc_words for x in kws["kw"]]
            ind_kw_container.append(kws)

            t1 = time.time()
            perf = pd.Series(
                [t1 - t0, METHOD_NAMES[n], SAMPLE_PARAMS["sample_size"]],
                index=["time", "method", "sample_size"],
            )
            ind_perf_container.append(perf)

        kws_container.append(
            pd.concat(ind_kw_container)
            .assign(sect=code)
            .rename(columns={"sect": category})
        )

        perf_container.append(
            pd.DataFrame(ind_perf_container)
            .assign(sect=code)
            .rename(columns={"sect": category})
            .reset_index(drop=True)
        )

    # Save results

    codes = "_".join([str(ind["code"]) for ind in INDUSTRY_PARAMS])

    (
        pd.concat(kws_container)
        .reset_index(drop=True)
        .to_csv(
            f"{project_dir}/data/processed/description_kws_{codes}.csv", index=False
        )
    )

    (
        pd.concat(perf_container)
        .reset_index(drop=True)
        .to_csv(
            f"{project_dir}/data/processed/extraction_performance_{codes}.csv",
            index=False,
        )
    )
