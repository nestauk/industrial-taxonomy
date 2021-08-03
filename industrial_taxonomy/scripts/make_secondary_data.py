# Make secondary data for complexity analysis

import io
import os
import pickle

import numpy as np
import pandas as pd
import requests


from industrial_taxonomy import project_dir
from industrial_taxonomy.scripts.complexity_analysis import (
    fetch_local_gdp,
    make_glass_geocoded,
    calc_eci,
)


def read_lad_nuts1_lookup(year=2019):
    """Read a lookup between local authorities and NUTS"""

    if year == 2019:
        lu_df = pd.read_csv(
            "https://opendata.arcgis.com/datasets/3ba3daf9278f47daba0f561889c3521a_0.csv"
        )
        return lu_df.set_index("LAD19CD")["RGN19NM"].to_dict()
    else:
        lu_df = pd.read_csv(
            "https://opendata.arcgis.com/datasets/054349b09c094df2a97f8ddbd169c7a7_0.csv"
        )
        return lu_df.set_index("LAD20CD")["RGN20NM"].to_dict()


lad_nuts_lookup = read_lad_nuts1_lookup()


def assign_nuts1_to_lad(c, lu=lad_nuts_lookup):
    """Assigns nuts1 to LAD"""

    if c in lu.keys():
        return lu[c]
    elif c[0] == "S":
        return "Scotland"
    elif c[0] == "W":
        return "Wales"
    elif c[0] == "N":
        return "Northern Ireland"
    else:
        return np.nan


def fetch_pop():
    # def make_bres():
    url = "https://www.ons.gov.uk/file?uri=/peoplepopulationandcommunity/populationandmigration/populationestimates/datasets/populationestimatesforukenglandandwalesscotlandandnorthernireland/mid2020/ukpopestimatesmid2020on2021geography.xls"

    req = requests.get(url)

    with io.BytesIO(req.content) as s:

        pop = pd.io.excel.read_excel(s, sheet_name=6, skiprows=7)[
            ["Code", "All ages"]
        ].rename(columns={"Code": "code", "All ages": "pop_2020"})
    return pop


def get_glass_tok_processed():
    with open(
        f"{project_dir}/data/processed/glass_tokenised_processed.p", "rb"
    ) as infile:
        return pickle.load(infile)


def make_official():

    population, gdp_ph = fetch_local_gdp()
    gdp_ph["growth"] = gdp_ph["2019"] / gdp_ph["2010"]
    gdp_ph["mean_growth"] = pd.DataFrame(
        [(gdp_ph[str(x)] / gdp_ph[str(x - 1)]) - 1 for x in range(2010, 2019)]
    ).mean(axis=0)

    pop = fetch_pop()

    # pop_filtered = (population[['LA code', '2019']]
    #               .rename(columns={'2019':'2019_pop'}))

    sec_df = pd.read_csv(f"{project_dir}/data/processed/lad_secondary.csv")
    know_variables = ["Annual pay - gross", "% with NVQ4+ - aged 16-64"]
    know_rename = {l: s for l, s in zip(know_variables, ["gross_pay", "higher_ed_pc"])}

    sec_df_wide = (
        sec_df.pivot_table(index="geography_code", columns="variable", values="value")[
            know_variables
        ]
        .rename(columns=know_rename)
        .reset_index(drop=False)
    )

    all_official = (
        gdp_ph[["LA code", "LA name", "2019", "growth", "mean_growth"]]
        .rename(columns={"2019": "gdp_ph_2019"})
        .merge(pop, left_on="LA code", right_on="code")[
            [
                "LA code",
                "LA name",
                "gdp_ph_2019",
                "growth",
                "mean_growth",
                "pop_2020",
            ]
        ]
        .merge(sec_df_wide, left_on="LA code", right_on="geography_code")
        .rename(columns={"LA code": "lad_code", "LA name": "lad_name"})
    )

    return all_official


def make_regression_data():

    if os.path.exists(f"{project_dir}/data/processed/region_dataset.csv") is False:
        off_df = make_official()
        glass_tok = get_glass_tok_processed()
        glass_geo = make_glass_geocoded(glass_tok)
        lad_complexities = pd.concat(
            [
                calc_eci(
                    glass_geo.pivot_table(
                        index=["lad_code"], columns=sector, aggfunc="size"
                    ).fillna(0)
                )
                for sector in ["sic4", "comm_assigned"]
            ],
            axis=1,
        )
        lad_complexities.columns = ["compl_sic4", "compl_new"]

        # %%
        lad_merged = off_df.merge(
            lad_complexities.reset_index(drop=False), on="lad_code"
        ).set_index(["lad_code", "lad_name"])

        lad_merged["nuts1"] = [
            assign_nuts1_to_lad(x) for x in lad_merged.index.get_level_values(0)
        ]

        lad_merged.to_csv(f"{project_dir}/data/processed/region_dataset.csv")
        return lad_merged
    else:
        return pd.read_csv(f"{project_dir}/data/processed/region_dataset.csv")
