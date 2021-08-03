# Complexity regression

import logging
import warnings
from itertools import product

import altair as alt
import arviz as az
import numpy as np
import pandas as pd
from bambi import Model
from scipy.stats import zscore

from industrial_taxonomy import project_dir
from industrial_taxonomy.scripts.make_secondary_data import make_regression_data
from industrial_taxonomy.utils.altair_save_utils import (
    google_chrome_driver_setup,
    save_altair,
    altair_text_resize,
)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import filelock


def clean_table_variables(table, variables, lookup):

    t = table.copy()
    for v in variables:
        t[v] = t[v].map(lookup)
    return t


if __name__ == "__main__":

    logging.info("Reading regression data")
    fig_path = f"{project_dir}/figures/cnei"

    driv = google_chrome_driver_setup()
    lad_df = make_regression_data().dropna().drop(axis=1, labels=["geography_code"])

    lad_df["pop_log"] = np.log(lad_df["pop_2020"])

    corr_mat = (
        lad_df.drop(axis=1, labels=["lad_code", "lad_name", "nuts1", "pop_2020"])
        .corr()
        .reset_index(drop=False)
        .melt(id_vars=["index"])
    )

    corr_base = alt.Chart(corr_mat).encode(x="index", y="variable")

    corr_color = corr_base.mark_rect().encode(color="value")

    corr_text = corr_base.mark_text().encode(
        text=alt.Text("value", format=".2"),
        color=alt.condition(
            "datum.value>0.5", alt.ColorValue("white"), alt.ColorValue("black")
        ),
    )

    (corr_color + corr_text).properties(width=400, height=300)

    combs = product(
        ["compl_sic4", "compl_new"],
        ["gdp_ph_2019", "growth", "higher_ed_pc", "gross_pay"],
    )

    logging.info("Fitting models")
    model_results = []

    for item in combs:
        lad_df_ = lad_df.copy()
        lad_df_[item[1]] = np.log(lad_df_[item[1]])

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = Model(lad_df_)
            f = m.fit(f"{item[1]} ~ {item[0]} + nuts1 + pop_log")

        model_results.append(f)

    clean_var_names = {
        "gdp_ph_2019": "GDP per capita",
        "gross_pay": "Annual gross pay",
        "growth": "GDP growth",
        "higher_ed_pc": "Share tertiary",
        "compl_new": "ECI text",
        "compl_sic4": "ECI SIC4",
    }

    summary_tables = []

    for n, item in enumerate(
        product(
            ["compl_sic4", "compl_new"],
            ["gdp_ph_2019", "growth", "higher_ed_pc", "gross_pay"],
        )
    ):

        summ = (
            az.summary(model_results[n])
            .assign(dep=item[1])
            .loc[item[0], ["dep", "hdi_3%", "hdi_97%", "mean"]]
        )
        summary_tables.append(summ)
    results_df = pd.DataFrame(summary_tables).reset_index(drop=False)

    results_df = clean_table_variables(results_df, ["index", "dep"], clean_var_names)

    # %%
    base = alt.Chart().encode(y=alt.Y("index", title=None))
    point_ch = (
        base.mark_point(filled=True, size=40, stroke="black", strokeWidth=0.5)
        .encode(
            x=alt.X("mean", title="HDI 3%-97%"), color=alt.Color("index", legend=None)
        )
        .properties(height=100, width=250)
    )
    point_err = base.mark_errorbar(color="black").encode(
        x=alt.X("hdi_3%", title="HDI 3%-97%"), x2="hdi_97%"
    )

    lay = alt.layer(point_ch, point_err, data=results_df).facet(
        row=alt.Row(
            "dep",
            title="Dependent variable",
            sort=["GDP per capita", "GDP growth", "Annual gross pay", "Share tertiary"],
        )
    )
    save_altair(
        altair_text_resize(lay), "modelling_results", driver=driv, path=fig_path
    )

    # %%
    comp = []
    dep_vars = ["gdp_ph_2019", "growth", "higher_ed_pc", "gross_pay"]
    for n, m in zip(range(0, 4), range(4, 8)):
        comparisons = az.compare(
            {"compl_sic4": model_results[n], "compl_new": model_results[m]}
        ).assign(dep=dep_vars[n])
        comp.append(comparisons)

    # %%
    pd.concat(comp)
