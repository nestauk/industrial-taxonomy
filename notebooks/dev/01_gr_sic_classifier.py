# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     comment_magics: true
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.9.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
# %run ../notebook_preamble.ipy

# %%
from industrial_taxonomy.getters.glass_house import get_glass_house
from industrial_taxonomy.getters.companies_house import get_organisation, get_sector
from industrial_taxonomy.getters.glass import get_sector
from industrial_taxonomy.queries.sector import get_sector

# %%
from industrial_taxonomy.queries.sector import get_glass_SIC_sectors

# %%
glass_house = get_glass_house()

# %%
glass_house[glass_house['score'] > 60].shape

# %%
ch_orgs = get_organisation()

# %%
ch_sectors = get_sector()

# %%
ch_sectors['SIC4_code'] = ch_sectors['SIC5_code'].astype(str).str[:-1].astype(int)

# %%
