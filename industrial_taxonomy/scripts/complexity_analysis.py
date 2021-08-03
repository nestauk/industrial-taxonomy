# Scripts to fetch / process data for complexity analysis
import io
import os
import requests
from zipfile import ZipFile
import logging
import numpy as np
import pandas as pd
from scipy.linalg import eig
from industrial_taxonomy.getters.companies_house import get_organisation
from industrial_taxonomy.getters.glass import get_address
from industrial_taxonomy.getters.glass_house import get_glass_house


import industrial_taxonomy

project_dir = industrial_taxonomy.project_dir

_APS_URL = "https://www.nomisweb.co.uk/api/v01/dataset/NM_17_5.data.csv?geography=1811939329...1811939332,1811939334...1811939336,1811939338...1811939497,1811939499...1811939501,1811939503,1811939505...1811939507,1811939509...1811939517,1811939519,1811939520,1811939524...1811939570,1811939575...1811939599,1811939601...1811939628,1811939630...1811939634,1811939636...1811939647,1811939649,1811939655...1811939664,1811939667...1811939680,1811939682,1811939683,1811939685,1811939687...1811939704,1811939707,1811939708,1811939710,1811939712...1811939717,1811939719,1811939720,1811939722...1811939730&date=2019-12&variable=18,45,290,335,344&measures=20599,21001,21002,21003"
_ASHE_URL = "https://www.nomisweb.co.uk/api/v01/dataset/NM_30_1.data.csv?geography=1811939329...1811939332,1811939334...1811939336,1811939338...1811939497,1811939499...1811939501,1811939503,1811939505...1811939507,1811939509...1811939517,1811939519,1811939520,1811939524...1811939570,1811939575...1811939599,1811939601...1811939628,1811939630...1811939634,1811939636...1811939647,1811939649,1811939655...1811939664,1811939667...1811939680,1811939682,1811939683,1811939685,1811939687...1811939704,1811939707,1811939708,1811939710,1811939712...1811939717,1811939719,1811939720,1811939722...1811939730&date=latest&sex=8&item=2&pay=7&measures=20100,20701"
_SIMD_URL = "https://www.gov.scot/binaries/content/documents/govscot/publications/statistics/2020/01/scottish-index-of-multiple-deprivation-2020-ranks-and-domain-ranks/documents/scottish-index-of-multiple-deprivation-2020-ranks-and-domain-ranks/scottish-index-of-multiple-deprivation-2020-ranks-and-domain-ranks/govscot%3Adocument/SIMD%2B2020v2%2B-%2Branks.xlsx"
_DZ_LU = "http://statistics.gov.scot/downloads/file?id=2a2be2f0-bf5f-4e53-9726-7ef16fa893b7%2FDatazone2011lookup.csv"
_SECOND = f"{project_dir}/data/processed/lad_secondary.csv"

logger = logging.getLogger(__name__)
NSPL_PATH = f"{project_dir}/data/raw/nspl"


def fetch_nomis(url):
    """Fetch Nomis data
    Args:
        url (str): nomis url
    """
    return pd.read_csv(url)


def process_nomis(
    df, indicator_name, value_column, source, indicator_column="MEASURES_NAME"
):
    """Fetch nomis data
    Args:
        df (df): nomis table
        indicator_name (str): name of indicator
        value_column (str): value column
        source (str): data source
        indicator_column (str): column that contains the indicator
    Returns:
        A clean table with secondary data
    """
    return (
        df.query(f"{indicator_column}=='{indicator_name}'")[
            ["DATE", "GEOGRAPHY_NAME", "GEOGRAPHY_CODE", value_column, "OBS_VALUE"]
        ]
        .reset_index(drop=True)
        .rename(columns={"OBS_VALUE": "VALUE", value_column: "VARIABLE"})
        .assign(source=source)
        .rename(columns=str.lower)
    )


def read_secondary():
    """REad secondary data"""
    return pd.read_csv(_SECOND)


def fetch_local_gdp(sheets=[7, 8]):

    r = requests.get(
        "https://www.ons.gov.uk/file?uri=/economy/grossdomesticproductgdp/datasets/regionalgrossdomesticproductlocalauthorities/1998to2019/regionalgrossdomesticproductlocalauthorities.xlsx"
    )

    out = []

    for s in sheets:
        with io.BytesIO(r.content) as fh:
            ons_table = (
                pd.io.excel.read_excel(fh, sheet_name=s, skiprows=1)
                .dropna()
                .rename(columns={"2019\n[note 3]": "2019"})
            )
            ons_table.columns = [str(x) for x in ons_table.columns]

            out.append(ons_table)

    return out


def fetch_nspl():
    """Fetch NSPL if needed"""

    if os.path.exists(NSPL_PATH) is False:
        logging.info("Fetching NSPL")
        nspl_req = requests.get(
            "https://www.arcgis.com/sharing/rest/content/items/7606baba633d4bbca3f2510ab78acf61/data"
        )
        zipf = ZipFile(io.BytesIO(nspl_req.content))
        zipf.extractall(path=NSPL_PATH)
    else:
        logging.info("Already fetched nspl")


def make_nspl_to_merge():
    """Merge NSPL postcodes with TTWA names"""

    nspl = pd.read_csv(
        f"{NSPL_PATH}/Data/NSPL_FEB_2021_UK.csv", usecols=["pcds", "laua"]
    )
    laua = pd.read_csv(
        f"{NSPL_PATH}/Documents/LA_UA names and codes UK as at 04_20.csv"
    )
    nspl = (
        nspl.merge(laua, left_on="laua", right_on="LAD20CD")
        .drop(axis=1, labels=["LAD20CD", "LAD20NMW"])
        .rename(columns={"laua": "lad_code", "LAD20NM": "lad_name"})
    )

    return nspl


def create_lq(X, threshold=1, binary=False):
    """Calculate the location quotient.

    Divides the share of activity in a location by the share of activity in
    the UK total.

    Args:
        X (pandas.DataFrame): Rows are locations, columns are sectors,
        threshold (float, optional): Binarisation threshold.
        binary (bool, optional): If True, binarise matrix at `threshold`.
            and values are activity in a given sector at a location.

    Returns:
        pandas.DataFrame

    #UTILS
    """

    Xm = X.values
    with np.errstate(invalid="ignore"):  # Accounted for divide by zero
        X = pd.DataFrame(
            (Xm * Xm.sum()) / (Xm.sum(1)[:, np.newaxis] * Xm.sum(0)),
            index=X.index,
            columns=X.columns,
        ).fillna(0)

    return (X > threshold).astype(float) if binary else X


def calc_eci(X, sign_correction=None):
    """Calculate the original economic complexity index (ECI).

    Args:
        X (pandas.DataFrame): Rows are locations, columns are sectors,
            and values are activity in a given sector at a location.
        sign_correction (pd.Series, optional): Array to correlate with ECI
            to calculate sign correction. Typically, ubiquity. If None, uses
            the sum over columns of the input data.

    Returns:
        pandas.DataFrame

    #UTILS
    """

    X = _drop_zero_rows_cols(X)

    C = np.diag(1 / X.sum(1))  # Diagonal entries k_C
    P = np.diag(1 / X.sum(0))  # Diagonal entries k_P
    H = C @ X.values @ P @ X.T.values
    w, v = eig(H, left=False, right=True)

    eci = pd.DataFrame(v[:, 1].real, index=X.index, columns=["eci"])

    # Positively correlate `sign_correction` (some proxy for diversity) w/ ECI
    if sign_correction is None:
        sign_correction = X.sum(1)
    else:
        sign_correction = sign_correction.loc[X.index]
    sign = np.sign(np.corrcoef(sign_correction, eci.eci.values)[0, 1])
    logger.info(f"CI sign: {sign}")

    return (eci - eci.mean()) / eci.std() * sign


def _drop_zero_rows_cols(X):
    """Drop regions/entities with no activity

    Fully zero column/row means ECI cannot be calculated
    """

    nz_rows = X.sum(1) > 0
    has_zero_rows = nz_rows.sum() != X.shape[0]
    if has_zero_rows:
        logger.warning(f"Dropping all zero rows: {X.loc[~nz_rows].index.values}")
        X = X.loc[nz_rows]
    nz_cols = X.sum(0) > 0
    has_zero_cols = nz_cols.sum() != X.shape[1]
    if has_zero_cols:
        logger.warning(f"Dropping all zero cols: {X.loc[:, ~nz_cols].columns.values}")
        X = X.loc[:, nz_cols]

    return X

def fetch_local_gdp(sheets=[7,8]):
    
    r = requests.get('https://www.ons.gov.uk/file?uri=/economy/grossdomesticproductgdp/datasets/regionalgrossdomesticproductlocalauthorities/1998to2019/regionalgrossdomesticproductlocalauthorities.xlsx')
    
    out = []
    
    for s in sheets:    
        with io.BytesIO(r.content) as fh:
            ons_table = (pd.io.excel.read_excel(fh, sheet_name=s,skiprows=1).dropna()
                         .rename(columns={'2019\n[note 3]':'2019'}))
            ons_table.columns = [str(x) for x in ons_table.columns]
            
            out.append(ons_table)
    
    return out

def make_glass_geo_lookups():
    logging.info("Fetching and processing nspl")
    glass_add = get_address()

    lad_postcode = make_nspl_to_merge()

    logging.info("merging with nspl")

    glass_add = (glass_add   
           .query("rank==1")
           .drop_duplicates('org_id')
           [['org_id','postcode']]
           .merge(lad_postcode,left_on='postcode',right_on='pcds')
                )
    logging.info("Saving lookups")
    lookups = [glass_add.set_index('org_id')[var].to_dict() for var in ['lad_code', 'lad_name']]

    return lookups

def make_glass_geocoded(glass):

    lookups = make_glass_geo_lookups()

    glass['lad_code'] = glass['org_id'].map(lookups[0])
    glass_filtered = (glass
                      .dropna(axis=0, subset=['lad_code', 'comm_assigned'])
                      .reset_index(drop=True)
                      )
    return glass_filtered    

if __name__ == "__main__":

    logging.info("Fetching secondary data")
    ashe = process_nomis(fetch_nomis(_ASHE_URL), "Value", "PAY_NAME", "ashe")
    aps = process_nomis(fetch_nomis(_APS_URL), "Variable", "VARIABLE_NAME", "aps")

    pd.concat([ashe, aps], axis=0).to_csv(_SECOND, index=False)
