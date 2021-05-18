import pandas as pd
import requests


def subclass_fill(df):
    """Resolve."""

    # Forward fill classes
    df.loc[:, ("class_", "class__name")] = df.loc[:, ("class_", "class__name")].ffill()

    # Backfill subclass by one
    # (each bfill eventually yields a duplicate row which we can drop with
    # `drop_duplicates` rather than tedious index accounting)
    df.loc[:, ("subclass", "subclass_name")] = df.loc[
        :, ("subclass", "subclass_name")
    ].bfill(limit=1)

    # If no subclass, derive from class by adding a zero, and using class name
    idx = df["subclass"].isna()
    df.loc[idx] = df.loc[idx].assign(
        subclass=lambda x: x["class_"] + "/0", subclass_name=lambda x: x.class__name
    )

    return df.drop_duplicates()  # Drops dups we induced with bfill


def generic_fill(df, col):
    """Ffill columns relating to `col`, dropping rows that were originally not NaN."""
    cols = [col, f"{col}_name"]
    subdf = (
        df[cols].copy()
        # Drop rows that only have either code or name - they correspond to notes
        .loc[lambda x: x[cols].isna().sum(1) != 1]
    )
    label_idx = subdf[cols].notna().sum(1).astype(bool)
    return subdf.ffill().loc[~label_idx].join(df.drop(cols, 1))


def fill(df):
    """Fill missing information."""
    return (
        df.pipe(generic_fill, "section")
        .pipe(generic_fill, "division")
        .pipe(generic_fill, "group")
        .pipe(subclass_fill)
    )


def add_companies_house_extras(df):
    """Add Companies House specific SIC codes."""
    extras = {
        "subclass": ["74990", "98000", "99999"],
        "subclass_name": [
            "Non-trading company",
            "Residents property management",
            "Dormant company",
        ],
    }

    return df.append(pd.DataFrame(extras))


def normalise_codes(df):
    """Remove dots and slashes from SIC digits."""
    cols = ["group", "division", "section", "class_", "subclass"]
    df.loc[:, cols] = df.loc[:, cols].apply(
        lambda col: col.str.replace("[./]", "", regex=True)
    )
    return df


def read(url):
    """Read SIC 2007 structure from ONS hosted excel file."""

    response = requests.get(url)
    response.raise_for_status()

    return (
        pd.read_excel(
            response.content,
            skiprows=1,
            names=[
                "section",
                "section_name",
                "division",
                "division_name",
                "group",
                "group_name",
                "class_",
                "class__name",
                "subclass",
                "subclass_name",
            ],
            dtype=str,
        )
        .dropna(how="all")
        .apply(lambda column: column.str.strip(), axis=1)
    )
