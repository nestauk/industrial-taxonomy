"""Economic geography utilities."""
import numpy as np
import pandas as pd


def location_quotient(
    X: pd.DataFrame, threshold: float = 1, binary: bool = False
) -> pd.DataFrame:
    """Calculate the location quotient.

    Divides the share of activity in a location by the share of activity in
    the population total.

    Args:
        X: Rows are locations, columns are sectors, and values are activity in
           a given sector at a location.
        threshold: Binarisation threshold.
        binary: If True, binarise matrix at `threshold`.

    Returns:
        pandas.DataFrame
    """

    Xm = X.values
    with np.errstate(invalid="ignore"):  # Accounted for divide by zero
        X = pd.DataFrame(
            (Xm * Xm.sum()) / (Xm.sum(1)[:, np.newaxis] * Xm.sum(0)),
            index=X.index,
            columns=X.columns,
        ).fillna(0)

    return (X > threshold).astype(float) if binary else X
