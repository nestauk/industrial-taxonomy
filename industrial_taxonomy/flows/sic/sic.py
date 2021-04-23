"""SIC 2007 taxonomy structure."""
from typing import Dict, List, Union

from metaflow import conda_base, FlowSpec, step

from utils import add_companies_house_extras, fill, normalise_codes, read


@conda_base(libraries={"pandas": "1.2.3", "xlrd": ">=1"})
class Sic2007Structure(FlowSpec):
    """SIC 2007 taxonomy structure.

    Includes extra subclasses used by companies house.

    Attributes:
        url: Source excel file
        data: Taxonomy structure as list of records.
        section: Lookup from code to name
        division: Lookup from code to name
        group: Lookup from code to name
        class_: Lookup from code to name
        subclass: Lookup from code to name
    """

    url: str
    data: List[Dict[str, Union[str, float]]]
    section: Dict[str, str]
    division: Dict[str, str]
    group: Dict[str, str]
    class_: Dict[str, str]
    subclass: Dict[str, str]

    @step
    def start(self):
        """Fetch data."""

        self.url = (
            "http://www.ons.gov.uk/file?uri=/methodology/classificationsandstandards/"
            "ukstandardindustrialclassificationofeconomicactivities/uksic2007/"
            "sic2007summaryofstructurtcm6.xls"
        )
        self._data = read(self.url)
        self.next(self.run)

    @step
    def run(self):
        """Clean data."""

        self._data = (
            self._data.pipe(fill).pipe(normalise_codes).pipe(add_companies_house_extras)
        )

        self.next(self.end)

    @step
    def end(self):
        """Multiple output formats."""

        # Lookups:
        for level in ["section", "division", "group", "class_", "subclass"]:
            setattr(
                self,
                level,
                self._data[[level, f"{level}_name"]]
                .set_index(level)
                .dropna()
                .to_dict()[f"{level}_name"],
            )

        self.data = self._data.to_dict(orient="records")
        del self._data


if __name__ == "__main__":
    Sic2007Structure()
