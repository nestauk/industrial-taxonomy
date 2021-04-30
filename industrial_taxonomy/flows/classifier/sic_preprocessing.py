"""Collect and split matched Companies House and Glass data.
"""

from metaflow import FlowSpec, step, Parameter, JSONType, S3, Run
from sklearn.model_selection import train_test_split

from industrial_taxonomy.flows.classifier.classifier_utils import create_org_data


class SicPreprocess(FlowSpec):
    sic_level = Parameter(
            "sic_level",
            help="SIC Level to use",
            type=int,
            )
    match_threshold = Parameter(
            "match_threshold",
            help="Glass + CH fuzzy matching score threshold",
            type=int
            )
    config = Parameter(
            "config",
            help="Params for functions",
            type=JSONType
            )
    test = Parameter(
            "test",
            help="Run in test mode (only 500 samples)",
            type=bool,
            default=False
            )

    @step
    def start(self):
        self.next(self.load_match_glass_ch)

    @step
    def load_match_glass_ch(self):
        org_data = create_org_data(self.match_threshold, self.sic_level)
        if self.test:
            org_data = org_data[:500]
        self.org_data = org_data
        self.next(self.split)

    @step
    def split(self):
        train_test_config = config['train_test_split']
        self.train_set, self.test_set = train_test_split(
                self.org_data,
                **train_test_config
                )
        self.next(self.end)

    @step
    def end(self):
        pass

if __name__ == "__main__":
    SicPreprocess()
