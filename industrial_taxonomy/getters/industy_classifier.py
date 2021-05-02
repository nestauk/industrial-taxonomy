import logging

from industrial_taxonomy.flows.classifier.classifier_train import TextClassifier
from industrial_taxonomy import config


logger = logging.getLogger(__name__)

def sic4_predictor(run_id=None):
    """Generates an object for performaing 4-digit SIC code classifications"""
    return IndustryPredictor('sic4_classifier', run_id)

def new_sector_predictor(run_id=None):
    """Generates an object for performaing 4-digit SIC code classifications"""
    return IndustryPredictor('new_sector_classifier', run_id)



class IndustryPredictor:
    """Loads model from the Metaflow datastore based on a task"""

    def __init__(self, task_id, run_id=None):
        self.task_id = task_id
        self.run_id = self._set_run_id(run_id)
        logger.info('Loading saved model...')
        self.model = TextClassifier._load_trained_model(self.run_id)

    def _set_run_id(self, run_id):
        if run_id is None:
            return config["flows"][f"{self.task_id}"]["run_id"]
        else:
            return run_id

    def predict(self, dataset):
        """Predicts the industry of a series of texts.
        
        Arguments:
            dataset (`list`): A dataset to perform predictions upon. The data 
                should be a list of dictionaries with fields `index`, `text` 
                and `label`. For example:
                   [{'id': 0, 'text': 'meow', 'label': 'cat'}.
                    {'id': 1, 'text': 'woof', 'label': 'dog'}.
                    {'id': 2, 'text': 'grrrr', 'label': 'dog'}.
                    {'id': 3, 'text': 'meeewwwooww', 'label': 'cat'}.
                    ...
                    ] 
                The index field should be an integer. If the dataset has no 
                labels, then all `label` fields should be set to `None`.

        Returns:
            pred_labels (`list`): The predicted class labels.
            pred_probs (`list`): The prediction probabilities of every class 
                for each sample.
            doc_ids (`list`): The document IDs. Documents are re-ordered during
                the prediction process.

        """
        logger.info(f'Predicting labels for {len(dataset)} samples')
        pred_labels, pred_probs, doc_ids, lookup = TextClassifier.predict(
                dataset, self.run_id, self.model)
        return pred_labels, pred_probs, doc_ids, lookup
