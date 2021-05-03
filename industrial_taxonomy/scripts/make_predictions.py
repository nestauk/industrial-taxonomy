from industrial_taxonomy.getters.industy_classifier import new_sector_predictor
import industrial_taxonomy
import pickle
import json
import logging

project_dir = industrial_taxonomy.project_dir
print(project_dir)

def get_new_sector(category="test"):
    with open(f"{project_dir}/data/raw/new_sector_test.json", "r") as infile:
        data = json.load(infile)
    return data


def save_predictions(predictions):
    with open(f"{project_dir}/data/processed/predicted.p", "wb") as outfile:
        pickle.dump(predictions, outfile)


if __name__=='__main'__:

    predictor = new_sector_predictor()

    data = get_new_sector()
    preds = predictor.predict(data)

    save_predictions(preds)
