"""
Helper script to identify participants.json files in data/ that contain Neurobagel annotations
and whether those datasets were previously annotated internally.
"""

import json
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).parents[1]
DATA_DIR = ROOT_DIR / "data"
RESOURCES_DIR = ROOT_DIR / "resources"
# Overview file with info about which datasets have already been annotated by us
OPENNEURO_TABULAR_DATASETS_OVERVIEW = (
    RESOURCES_DIR / "openneuro_tabular_all_datasets.tsv"
)

OUTPUT_PATH = RESOURCES_DIR / "participants_json_files_with_annotations.txt"


def count_json_files_containing_annotations(
    dir_path: Path, overview_tsv_path: Path, output_path: Path
):
    overview = pd.read_csv(overview_tsv_path, sep="\t")
    overview_indexed = overview.set_index("dataset")["annotated"]

    json_files = list(dir_path.glob("*.json"))

    unparsable_jsons = []
    jsons_with_annotations = []
    for json_file in json_files:
        dataset = json_file.stem
        try:
            with open(json_file, "r", encoding="utf-8-sig") as f:
                contents = json.load(f)
        except Exception as e:
            print(f"ERROR: Error loading JSON file {json_file}: {e}")
            unparsable_jsons.append(dataset)
            continue
        if not isinstance(contents, dict):
            print(
                f"WARNING: {json_file} does not contain a dictionary at the top level, skipping."
            )
            unparsable_jsons.append(dataset)
            continue
        for column_contents in contents.values():
            if isinstance(column_contents, dict) and column_contents.get("Annotations"):
                jsons_with_annotations.append(
                    {
                        "dataset_id": dataset,
                        "already_annotated_by_us": overview_indexed.loc[dataset],
                    }
                )
                break

    if unparsable_jsons:
        print(
            f"Unparsable JSON files ({len(unparsable_jsons)}/{len(json_files)}): {', '.join(unparsable_jsons)}"
        )
    print(
        f"Number of OpenNeuroDatasets participants.json files in {dir_path} containing 'Annotations': {len(jsons_with_annotations)}/{len(json_files)}"
    )

    pd.DataFrame(jsons_with_annotations).to_csv(output_path, index=False, sep="\t")
    print(f"List of dataset IDs with annotations saved to {output_path}")


if __name__ == "__main__":
    count_json_files_containing_annotations(
        DATA_DIR, OPENNEURO_TABULAR_DATASETS_OVERVIEW, OUTPUT_PATH
    )
