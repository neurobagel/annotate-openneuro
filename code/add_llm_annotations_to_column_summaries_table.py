import json
import logging
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    datefmt="[%Y-%m-%d %H:%M:%S]",
)
logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).parents[1]
DATA_DIR = ROOT_DIR / "data"
RESOURCES_DIR = ROOT_DIR / "resources"
INPUT_DIR = DATA_DIR / "llm_assessment_annotations"
OUTPUT_PATH = (
    RESOURCES_DIR
    / "participants_tsv_columns_summary_first_guess_manual_pass_with_assessments.tsv"
)
COLUMN_SUMMARIES_PATH = (
    RESOURCES_DIR / "participants_tsv_columns_summary_first_guess_manual_pass.tsv"
)

ASSESSMENT_VOCAB_URL = "https://raw.githubusercontent.com/neurobagel/communities/refs/heads/main/configs/Neurobagel/assessment.json"


def fetch_file_from_url(url: str) -> dict:
    response = requests.get(url)
    response.raise_for_status()
    return response.json()


ASSESSMENT_TERMS = fetch_file_from_url(ASSESSMENT_VOCAB_URL)[0]["terms"]


def get_term_for_id(term_id: str) -> str:
    """
    Given a term ID (e.g., "snomed:12345"), return the corresponding term name from the ASSESSMENT_TERMS list,
    or "not found" if the term ID is not in the list.
    """
    for term in ASSESSMENT_TERMS:
        if term["id"] == term_id.removeprefix("snomed:"):
            return term["name"]
    return "not found"


def main():
    column_summaries = pd.read_csv(
        COLUMN_SUMMARIES_PATH, sep="\t", dtype=str, keep_default_na=False
    )

    datasets_with_assessments_annotated = []
    input_files = list(INPUT_DIR.glob("*.json"))
    for llm_annotations_file in tqdm(
        input_files, desc="Processing LLM annotations for datasets"
    ):
        ds_id = llm_annotations_file.stem.split("_")[0]
        with open(llm_annotations_file, "r", encoding="utf-8") as f:
            llm_annotations = json.load(f)

        for column, annotation in llm_annotations.items():
            if annotation.get("variable_id") == "nb:Assessment":
                row_mask = (column_summaries["dataset"] == ds_id) & (
                    column_summaries["column"] == column
                )
                column_summaries.loc[row_mask, "llm_classification"] = "nb:Assessment"
                column_summaries.loc[row_mask, "llm_snomed_term"] = annotation[
                    "assessment_id"
                ].removeprefix("snomed:")
                column_summaries.loc[row_mask, "llm_label"] = annotation[
                    "assessment_label"
                ]
                column_summaries.loc[row_mask, "snomed_label"] = get_term_for_id(
                    annotation["assessment_id"]
                )
                column_summaries.loc[row_mask, "llm_confidence"] = annotation[
                    "confidence"
                ]

        if (
            column_summaries.loc[
                column_summaries["dataset"] == ds_id, "llm_classification"
            ]
            .eq("nb:Assessment")
            .any()
        ):
            datasets_with_assessments_annotated.append(ds_id)

    logger.info(f"Datasets annotated by LLM: {len(input_files)}")
    logger.info(
        f"Datasets with assessments annotated ({len(datasets_with_assessments_annotated)}/{len(input_files)}): "
        f"{datasets_with_assessments_annotated}"
    )

    column_summaries.to_csv(OUTPUT_PATH, sep="\t", index=False)


if __name__ == "__main__":
    main()
