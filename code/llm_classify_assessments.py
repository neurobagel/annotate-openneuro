import json
import logging
import os
from pathlib import Path

import pandas as pd
import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    datefmt="[%Y-%m-%d %H:%M:%S]",
)
logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).parents[1]
DATA_DIR = ROOT_DIR / "data"
RESOURCES_DIR = ROOT_DIR / "resources"
OUT_DIR = DATA_DIR / "llm_assessment_annotations"

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_REQUEST_HEADERS = {
    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
    "Content-Type": "application/json",
}
MODEL = "gpt-4o-mini"

ASSESSMENT_VOCAB_URL = "https://raw.githubusercontent.com/neurobagel/communities/refs/heads/main/configs/Neurobagel/assessment.json"


def save_json(data: dict, path: Path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def fetch_file_from_url(url: str) -> dict:
    response = requests.get(url)
    response.raise_for_status()
    return response.json()


def send_prompt_to_llm(prompt: str) -> dict:
    data = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
    }

    response = requests.post(
        OPENROUTER_API_URL, headers=OPENROUTER_REQUEST_HEADERS, json=data
    )
    response = response.json()

    # Parse GPT output
    content = response["choices"][0]["message"]["content"]
    result = json.loads(content)
    return result


def create_assessment_prompt(col_summaries: str, vocab: str) -> str:
    prompt = f"""
Assume you are a clinical neuroscience researcher and you are examining a dataset table with the following columns.

Column summaries:
{col_summaries}

Based on the column name and the column summary info, first determine which of the columns
are about clinical/cognitive assessment scores or data. Give these columns an ID of "nb:Assessment".

For each column you have determined is about assessment data, also give your best guess for which specific assessment/instrument it is about,
using terms from the following SNOMED-CT vocabulary (try to expand abbreviations where possible):
{vocab}

Return a JSON data dictionary where each key is a column name, and each value is a dict with the following keys:
- "variable_id": "nb:Assessment" if the column is an assessment column, or "Other" if it is not
- "assessment_id": the value should be either 1) if the column describes assessment data, the SNOMED ID of the specific assessment the column is about,
in the format "snomed:<ID>" 2) "other assessment" if the column is about an assessment but it's unclear which one 3) "n/a" if the column is not about an assessment
- "assessment_label": the assessment "name" label from the SNOMED vocabulary, if applicable, otherwise "n/a"
- "confidence": one of "high", "medium", or "low" confidence based on how confident you are in the column mapping

Do not use SNOMED terms outside of the provided terms list, and do not exclude any input columns from the output dictionary.

Just return a plain JSON string, with no formatting. Do not provide any additional text or explanation.
"""
    return prompt


def create_dataset_tabular_summary(
    dataset_columns: pd.DataFrame, dataset_values: pd.DataFrame
) -> list[dict]:
    ds_columns_dict = dataset_columns.drop(columns=["dataset"]).to_dict(
        orient="records"
    )
    ds_values_by_column = dataset_values.groupby("column")

    value_summaries_by_column = {}
    for column_name, column_values in ds_values_by_column:
        value_summaries_by_column[column_name] = column_values.drop(
            columns=["column"]
        ).to_dict(orient="records")

    ds_columns_with_value_summaries = []
    for ds_column in ds_columns_dict:
        column_name = ds_column["column"]
        ds_columns_with_value_summaries.append(
            {
                **ds_column,
                "value_summaries": value_summaries_by_column.get(column_name, []),
            }
        )

    return ds_columns_with_value_summaries


def classify_assessments_in_datasets(
    column_summaries_path: Path, value_summaries_path: Path, out_dir: Path
):
    out_dir.mkdir(exist_ok=True)

    # test_datasets = ["ds004796", "ds005498", "ds005752"]
    assessment_vocab = fetch_file_from_url(ASSESSMENT_VOCAB_URL)
    assessment_vocab_str = json.dumps(assessment_vocab, indent=2)

    column_summaries = pd.read_csv(
        column_summaries_path,
        sep="\t",
        keep_default_na=False,
        dtype={
            "column": str,
            "standardized_var": str,
            "age_format": str,
            "exclude": str,
        },
    )
    column_summaries = column_summaries.drop(
        columns=["assessment_term", "assessment_label", "exclude"]
    )
    value_summaries = pd.read_csv(
        value_summaries_path,
        sep="\t",
        keep_default_na=False,
        dtype={
            "column": str,
            "value": str,
            "standardized_term": str,
            "standardized_label": str,
            "is_missing_value": str,
        },
    )

    ds_groups = column_summaries.groupby("dataset")

    for idx, (ds_id, ds_columns) in enumerate(ds_groups, start=1):
        logger.info(f"({idx}/{len(ds_groups)}): Processing dataset {ds_id}")
        if not (ds_columns["standardized_var"] == "").any():
            logger.warning(f"{ds_id}: All columns already annotated. Skipping.")
            continue
        ds_values = value_summaries[value_summaries["dataset"] == ds_id].drop(
            columns=["dataset"]
        )

        ds_tabular_data_summary = create_dataset_tabular_summary(ds_columns, ds_values)
        ds_tabular_data_summary_str = json.dumps(ds_tabular_data_summary, indent=2)
        prompt = create_assessment_prompt(
            ds_tabular_data_summary_str,
            assessment_vocab_str,
        )

        result = send_prompt_to_llm(prompt)
        results_path = out_dir / f"{ds_id}_assessments.json"
        save_json(result, results_path)


if __name__ == "__main__":
    COLUMN_SUMMARIES_PATH = (
        RESOURCES_DIR / "participants_tsv_columns_summary_first_guess_manual_pass.tsv"
    )
    VALUE_SUMMARIES_PATH = (
        RESOURCES_DIR
        / "participants_tsv_categorical_values_summary_first_guess_manual_pass.tsv"
    )
    classify_assessments_in_datasets(
        COLUMN_SUMMARIES_PATH, VALUE_SUMMARIES_PATH, OUT_DIR
    )
