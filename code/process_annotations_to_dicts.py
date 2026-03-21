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
    filename=Path(__file__).parent / "logs" / f"{Path(__file__).stem}.log",
)
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parents[1] / "data"
OUT_DIR = DATA_DIR
RESOURCES_DIR = Path(__file__).parents[1] / "resources"
COLUMN_SUMMARIES_PATH = (
    RESOURCES_DIR / "participants_tsv_columns_summary_first_guess_manual_pass.tsv"
)
VALUE_SUMMARIES_PATH = (
    RESOURCES_DIR
    / "participants_tsv_categorical_values_summary_first_guess_manual_pass.tsv"
)

NEUROBAGEL_VARIABLES_VOCAB_URL = "https://raw.githubusercontent.com/neurobagel/communities/refs/heads/main/configs/Neurobagel/config.json"

COMMON_MISSING_VALUES = ["n/a", "N/A", "na", "NA", "nan", "NaN", ""]


def fetch_neurobagel_standardized_vars_as_dict() -> dict:
    response = requests.get(NEUROBAGEL_VARIABLES_VOCAB_URL)
    response.raise_for_status()
    vocab = response.json()[0]

    standardized_vars = {}
    # var_prefix = vocab["namespace_prefix"]
    for standardized_var in vocab["standardized_variables"]:
        var_term_url = f"nb:{standardized_var['id']}"
        standardized_vars[var_term_url] = standardized_var

    return standardized_vars


NEUROBAGEL_VARS_VOCAB = fetch_neurobagel_standardized_vars_as_dict()


def get_formats_for_variable(var_term_url: str) -> dict:
    available_formats = NEUROBAGEL_VARS_VOCAB[var_term_url]["formats"]
    formats_dict = {}
    for available_format in available_formats:
        formats_dict[f"nb:{available_format['id']}"] = available_format["name"]
    return formats_dict


AGE_FORMAT_LABELS = get_formats_for_variable("nb:Age")


def load_participants_json(dataset: str, path: Path) -> dict:
    """Load a participants.json file (if it exists) and return its contents as a dictionary."""
    if not path.exists():
        logger.warning(f"{dataset}: No participants.json available")
        return {}
    try:
        # Use encoding "utf-8-sig" to handle potential BOM in JSON files - common in Windows-created files
        with open(path, "r", encoding="utf-8-sig") as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"{dataset}: Error loading JSON file {path.name}: {e}")
        return {}


def save_json(data: dict, path: Path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def any_columns_annotated(dataset_columns: pd.DataFrame) -> bool:
    return dataset_columns["standardized_var"].any()


def is_identifier_column(var_term_url: str) -> bool:
    identifier_vars = ["nb:ParticipantID", "nb:SessionID"]
    return var_term_url in identifier_vars


def annotate_term(term_url: str, label: str) -> dict:
    return {"TermURL": term_url, "Label": label}


def get_isabout(var_term_url: str) -> dict:
    return {
        "IsAbout": annotate_term(
            var_term_url, NEUROBAGEL_VARS_VOCAB[var_term_url]["name"]
        )
    }


def get_variabletype(var_term_url: str) -> dict:
    return {"VariableType": NEUROBAGEL_VARS_VOCAB[var_term_url]["variable_type"]}


def get_base_annotations(var_term_url: str) -> dict:
    return {**get_isabout(var_term_url), **get_variabletype(var_term_url)}


def get_identifier_annotations(var_term_url: str) -> dict:
    return get_base_annotations(var_term_url)


def get_age_annotations(column_row: pd.Series, column_values: pd.DataFrame) -> dict:
    format_term_url = column_row["age_format"]
    missing_values = []

    if not format_term_url:
        return {}
    if format_term_url not in AGE_FORMAT_LABELS:
        logger.warning(
            f"{column_row['dataset']}: Unknown age format for column '{column_row['column']}': '{format_term_url}'. "
            "Skipping age annotations for this column."
        )
        return {}

    format_annotation = annotate_term(
        format_term_url, AGE_FORMAT_LABELS[format_term_url]
    )

    if column_row["n_empty_values"] > 0:
        # For age columns with few enough unique values to have been detected as 'categorical',
        # we could already have annotations for the specific values detected as missing values.
        # TODO: This is an extra precautionary step that might be able to be removed in future,
        # since most age columns will be detected as continuous
        detected_missing_values = column_values.loc[
            column_values["is_missing_value"] == True, "value"  # noqa: E712
        ].tolist()
        missing_values = list({*detected_missing_values, *COMMON_MISSING_VALUES})

    annotations = {
        **get_base_annotations("nb:Age"),
        "Format": format_annotation,
        "MissingValues": missing_values,
    }
    return annotations


def get_sex_annotations(column_values: pd.DataFrame) -> dict:
    levels = {}
    missing_values = []
    for _, value_row in column_values.iterrows():
        value = value_row["value"]
        if (term_url := value_row["standardized_term"]) and (
            label := value_row["standardized_label"]
        ):
            levels[value] = annotate_term(term_url, label)
        else:
            missing_values.append(value)

    annotations = {
        **get_base_annotations("nb:Sex"),
        "Levels": levels,
        "MissingValues": missing_values,
    }
    return annotations


def process_annotations_to_dict(
    dataset: str,
    dataset_columns: pd.DataFrame,
    dataset_values: pd.DataFrame,
    data_dict: dict,
) -> dict:
    columns_with_annotations_added = 0
    for _, ds_column in dataset_columns.iterrows():
        column_name = ds_column["column"]
        ds_column_values = dataset_values[dataset_values["column"] == column_name]

        if (
            data_dict.get(column_name) is None
            or data_dict[column_name].get("Description") is None
        ):
            data_dict[column_name] = {"Description": ""}

        standardized_var = ds_column["standardized_var"]
        column_annotations = {}
        if ds_column["exclude"] == True or not standardized_var:  # noqa: E712
            pass
        elif is_identifier_column(standardized_var):
            column_annotations = get_identifier_annotations(standardized_var)
        elif standardized_var == "nb:Age":
            column_annotations = get_age_annotations(ds_column, ds_column_values)
        elif standardized_var == "nb:Sex":
            column_annotations = get_sex_annotations(ds_column_values)

        if column_annotations:
            data_dict[column_name]["Annotations"] = column_annotations
            columns_with_annotations_added += 1

    logger.info(
        f"{dataset}: Neurobagel annotations added for {columns_with_annotations_added}/{len(dataset_columns)} columns."
    )
    return data_dict


def main():
    """
    TODO:
    - Validate data dictionary
    - Process assessment annotations
    """
    OUT_DIR.mkdir(exist_ok=True)

    column_summaries = pd.read_csv(
        COLUMN_SUMMARIES_PATH,
        sep="\t",
        dtype={"column": str, "standardized_var": str, "age_format": str},
        keep_default_na=False,
    )
    value_summaries = pd.read_csv(
        VALUE_SUMMARIES_PATH,
        sep="\t",
        dtype={
            "column": str,
            "value": str,
            "standardized_term": str,
            "standardized_label": str,
        },
        keep_default_na=False,
    )

    ds_groups = column_summaries.groupby("dataset")
    annotated_data_dicts_created = 0

    # TODO: Remove subsetting for testing
    for idx, (ds_id, ds_columns) in enumerate(
        tqdm(list(ds_groups)[:3], desc="Processing datasets"), start=1
    ):
        out_file = OUT_DIR / f"{ds_id}_annotated.json"
        logger.info(f"({idx}/{len(ds_groups)}): Processing dataset: {ds_id}")

        # Sanity check - do not attempt to create a Neurobagel data dictionary if no columns have been annotated
        if not any_columns_annotated(ds_columns):
            logger.info(
                f"{ds_id}: No columns mapped to standardized variables. Skipping dataset."
            )
            continue

        data_dict = load_participants_json(ds_id, DATA_DIR / f"{ds_id}.json")
        ds_values = value_summaries[value_summaries["dataset"] == ds_id]

        data_dict = process_annotations_to_dict(ds_id, ds_columns, ds_values, data_dict)

        save_json(data_dict, out_file)
        logger.info(f"{ds_id}: Saved annotated data dictionary to {out_file.name}")
        annotated_data_dicts_created += 1

    logger.info(
        f"Created annotated data dictionaries for {annotated_data_dicts_created}/{len(ds_groups)} datasets."
    )


if __name__ == "__main__":
    main()
