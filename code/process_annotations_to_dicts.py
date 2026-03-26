import json
import logging
from pathlib import Path

import jsonschema
import pandas as pd
import requests
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="[%Y-%m-%d %H:%M:%S]",
    filename=Path(__file__).parent / "logs" / f"{Path(__file__).stem}.log",
    filemode="w",
)
logger = logging.getLogger(Path(__file__).stem)

ROOT_PATH = Path(__file__).parents[1]
DATA_DIR = ROOT_PATH / "data"
RESOURCES_DIR = ROOT_PATH / "resources"
OUT_DIR = DATA_DIR / "annotated_dictionaries"

NEUROBAGEL_VARIABLES_VOCAB_URL = "https://raw.githubusercontent.com/neurobagel/communities/refs/heads/main/configs/Neurobagel/config.json"
NEUROBAGEL_ASSESSMENTS_VOCAB_URL = "https://raw.githubusercontent.com/neurobagel/communities/refs/heads/main/configs/Neurobagel/assessment.json"
NEUROBAGEL_DATA_DICT_SCHEMA_URL = "https://raw.githubusercontent.com/neurobagel/bagelschema/refs/heads/main/neurobagel_data_dictionary.schema.json"

# NOTE: All values in this list should be interpreted by pd.read_csv as missing values by default
# https://pandas.pydata.org/pandas-docs/stable/user_guide/io.html#na-values
COMMON_MISSING_VALUES = ["n/a", "N/A", "na", "NA", "nan", "NaN", ""]

TEST_DATASETS = ["ds004856", "ds005237"]


def fetch_file_from_url(url: str) -> dict:
    response = requests.get(url)
    response.raise_for_status()
    return response.json()


DATA_DICT_SCHEMA = fetch_file_from_url(NEUROBAGEL_DATA_DICT_SCHEMA_URL)


def fetch_neurobagel_standardized_vars_as_dict() -> dict:
    vocab = fetch_file_from_url(NEUROBAGEL_VARIABLES_VOCAB_URL)[0]
    standardized_vars = {}
    # var_prefix = vocab["namespace_prefix"]
    for standardized_var in vocab["standardized_variables"]:
        var_term_url = f"nb:{standardized_var['id']}"
        standardized_vars[var_term_url] = standardized_var

    return standardized_vars


NEUROBAGEL_VARS_VOCAB = fetch_neurobagel_standardized_vars_as_dict()


def get_formats_for_variable(var_term_url: str) -> dict:
    formats_dict = {}
    for available_format in NEUROBAGEL_VARS_VOCAB[var_term_url]["formats"]:
        formats_dict[f"nb:{available_format['id']}"] = available_format["name"]
    return formats_dict


AGE_FORMAT_LABELS = get_formats_for_variable("nb:Age")


def fetch_assessments_vocabulary_as_dict() -> dict:
    vocab = fetch_file_from_url(NEUROBAGEL_ASSESSMENTS_VOCAB_URL)[0]
    assessments = {}
    for term in vocab["terms"]:
        term_url = f"snomed:{term['id']}"
        assessments[term_url] = term["name"]
    return assessments


NEUROBAGEL_ASSESSMENTS_VOCAB = fetch_assessments_vocabulary_as_dict()


def get_single_instance_variables() -> list:
    return [
        var_term_url
        for var_term_url, var_info in NEUROBAGEL_VARS_VOCAB.items()
        if var_info["can_have_multiple_columns"] is False
    ]


def load_json(path: Path) -> dict:
    # Use encoding "utf-8-sig" to handle potential BOM in JSON files - common in Windows-created files
    with open(path, "r", encoding="utf-8-sig") as f:
        return json.load(f)


def load_participants_json(dataset: str, path: Path) -> dict:
    """Load a participants.json file (if it exists) and return its contents as a dictionary."""
    if not path.exists():
        logger.warning(f"{dataset}: No participants.json available")
        return {}
    try:
        return load_json(path)
    except Exception as e:
        logger.warning(f"{dataset}: Error loading JSON file {path.name}: {e}")
        return {}


def save_json(data: dict, path: Path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def get_annotated_columns_in_data_dict_by_dataset(
    column_summaries: pd.DataFrame,
) -> dict:
    """
    Create a dict containing lists of columns with and without Neurobagel data dictionary annotations by dataset,
    based on the annotated data dictionaries created by this script and the original column summaries TSV.

    We use the annotated data dictionaries created by this script as the source of truth for
    which columns have a complete and validated annotation (e.g., some columns in the column summaries table
    may be mapped to a standardized variable but could be lacking value annotations or have other data quality issues).

    The output dictionary can be saved as a reference to inform future annotation efforts.
    """
    column_annotations_overview = {}
    for ds_id, ds_columns in column_summaries.groupby("dataset"):
        annotated_columns = []
        unannotated_columns = []

        annotated_dict_path = OUT_DIR / f"{ds_id}_annotated.json"
        if annotated_dict_path.exists():
            annotated_dict = load_json(annotated_dict_path)
            for column, contents in annotated_dict.items():
                if "Annotations" in contents:
                    annotated_columns.append(column)
                else:
                    unannotated_columns.append(column)
        else:
            unannotated_columns = ds_columns["column"].tolist()

        column_annotations_overview[ds_id] = {
            "total_columns": len(ds_columns),
            "annotated_columns": {
                "count": len(annotated_columns),
                "names": annotated_columns,
            },
            "unannotated_columns": {
                "count": len(unannotated_columns),
                "names": unannotated_columns,
            },
        }

    return column_annotations_overview


def any_columns_annotated(dataset_columns: pd.DataFrame) -> bool:
    """
    Return True if any columns in the dataset have been mapped to a standardized variable,
    either through manual/heuristic-based annotation or LLM classification.
    """
    return (
        dataset_columns["standardized_var"].any()
        or dataset_columns["llm_classification"].any()
    )


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

    # For age columns with few enough unique values to have been marked 'categorical' in the column summaries table,
    # we may already have annotations for the specific values detected as being missing values.
    # TODO: This extra check can probably be removed in future, since most age columns will be detected as continuous.
    detected_missing_values = column_values.loc[
        column_values["is_missing_value"].str.lower() == "true", "value"
    ].tolist()
    # Also include common missing values by default as a workaround for not accurately detecting them in age columns
    # (especially columns where unique values are not available in the value summaries table).
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
        term_url = value_row["standardized_term"]
        term_label = value_row["standardized_label"]
        if term_url and term_label:
            levels[value] = annotate_term(term_url, term_label)
        else:
            if term_url and not term_label:
                logger.warning(
                    f"{value_row['dataset']}: Missing label for standardized term '{term_url}'. "
                    f"Column: '{value_row['column']}', value: '{value}'"
                )
            # Any unannotated values must be included as missing to avoid CLI errors later on
            # This will also capture any values where "is_missing_value" was marked True,
            # since those rows will also have empty values for "standardized_term" and "standardized_label"
            missing_values.append(value)

    annotations = {
        **get_base_annotations("nb:Sex"),
        "Levels": levels,
        "MissingValues": missing_values,
    }
    return annotations


def get_assessment_annotations(
    column_row: pd.Series, column_values: pd.DataFrame
) -> dict:
    term_id = ""
    if manual_term_id := column_row["assessment_term"]:
        term_id = manual_term_id
    elif column_row["reviewer_rating"] == "correct":
        term_id = column_row["llm_snomed_term"]
    else:
        # Assessment annotation was invalid or uses an unsupported assessment term
        return {}

    # NOTE: Assessment term IDs in the table did not have prefixes in order to enable easier lookup,
    # so we add it back here
    snomed_term_id = f"snomed:{term_id.removeprefix('snomed:')}"

    term_label = NEUROBAGEL_ASSESSMENTS_VOCAB.get(snomed_term_id)
    if not term_label:
        logger.warning(
            f"{column_row['dataset']}: Unknown assessment term for column {column_row['column']}: {term_id}. "
            "Skipping assessment annotation."
        )
        return {}

    # Include any values that may already have been marked as missing
    detected_missing_values = column_values.loc[
        column_values["is_missing_value"].str.lower() == "true", "value"
    ].tolist()
    # Also include common missing values by default as a workaround for not accurately
    # detecting them in assessment columns
    missing_values = list({*detected_missing_values, *COMMON_MISSING_VALUES})

    annotations = {
        **get_base_annotations("nb:Assessment"),
        "IsPartOf": annotate_term(snomed_term_id, term_label),
        "MissingValues": missing_values,
    }

    return annotations


def process_dataset_annotations_to_dict(
    dataset: str,
    dataset_columns: pd.DataFrame,
    dataset_values: pd.DataFrame,
    data_dict: dict,
) -> dict:
    columns_with_annotations_added = 0
    for _, ds_column in dataset_columns.iterrows():
        column_name = ds_column["column"]
        ds_column_values = dataset_values[dataset_values["column"] == column_name]

        # Ensure each column has at least a description to satisfy the Neurobagel data dictionary model
        data_dict.setdefault(column_name, {}).setdefault("Description", "")

        standardized_var = ds_column["standardized_var"]
        llm_classified_var = ds_column["llm_classification"]
        column_annotations = {}
        if ds_column["exclude"].lower() == "true" or (
            not standardized_var and not llm_classified_var
        ):
            pass
        elif is_identifier_column(standardized_var):
            column_annotations = get_identifier_annotations(standardized_var)
        elif standardized_var == "nb:Age":
            column_annotations = get_age_annotations(ds_column, ds_column_values)
        elif standardized_var == "nb:Sex":
            column_annotations = get_sex_annotations(ds_column_values)
        # Assessments were annotated by an LLM and then human-reviewed
        elif llm_classified_var == "nb:Assessment":
            column_annotations = get_assessment_annotations(ds_column, ds_column_values)

        if column_annotations:
            # TODO: Eventually check for and handle any existing annotations for the column?
            data_dict[column_name]["Annotations"] = column_annotations
            columns_with_annotations_added += 1

    logger.info(
        f"{dataset}: Neurobagel annotations added for {columns_with_annotations_added}/{len(dataset_columns)} columns."
    )
    if columns_with_annotations_added == 0:
        return {}
    return data_dict


def is_valid_data_dict(dataset: str, data_dict: dict) -> bool:
    """Return True for a valid Neurobagel data dictionary."""
    try:
        jsonschema.validate(instance=data_dict, schema=DATA_DICT_SCHEMA)
        return True
    except jsonschema.ValidationError as err:
        logger.error(f"{dataset}: Output data dictionary validation errors: {err}")
        return False


def mark_duplicate_single_instance_vars_for_exclusion(
    column_summaries: pd.DataFrame,
) -> pd.DataFrame:
    """
    Mark select duplicate column annotations for exclusion.

    For standardized variables that support a max of 1 mapped column per dataset,
    include only the first annotated column (based on original TSV column order)
    and mark the rest for exclusion so their annotations are not added to the data dictionary.

    The original TSV column order is used to infer variable precedence
    (e.g., in a TSV with columns ["participant_id", "age", "age_of_disease_onset"],
    "age" is most likely to be the primary age column),
    and to avoid dependency on the column summaries row order, which can change.

    TODO: We may want to write out the column summaries with the updated "exclude" column.
    """
    single_instance_vars = get_single_instance_variables()
    column_summaries = column_summaries.copy()

    for ds_id, ds_columns in tqdm(
        list(column_summaries.groupby("dataset")),
        desc="Checking single-column variable annotations in datasets",
    ):
        ds_columns = ds_columns.copy()
        ds_tsv_column_order = pd.read_csv(
            DATA_DIR / f"{ds_id}.tsv", sep="\t", nrows=0
        ).columns.tolist()
        ds_columns["_column_order"] = ds_columns["column"].map(
            {col: idx for idx, col in enumerate(ds_tsv_column_order)}
        )
        ds_columns = ds_columns.sort_values("_column_order").drop(
            columns="_column_order"
        )
        standardized_var_duplicates_mask = ds_columns["standardized_var"].isin(
            single_instance_vars
        ) & ds_columns.duplicated(subset=["standardized_var"], keep="first")

        for var in single_instance_vars:
            duplicated_columns = ds_columns[
                standardized_var_duplicates_mask
                & (ds_columns["standardized_var"] == var)
            ]["column"].tolist()
            if duplicated_columns:
                logger.warning(
                    f"{ds_id}: Skipping columns with duplicate annotation to '{var}': {duplicated_columns}."
                )

        column_summaries.loc[
            ds_columns.index[standardized_var_duplicates_mask], "exclude"
        ] = "true"

    return column_summaries


def process_annotations_to_dicts(
    column_summaries_path: Path, value_summaries_path: Path
):
    OUT_DIR.mkdir(exist_ok=True)

    # NOTE: keep_default_na=False prevents pandas from converting 'empty' cells to NaN,
    # which may cause columns to not be inferred as non-string dtypes due to resulting mixed values
    column_summaries = pd.read_csv(
        column_summaries_path,
        sep="\t",
        dtype={
            "column": str,
            "standardized_var": str,
            "age_format": str,
            "exclude": str,
            "llm_classification": str,  # "nb:Assessment" for columns classified as an assessment by the LLM
            "llm_snomed_term": str,  # LLM-designated SNOMED term ID
            "reviewer_rating": str,  # one of "correct", "missed", or "incorrect" for LLM-annotated columns
            "assessment_term": str,  # manually annotated SNOMED term ID for columns missed / incorrectly annotated by LLM
        },
        keep_default_na=False,
    )
    # Drop some columns we don't need
    column_summaries = column_summaries.drop(
        columns=["llm_confidence", "reviewer_notes", "reviewer_name"]
    )

    value_summaries = pd.read_csv(
        value_summaries_path,
        sep="\t",
        dtype={
            "column": str,
            "value": str,
            "standardized_term": str,
            "standardized_label": str,
            "is_missing_value": str,
        },
        keep_default_na=False,
    )

    column_summaries = mark_duplicate_single_instance_vars_for_exclusion(
        column_summaries
    )

    ds_groups = column_summaries.groupby("dataset")
    annotated_data_dicts_created = 0
    for idx, (ds_id, ds_columns) in enumerate(
        tqdm(list(ds_groups), desc="Processing datasets"), start=1
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

        data_dict = process_dataset_annotations_to_dict(
            ds_id, ds_columns, ds_values, data_dict
        )

        if not data_dict:
            logger.warning(
                f"{ds_id}: No Neurobagel annotations could be added to the data dictionary. Skipping save."
            )
            continue
        if not is_valid_data_dict(ds_id, data_dict):
            logger.error(
                f"{ds_id}: Output is not a valid Neurobagel data dictionary. Skipping save."
            )
            continue

        save_json(data_dict, out_file)
        logger.info(f"{ds_id}: Saved annotated data dictionary to {out_file.name}")
        annotated_data_dicts_created += 1

    logger.info(
        f"Created annotated data dictionaries for {annotated_data_dicts_created}/{len(ds_groups)} datasets."
    )

    # Save lists of annotated vs unannotated columns by dataset as a reference
    annotated_columns_by_dataset = get_annotated_columns_in_data_dict_by_dataset(
        column_summaries
    )
    save_json(
        annotated_columns_by_dataset,
        RESOURCES_DIR / "annotated_columns_by_dataset.json",
    )


if __name__ == "__main__":
    COLUMN_SUMMARIES_PATH = (
        RESOURCES_DIR
        / "participants_tsv_columns_summary_with_reviewed_assessment_annotations.tsv"
    )
    VALUE_SUMMARIES_PATH = (
        RESOURCES_DIR
        / "participants_tsv_categorical_values_summary_first_guess_manual_pass.tsv"
    )
    process_annotations_to_dicts(COLUMN_SUMMARIES_PATH, VALUE_SUMMARIES_PATH)
