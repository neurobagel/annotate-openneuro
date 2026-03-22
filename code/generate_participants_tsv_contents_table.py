import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    datefmt="[%Y-%m-%d %H:%M:%S]",
    filename=Path(__file__).parent / "logs" / f"{Path(__file__).stem}.log",
)
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parents[1] / "data"
RESOURCES_DIR = Path(__file__).parents[1] / "resources"
DATASETS_TO_ANNOTATE = (
    RESOURCES_DIR / "openneuro_tabular_top_50_percent_unannotated.tsv"
)

COLUMN_SUMMARIES_OUT_FILE = RESOURCES_DIR / "participants_tsv_columns_summary.tsv"
VALUE_SUMMARIES_OUT_FILE = (
    RESOURCES_DIR / "participants_tsv_categorical_values_summary.tsv"
)

PARTICIPANT_ID_COLUMN = "participant_id"
EXPECTED_CATEGORICAL_COLUMNS = ["sex", "gender"]
COMMON_COLUMN_MAPPINGS = {
    "nb:ParticipantID": [
        "participant_id",
        "participant",
        "bids_id",
        "subject_id",
        "sub_id",
        "subjectid",
        "participantid",
    ],
    # NOTE: we do not include "ses" since it could be confused with socioeconomic status
    "nb:SessionID": ["session_id", "session"],
    "nb:Sex": ["sex", "gender"],
    "nb:Age": ["age", "age_years", "age_yrs", "participant_age"],
    "nb:Diagnosis": ["diagnosis", "dx", "group_dx", "group", "study_group"],
}
COMMON_SEX_VALUES = {
    "snomed:248153007": {
        "standardized_label": "Male",
        "common_values": ["male", "m", "man"],
    },
    "snomed:248152002": {
        "standardized_label": "Female",
        "common_values": ["female", "f", "woman"],
    },
}
# NOTE: Do not include "none", as it can be a legitimate coded value denoting absence of a condition
COMMON_MISSING_VALUES = ["nan", "n/a", "na", "null", "<na>", "#na", ""]

# arbitrary threshold for number of unique values to consider a column categorical
CATEGORICAL_COLUMN_THRESHOLD = 10


@lru_cache
def read_tsv(path: Path, as_strings: bool = False) -> pd.DataFrame | None:
    """Read a TSV file into a DataFrame, and check if it has more than one column."""
    if as_strings:
        df = pd.read_csv(path, sep="\t", keep_default_na=False, dtype=str)
    else:
        # Use nullable data types to ensure that int columns with missing values aren't upcast to float
        df = pd.read_csv(path, sep="\t", dtype_backend="numpy_nullable")
    if df.shape[1] == 1:
        logger.warning(f"{path.name} has only one column. Skipping.")
        return None
    return df


def load_json(path: Path) -> dict:
    """Load a JSON file (if it exists) and return its contents as a dictionary."""
    if not path.exists():
        return {}
    try:
        # Use encoding "utf-8-sig" to handle potential BOM in JSON files - common in Windows-created files
        with open(path, "r", encoding="utf-8-sig") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading JSON file {path.name}: {e}")
        return {}


def get_column_description(column_json_info: dict) -> str | None:
    """Get the description for a column from the participants.json file, if it exists."""
    return column_json_info.get("Description", None)


def get_column_bids_levels(column_json_info: dict) -> list:
    """Get the BIDS levels for a column from the participants.json file, if they exist."""
    try:
        return list(column_json_info.get("Levels", {}).keys())
    # In case the "Levels" key is incorrectly formatted
    except Exception:
        return []


def get_column_bids_units(column_json_info: dict) -> str | None:
    """Get the BIDS units for a column from the participants.json file, if they exist."""
    return column_json_info.get("Units", None)


def is_column_str_like_dtype(column_data: pd.Series) -> bool:
    """
    Check if a column has a string-like dtype (object or string).
    NOTE: On pandas>=3.0, string columns have default dtype "str" instead of "object".
    """
    return pd.api.types.is_object_dtype(column_data) or pd.api.types.is_string_dtype(
        column_data
    )


def is_column_numeric_dtype(column_data: pd.Series) -> bool:
    """
    Check if a column has a numeric dtype (integer or float).

    NOTE: boolean columns are treated as numeric dtype by default,
    so here we explicitly check for integer or float columns
    """
    return pd.api.types.is_integer_dtype(column_data) or pd.api.types.is_float_dtype(
        column_data
    )


def are_column_values_euro_decimals(column_data: pd.Series) -> bool:
    """Check if a column's values are in European decimal format (i.e. using commas as decimal separators)."""
    euro_decimal_pattern = r"^\d+,\d+$"
    if not is_column_str_like_dtype(column_data):
        return False
    values = column_data.dropna().astype(str)
    return values.str.match(euro_decimal_pattern).any()


def is_categorical_column_basic(column_data: pd.Series) -> bool | None:
    """Infer if a column is categorical based on basic heuristics (e.g. low number of unique values)."""
    column_name = column_data.name.lower()
    if column_name == PARTICIPANT_ID_COLUMN:
        return False
    if column_name in EXPECTED_CATEGORICAL_COLUMNS:
        return True
    if are_column_values_euro_decimals(column_data):
        return False
    if not is_column_str_like_dtype(column_data):
        # Account for boolean columns that might be number-coded
        if column_data.nunique() == 2:
            return True
    n_unique = column_data.nunique()
    if n_unique < CATEGORICAL_COLUMN_THRESHOLD:
        return True
    return None


def get_column_min_max(
    column_data: pd.Series,
) -> tuple[str | None, str | None]:
    """
    Get the minimum and maximum values for a numeric column, if applicable,
    and return them as strings to preserve the original values.
    This prevents upcasting of varied-type "min" and "max" values to float in the final dataframe.
    """
    if not is_column_numeric_dtype(column_data):
        return None, None
    min_val = column_data.min()
    max_val = column_data.max()
    return (
        str(min_val) if pd.notna(min_val) else None,
        str(max_val) if pd.notna(max_val) else None,
    )


def get_common_std_var_mapping(column_name: str) -> str | None:
    """Check if a column name matches any of the common column names for certain standardized variables."""
    column_name = column_name.lower()
    return next(
        (
            std_var
            for std_var, common_col_names in COMMON_COLUMN_MAPPINGS.items()
            if column_name in common_col_names
        ),
        None,
    )


def infer_age_format(column_data: pd.Series) -> str | None:
    """Determine if the column values match common age formats."""
    if is_column_numeric_dtype(column_data):
        return "nb:FromFloat"
    if are_column_values_euro_decimals(column_data):
        return "nb:FromEuro"
    return None


def get_value_description(column_json_info: dict, value: Any) -> str | None:
    """Get the description for a specific value in a column from the participants.json file, if it exists."""
    value = str(value)
    try:
        return column_json_info.get("Levels", {}).get(value, None)
    # In case the "Levels" key is incorrectly formatted
    except Exception:
        return None


def infer_if_missing_value(value: Any) -> bool | None:
    if pd.isna(value) or str(value).lower() in COMMON_MISSING_VALUES:
        return True
    return None


def get_common_std_term_mapping_for_sex_value(
    value: Any, description: Any
) -> tuple[str | None, str | None]:
    """Check if a value (or its description) matches any of the common values mapped to sex terms."""
    value = str(value).lower()
    description = description.lower() if not pd.isna(description) else description
    for std_term, std_term_info in COMMON_SEX_VALUES.items():
        if (
            value in std_term_info["common_values"]
            or description in std_term_info["common_values"]
        ):
            return std_term, std_term_info["standardized_label"]
    return None, None


def get_column_summaries(
    participants_tsv: pd.DataFrame, participants_json: dict
) -> list[dict]:
    col_summaries = []
    for col_name, col_data in participants_tsv.items():
        column_json_info = participants_json.get(col_name, {})

        bids_levels = get_column_bids_levels(column_json_info)

        col_summary = {
            "column": col_name,
            "dtype": str(col_data.dtype),
            "description": get_column_description(column_json_info),
            "n_unique_values": col_data.nunique(),
            "n_empty_values": int(col_data.isna().sum()),
            "bids_levels": bids_levels,
            "bids_units": get_column_bids_units(column_json_info),
            "is_categorical": bool(bids_levels)
            or is_categorical_column_basic(col_data) is True,
        }
        col_summary["min"], col_summary["max"] = get_column_min_max(col_data)
        col_summaries.append(col_summary)

    return col_summaries


def get_value_summaries(
    participants_tsv: pd.DataFrame, participants_json: dict
) -> list[dict]:
    # NOTE: participants_tsv in this case should contain only raw strings
    value_summaries = []
    for col_name, col_data in participants_tsv.items():
        column_json_info = participants_json.get(col_name, {})
        # NOTE: This includes NaN values
        # NOTE: Unique values in the actual column don't always correspond to values listed in the participants.json "Levels"
        # e.g., an empty cell might be represented in "Levels" as "n/a"
        for col_value in col_data.unique():
            value_summary = {
                "column": col_name,
                "value": col_value,
                "description": get_value_description(column_json_info, col_value),
                "is_missing_value": infer_if_missing_value(col_value),
            }
            value_summaries.append(value_summary)

    return value_summaries


def infer_age_column_formats(column_summaries: pd.DataFrame) -> pd.DataFrame:
    """
    Infer the age format for all columns mapped to nb:Age,
    based on the column values in the corresponding participants.tsv file.
    """
    column_summaries = column_summaries.copy()
    age_columns_mask = column_summaries["standardized_var"] == "nb:Age"
    for idx, row in tqdm(
        column_summaries[age_columns_mask].iterrows(),
        desc="Inferring age formats for nb:Age columns",
        total=age_columns_mask.sum(),
    ):
        dataset_id = row["dataset"]
        column_name = row["column"]
        participants_tsv = read_tsv(DATA_DIR / f"{dataset_id}.tsv")
        if participants_tsv is not None and column_name in participants_tsv.columns:
            age_format = infer_age_format(participants_tsv[column_name])
            column_summaries.at[idx, "age_format"] = age_format

    return column_summaries


def infer_std_terms_for_sex_column_values(
    column_summaries: pd.DataFrame, value_summaries: pd.DataFrame
) -> pd.DataFrame:
    """
    Infer standardized term mappings for values in all columns mapped to nb:Sex.
    """
    value_summaries = value_summaries.copy()
    sex_columns_mask = column_summaries["standardized_var"] == "nb:Sex"
    for _, row in tqdm(
        column_summaries[sex_columns_mask].iterrows(),
        desc="Inferring standardized terms for values found in nb:Sex columns",
        total=sex_columns_mask.sum(),
    ):
        dataset_id = row["dataset"]
        sex_column_name = row["column"]
        col_values_mask = (value_summaries["dataset"] == dataset_id) & (
            value_summaries["column"] == sex_column_name
        )
        for value_row_idx, value_row in value_summaries[col_values_mask].iterrows():
            std_term, std_label = get_common_std_term_mapping_for_sex_value(
                value_row["value"], value_row["description"]
            )
            value_summaries.at[value_row_idx, "standardized_term"] = std_term
            value_summaries.at[value_row_idx, "standardized_label"] = std_label

    return value_summaries


def main():
    datasets_to_annotate = pd.read_csv(DATASETS_TO_ANNOTATE, sep="\t")

    all_columns_df_order = [
        "dataset",
        "column",
        "dtype",
        "description",
        "n_unique_values",
        "n_empty_values",
        "bids_levels",
        "bids_units",
        "is_categorical",
        "min",
        "max",
        "standardized_var",
        "assessment_term",
        "assessment_label",
        "age_format",
    ]

    all_cat_values_df_order = [
        "dataset",
        "column",
        "value",
        "description",
        "is_missing_value",
        "standardized_term",
        "standardized_label",
    ]

    all_columns_df = pd.DataFrame(columns=all_columns_df_order)
    all_cat_values_df = pd.DataFrame(columns=all_cat_values_df_order)

    for dataset_idx, dataset_id in enumerate(
        tqdm(datasets_to_annotate["dataset"], "Processing datasets"), start=1
    ):
        participants_tsv = read_tsv(DATA_DIR / f"{dataset_id}.tsv")
        if participants_tsv is None:
            continue
        participants_json = load_json(DATA_DIR / f"{dataset_id}.json")

        dataset_columns = get_column_summaries(participants_tsv, participants_json)
        dataset_columns_df = pd.DataFrame(dataset_columns)
        dataset_columns_df["dataset"] = dataset_id
        # transform bids_levels into a string for readability
        dataset_columns_df["bids_levels"] = dataset_columns_df["bids_levels"].apply(
            lambda x: ", ".join(x) if isinstance(x, list) else x
        )
        # add column summaries for dataset to mega-table
        all_columns_df = pd.concat(
            [all_columns_df, dataset_columns_df], ignore_index=True
        )

        # log number of detected categorical columns
        logger.info(
            f"({dataset_idx}/{len(datasets_to_annotate)}) {dataset_id}: "
            f"Number of categorical columns detected: {dataset_columns_df['is_categorical'].sum()}/{len(dataset_columns_df)}"
        )

        # NOTE: Re-read all columns as strings in order to generate the value summaries table,
        # to ensure that values are captured as they originally appear
        participants_tsv_all_str = read_tsv(
            DATA_DIR / f"{dataset_id}.tsv", as_strings=True
        )
        categorical_columns = dataset_columns_df.loc[
            dataset_columns_df["is_categorical"].eq(True), "column"
        ].tolist()
        dataset_values = get_value_summaries(
            participants_tsv_all_str[categorical_columns], participants_json
        )
        dataset_values_df = pd.DataFrame(dataset_values)
        dataset_values_df["dataset"] = dataset_id
        # add column value summaries for dataset to mega-table
        all_cat_values_df = pd.concat(
            [all_cat_values_df, dataset_values_df], ignore_index=True
        )

    # Reorder columns in mega-tables
    all_columns_df = all_columns_df[all_columns_df_order]
    all_cat_values_df = all_cat_values_df[all_cat_values_df_order]

    # Save summary tables
    all_columns_df.to_csv(COLUMN_SUMMARIES_OUT_FILE, sep="\t", index=False)
    all_cat_values_df.to_csv(VALUE_SUMMARIES_OUT_FILE, sep="\t", index=False)

    # Take first guess at standardized variable mappings
    all_columns_df_first_guess = all_columns_df.copy()
    all_columns_df_first_guess["standardized_var"] = all_columns_df_first_guess[
        "column"
    ].apply(get_common_std_var_mapping)
    # Take first guess at age format for columns mapped to nb:Age
    all_columns_df_first_guess = infer_age_column_formats(all_columns_df_first_guess)
    # Take first guess at standardized term mappings for values in columns mapped to nb:Sex
    all_cat_values_df_first_guess = infer_std_terms_for_sex_column_values(
        all_columns_df_first_guess, all_cat_values_df
    )

    # Save summary tables with first guesses
    all_columns_df_first_guess.to_csv(
        RESOURCES_DIR / f"{COLUMN_SUMMARIES_OUT_FILE.stem}_first_guess.tsv",
        sep="\t",
        index=False,
    )
    all_cat_values_df_first_guess.to_csv(
        RESOURCES_DIR / f"{VALUE_SUMMARIES_OUT_FILE.stem}_first_guess.tsv",
        sep="\t",
        index=False,
    )


if __name__ == "__main__":
    main()
