import json
import logging
from pathlib import Path

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
    "nb:SessionID": ["session_id", "session"],
    "nb:Sex": ["sex", "gender"],
    "nb:Age": ["age", "age_years", "age_yrs", "participant_age"],
    "nb:Diagnosis": ["diagnosis", "dx", "group_dx", "group", "study_group"],
}
COMMON_MISSING_VALUES = ["none", "nan", "n/a", "na", "null", ""]
# arbitrary threshold for number of unique values to consider a column categorical
CATEGORICAL_COLUMN_THRESHOLD = 10


def read_tsv(path: Path) -> pd.DataFrame | None:
    """Read a TSV file into a DataFrame, and check if it has more than one column."""
    df = pd.read_csv(path, sep="\t")
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


def get_value_description(column_json_info: dict, value: str) -> str | None:
    """Get the description for a specific value in a column from the participants.json file, if it exists."""
    try:
        return column_json_info.get("Levels", {}).get(value, None)
    # In case the "Levels" key is incorrectly formatted
    except Exception:
        return None


def infer_if_missing_value(value: str) -> bool | None:
    if pd.isna(value) or str(value).lower() in COMMON_MISSING_VALUES:
        return True
    return None


def get_column_and_value_summaries(
    participants_tsv: pd.DataFrame, participants_json: dict
) -> tuple[list[dict], list[dict]]:
    col_summaries = []
    value_summaries = []
    for col_name, col_data in participants_tsv.items():
        column_json_info = participants_json.get(col_name, {})

        bids_levels = get_column_bids_levels(column_json_info)
        standardized_var = get_common_std_var_mapping(col_name)

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
            "standardized_var": standardized_var,
            "age_format": (
                infer_age_format(col_data) if standardized_var == "nb:Age" else None
            ),
        }
        if is_column_numeric_dtype(col_data):
            col_summary["min"] = col_data.min()
            col_summary["max"] = col_data.max()
        else:
            col_summary["min"] = None
            col_summary["max"] = None

        col_summaries.append(col_summary)

        if col_summary["is_categorical"] is True:
            # NOTE: This includes NaN values
            for col_value in col_data.unique():
                value_summary = {
                    "column": col_name,
                    "value": col_value,
                    "description": get_value_description(column_json_info, col_value),
                    "is_missing_value": infer_if_missing_value(col_value),
                }
                value_summaries.append(value_summary)

    return col_summaries, value_summaries


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
        "age_format",
    ]

    all_cat_values_df_order = [
        "dataset",
        "column",
        "value",
        "description",
        "is_missing_value",
        "standardized_term",
    ]

    all_columns_df = pd.DataFrame(columns=all_columns_df_order)
    all_cat_values_df = pd.DataFrame(columns=all_cat_values_df_order)

    for dataset_id in tqdm(datasets_to_annotate["dataset"], "Processing datasets"):
        participants_tsv = read_tsv(DATA_DIR / f"{dataset_id}.tsv")
        if participants_tsv is None:
            continue
        participants_json = load_json(DATA_DIR / f"{dataset_id}.json")

        dataset_columns, dataset_values = get_column_and_value_summaries(
            participants_tsv, participants_json
        )

        # Add column summaries for dataset to mega-table
        dataset_columns_df = pd.DataFrame(dataset_columns)
        dataset_columns_df["dataset"] = dataset_id
        # transform bids_levels into a string for readability
        dataset_columns_df["bids_levels"] = dataset_columns_df["bids_levels"].apply(
            lambda x: ", ".join(x) if isinstance(x, list) else x
        )
        # create empty column to be filled during annotation
        dataset_columns_df["assessment_term"] = None
        all_columns_df = pd.concat(
            [all_columns_df, dataset_columns_df], ignore_index=True
        )

        # Add column value summaries for dataset to mega-table
        dataset_values_df = pd.DataFrame(dataset_values)
        dataset_values_df["dataset"] = dataset_id
        # create empty column to be filled during annotation
        dataset_values_df["standardized_term"] = None
        all_cat_values_df = pd.concat(
            [all_cat_values_df, dataset_values_df], ignore_index=True
        )

    # Reorder columns in mega-tables
    all_columns_df = all_columns_df[all_columns_df_order]
    all_cat_values_df = all_cat_values_df[all_cat_values_df_order]

    all_columns_df.to_csv(COLUMN_SUMMARIES_OUT_FILE, sep="\t", index=False)
    all_cat_values_df.to_csv(VALUE_SUMMARIES_OUT_FILE, sep="\t", index=False)


if __name__ == "__main__":
    main()
