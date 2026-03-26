"""
Summarize OpenNeuro participants.tsv files where the file is non-empty or non-header-only, including:
- the number of participants in the participants.tsv (based on presence of a participant_id column)
- whether the dataset has already been annotated using Neurobagel
- which datasets are needed to cover ~50% of all participants across datasets
"""

import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
)
logger = logging.getLogger(Path(__file__).stem)


OPENNEURO_DATA_DIR = Path(__file__).parents[1] / "data"
RESOURCES_DIR = Path(__file__).parents[1] / "resources"
# Assumes openneuro-annotations has been cloned into the parent directory of this repo
OPENNEURO_ANNOTATIONS_DIR = Path(__file__).parents[2] / "openneuro-annotations"


def write_tsv(df: pd.DataFrame, path: Path):
    df.to_csv(path, sep="\t", index=False)


def add_dataset_annotated_status(
    overview: pd.DataFrame, annotations_dir: Path
) -> pd.DataFrame:
    overview = overview.copy()
    overview["annotated"] = overview["dataset"].apply(
        lambda ds: (annotations_dir / f"{ds}.json").exists()
    )
    return overview


def get_datasets_covering_x_percent_participants(
    overview: pd.DataFrame, percentage=0.5
) -> pd.DataFrame:
    overview = overview.copy()
    total_participants = overview["n_participants"].sum()
    top_x_perc = overview[
        overview["cumulative_n_participants"] <= (total_participants * percentage)
    ]
    # include the dataset that pushes it over x%
    top_x_perc = overview.iloc[: len(top_x_perc) + 1]
    return top_x_perc


def main():
    all_datasets_overview = pd.read_csv(RESOURCES_DIR / "data_overview.tsv", sep="\t")

    total_datasets = len(all_datasets_overview)

    # Remove datasets where the participants.tsv is empty or only has a header row (n_rows = 0)
    datasets_with_tsvs = all_datasets_overview[
        all_datasets_overview["n_rows"] > 0
    ].copy()
    logger.info(
        f"Datasets with participants.tsv containing >=1 data rows: {len(datasets_with_tsvs)}/{total_datasets}"
    )

    # Count number of participants in each dataset based on presence of participant_id column in participants.tsv
    datasets_with_tsvs["n_participants"] = None
    datasets_missing_participant_id_col = []
    unparseable_datasets = []
    for idx, row in datasets_with_tsvs.iterrows():
        ds_id = row["dataset"]
        try:
            participants_tsv = pd.read_csv(
                OPENNEURO_DATA_DIR / f"{ds_id}.tsv", sep="\t"
            )
        except Exception as err:
            logger.warning(
                f"{ds_id}.tsv: Could not parse file due to error {err}, skipping n_participants calculation"
            )
            unparseable_datasets.append(ds_id)
            continue
        if "participant_id" in participants_tsv.columns:
            n_participants = participants_tsv["participant_id"].nunique()
            datasets_with_tsvs.at[idx, "n_participants"] = n_participants
        else:
            logger.warning(f"{ds_id}.tsv: does not have a 'participant_id' column")
            datasets_missing_participant_id_col.append(ds_id)

    logger.info(
        f"Datasets missing 'participant_id' column ({len(datasets_missing_participant_id_col)}/{len(datasets_with_tsvs)}): "
        f"{', '.join(datasets_missing_participant_id_col)}"
    )
    logger.info(
        f"Datasets that could not be parsed ({len(unparseable_datasets)}/{len(datasets_with_tsvs)}): "
        f"{', '.join(unparseable_datasets)}"
    )

    # Add info about whether the dataset has already been annotated using Neurobagel
    datasets_with_tsvs = add_dataset_annotated_status(
        datasets_with_tsvs, OPENNEURO_ANNOTATIONS_DIR
    )
    datasets_with_tsvs_sorted = datasets_with_tsvs.sort_values(
        by="n_participants", ascending=False
    )

    datasets_with_tsvs_sorted["cumulative_n_participants"] = datasets_with_tsvs_sorted[
        "n_participants"
    ].cumsum()
    # Determine datasets needed to cover ~50% of all participants across datasets
    top_50_perc = get_datasets_covering_x_percent_participants(
        overview=datasets_with_tsvs_sorted, percentage=0.5
    )
    top_50_perc_unannotated = top_50_perc[~top_50_perc["annotated"]]
    # Get unannotated dataset in top 50 with the biggest n_columns
    unannotated_max_cols_row = top_50_perc_unannotated.loc[
        top_50_perc_unannotated["n_columns"].idxmax(), ["dataset", "n_columns"]
    ]
    # Get unannotated dataset in top 50 with the smallest n_columns
    unannotated_min_cols_row = top_50_perc_unannotated.loc[
        top_50_perc_unannotated["n_columns"].idxmin(), ["dataset", "n_columns"]
    ]

    logger.info(
        f"Datasets needed to cover ~50% of participants: {len(top_50_perc)}/{len(datasets_with_tsvs_sorted)}"
    )
    logger.info(
        f"Num. annotated datasets in top 50%: {top_50_perc['annotated'].sum()}/{len(top_50_perc)}"
    )
    logger.info(
        f"Num. datasets requiring annotation in top 50%: {len(top_50_perc_unannotated)}/{len(top_50_perc)}"
    )
    logger.info(
        f"Unannotated dataset in top 50% with most columns: {unannotated_max_cols_row['dataset']}, n_columns: {unannotated_max_cols_row['n_columns']}"
    )
    logger.info(
        f"Unannotated dataset in top 50% with fewest columns: {unannotated_min_cols_row['dataset']}, n_columns: {unannotated_min_cols_row['n_columns']}"
    )

    write_tsv(datasets_with_tsvs_sorted, RESOURCES_DIR / "openneuro_tabular.tsv")
    write_tsv(top_50_perc, RESOURCES_DIR / "openneuro_tabular_top_50_percent.tsv")
    write_tsv(
        top_50_perc_unannotated,
        RESOURCES_DIR / "openneuro_tabular_top_50_percent_unannotated.tsv",
    )


if __name__ == "__main__":
    main()
