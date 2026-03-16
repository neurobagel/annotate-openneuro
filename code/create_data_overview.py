"""
Create a table listing the number of rows and columns in all available participants.tsv files from OpenNeuro.
"""

import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


OPENNEURO_DATA_DIR = Path(__file__).parents[1] / "data"
RESOURCES_DIR = Path(__file__).parents[1] / "resources"


def write_participants_tsv_overview(data_dir: Path, out_file: Path):
    all_participants_tsvs = list(data_dir.glob("*.tsv"))

    all_datasets_overview_rows = []
    unparseable_datasets = []
    for file in all_participants_tsvs:
        dataset = file.stem
        try:
            participants_tsv = pd.read_csv(file, sep="\t")
            n_rows = len(participants_tsv)
            n_columns = len(participants_tsv.columns)
        except pd.errors.EmptyDataError:
            n_rows, n_columns = 0, 0
        except Exception as err:
            logger.warning(f"{file}: Could not parse file due to error {err}, skipping")
            unparseable_datasets.append(dataset)
            continue
        all_datasets_overview_rows.append(
            {"dataset": dataset, "n_rows": n_rows, "n_columns": n_columns}
        )
    logger.info(
        f"Datasets that could not be parsed: {len(unparseable_datasets)}/{len(all_participants_tsvs)}: "
        f"{', '.join(unparseable_datasets)}"
    )
    all_datasets_overview = pd.DataFrame(all_datasets_overview_rows)
    all_datasets_overview = all_datasets_overview.sort_values(
        by="n_rows", ascending=False
    )

    all_datasets_overview.to_csv(out_file, sep="\t", index=False)


def main():
    write_participants_tsv_overview(
        OPENNEURO_DATA_DIR, RESOURCES_DIR / "data_overview.tsv"
    )


if __name__ == "__main__":
    main()
