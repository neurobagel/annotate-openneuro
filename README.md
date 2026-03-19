# annotate-openneuro
Scripts and data for bulk-generation of OpenNeuro data dictionaries.

## Get annotatable data

### Fetch all `participants.tsv` files (WIP)

1. Clone and install from source [`ondiagnostics`](https://github.com/OpenNeuroOrg/ondiagnostics):

2. Run the following script:

    ```python
    import asyncio
    from ondiagnostics.graphql import create_client, datasets_generator
    from ondiagnostics.graphql import Dataset
    from ondiagnostics.tasks import clone_dataset
    from pathlib import Path

    PUT_HERE = Path("/home/surchs/Repositories/external/ondiagnostics/seb_fun_things/data")
    client = create_client()


    async def main():
        async for dataset in datasets_generator(client):
            print("getting", dataset.id, "now")
            await clone_dataset(Dataset(id=dataset.id, tag=dataset.tag), cache_dir=PUT_HERE)


    if __name__ == "__main__":
        asyncio.run(main())%
    ```

3. Create files from git blob:

    ```bash
    for dir in /home/surchs/Repositories/external/ondiagnostics/seb_fun_things/data/*; do
        dataset_name=$(basename "$dir" .git)
        if [ ! -f "./${dataset_name}.tsv" ]; then
            git -C $dir show HEAD:participants.tsv > "./${dataset_name}.tsv"
        fi
    done
    ```

    NOTE: This script creates a TSV file for all datasets regardless of if the file exists or not in the repo.

### Fetch all `participants.json` files

> [!NOTE]
> This script is agnostic to whether participants.tsv files have already been fetched.

1. Create a file containing a private key for the Neurobagel Bot app
2. Set two environment variables
- `NB_BOT_ID`: app ID of the Neurobagel Bot app, which you can find on the settings page for the app
- `NB_BOT_KEY_PATH`: path to the private key file on your machine

3. Run the script to get all `participants.json` files:
    ```bash
    python run code/get_participants_json_files.py
    ```

### Create overview of `participants.tsv` files

```bash
python code/create_data_overview.py
```

This will create a file called `data_overview.tsv`.


### Determine datasets needed to cover ~50% of all participants

```bash
python code/get_openneuro_tabular_overview.py
```

### Generate summary mega-tables of all columns and categorical values in all `participants.tsv` files

```bash
python code/generate_participants_tsv_contents_table.py
```

This script will also generate a version of the column summaries table with heuristic-based first guesses of standardized variable mappings and age column formats,
and a version of the value summaries table with heuristic-based first guesses of standardized term mappings for sex column values.
