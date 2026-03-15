"""
Fetch participants.json files from all non-empty dataset repositories in the OpenNeuroDatasets organization.

NOTE: This script does not check which datasets' participants.tsv files have already been fetched.
"""

import base64
import logging
import os
from pathlib import Path

from github import Auth, GithubIntegration
from github.GithubException import GithubException, UnknownObjectException
from tqdm import tqdm

APP_ID = os.environ.get("NB_BOT_ID")
APP_PRIVATE_KEY_PATH = os.environ.get("NB_BOT_KEY_PATH")

JSONLDS_ORG = "OpenNeuroDatasets-JSONLD"
DATASETS_ORG = "OpenNeuroDatasets"
DATA_DIR = Path(__file__).parents[1] / "data"

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


def get_app_private_key():
    """Read the private key for the GitHub app to authenticate as from a file."""
    # Load private key from file to avoid newline issues when a multiline key is set in .env
    with open(APP_PRIVATE_KEY_PATH, "r") as f:
        return f.read()


def gh_authenticate_as_app():
    auth = Auth.AppAuth(APP_ID, get_app_private_key())
    gi = GithubIntegration(auth=auth)

    # Get the installation ID for the Neurobagel Bot app (for the OpenNeuroDatasets-JSONLD organization)
    installation = gi.get_org_installation(JSONLDS_ORG)
    installation_id = installation.id

    g = gi.get_github_for_installation(installation_id)
    return g


def get_nonempty_dataset_repos(repos: list) -> list:
    nonempty_repos = []
    for repo in tqdm(
        repos, total=repos.totalCount, desc="Getting non-empty dataset repos"
    ):
        if repo.name.startswith("ds"):
            if repo.size == 0:  # size is in KB; 0 means empty
                continue
            nonempty_repos.append(repo)
    return nonempty_repos


def main():
    g = gh_authenticate_as_app()
    org = g.get_organization(DATASETS_ORG)

    logger.info(f"Fetching dataset repositories from {DATASETS_ORG}...")
    all_repos = org.get_repos()
    dataset_repos = get_nonempty_dataset_repos(all_repos)
    logger.info(
        f"Non-empty dataset repos: {len(dataset_repos)} / {all_repos.totalCount}"
    )

    datasets_missing_participants_json = []
    for repo in tqdm(dataset_repos, desc="Fetching participants.json files"):
        if (DATA_DIR / f"{repo.name}.json").exists():
            continue
        try:
            file = repo.get_contents("participants.json")
            try:
                content = file.decoded_content
            except AssertionError:
                # File too large for contents API, fetch via blob SHA instead
                blob = repo.get_git_blob(file.sha)
                if blob.encoding == "base64":
                    content = base64.b64decode(blob.content)
                else:
                    content = blob.content.encode()
            except Exception as err:
                logger.warning(
                    f"{repo.name}: unexpected error parsing participants.json: {err}"
                )
            (DATA_DIR / f"{repo.name}.json").write_bytes(content)
        except UnknownObjectException:
            datasets_missing_participants_json.append(repo.name)
        except GithubException as err:
            logger.warning(f"{repo.name}: error fetching participants.json: {err}")

    logger.info(
        f"Datasets missing participants.json: {len(datasets_missing_participants_json)} / {len(dataset_repos)}"
    )


if __name__ == "__main__":
    main()
