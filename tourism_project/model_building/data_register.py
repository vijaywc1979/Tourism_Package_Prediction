import os
from huggingface_hub import HfApi, create_repo, upload_folder
from huggingface_hub.utils import RepositoryNotFoundError

repo_id = "vijaywc1979/Tourism-Package-Prediction"
repo_type = "dataset"

HF_TOKEN = os.getenv("HF_TOKEN")
api = HfApi(token=HF_TOKEN)

# Create repo if it doesn't exist
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Dataset repo {repo_id} exists.")
except RepositoryNotFoundError:
    print(f"Creating dataset repo {repo_id}...")
    create_repo(repo_id=repo_id, repo_type=repo_type, private=False)

# Upload the folder
api.upload_folder(
    folder_path="tourism_project/data",
    repo_id=repo_id,
    repo_type=repo_type,
)
print("Dataset uploaded successfully.")
