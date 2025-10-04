from huggingface_hub import HfApi
import os

api = HfApi(token=os.getenv("HF_TOKEN"))

# Define huggigface repo
repo_id = "sumansaha1980/Tourism_Package"
repo_type = "space"

# Upload to Hugging Face
api.upload_folder(
    folder_path="tourism_project/deployment",     # the local folder containing your files
    repo_id=repo_id,                              # the target repo
    repo_type=repo_type,                          # dataset, model, or space
    path_in_repo="",                              # optional: subfolder path inside the repo
)
