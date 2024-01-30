import wandb
import shutil
import os
from tqdm import tqdm

def get_model_number(model_name):
    # model is of format model_<number>:v<version>
    return int(model_name.split('_')[1].split(':')[0])

# Set your specific run ID here
run_id = "jp9tjfzd"
project_name = "procgen"

# Initialize wandb API
api = wandb.Api()

# Fetch the run
run = api.run(f"{project_name}/{run_id}")
artifacts = run.logged_artifacts()

# Define function to determine model filename
modelfile = lambda artifact_name: f"./models/model_{get_model_number(artifact_name)}.pth"

# Ensure the target directory exists
os.makedirs("./models", exist_ok=True)

# Iterate over artifacts
for artifact in tqdm(artifacts):
    # Determine the source and destination paths
    destination_path = modelfile(artifact.name)

    if os.path.exists(destination_path):
        continue

    artifact_to_download = api.artifact(f"{project_name}/{artifact.name}", type="model")
    artifact_dir = artifact_to_download.download()
    
    # Move artifact to the specified model file
    shutil.move(artifact_dir, destination_path)
    print(f"Artifact {artifact.name} moved to {destination_path}")