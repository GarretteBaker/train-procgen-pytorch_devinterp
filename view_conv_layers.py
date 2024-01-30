import torch
import os
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

for artifact_number in tqdm(range(8000)):
    # skip artifact 6697 because it is corrupted
    if artifact_number == 6697:
        continue
    # if fig already exists, skip
    if os.path.exists(f"filters/{artifact_number}.png"):
        continue
    artifacts_files = os.listdir("models")
    model_file = f"models/{artifacts_files[artifact_number]}/{artifacts_files[artifact_number]}"
    loaded_checkpoint = torch.load(model_file)
    conv_layer = loaded_checkpoint['state_dict']['embedder.block1.conv.weight']
    
    # Normalize the filters
    min_val = torch.min(conv_layer)
    range_val = torch.max(conv_layer) - min_val
    normalized_filters = (conv_layer - min_val) / range_val

    # Plot the filters
    fig, axes = plt.subplots(4, 4, figsize=(10,10)) # Adjust the subplot grid as needed
    for i, ax in enumerate(axes.flat):
        filter = normalized_filters[i].cpu().numpy()
        filter = np.transpose(filter, (1, 2, 0))  # Rearrange the dimensions to (H, W, C)
        ax.imshow(filter)
        ax.axis('off')
    os.makedirs("filters", exist_ok=True)
    plt.savefig(f"filters/{artifact_number}.png")
    plt.close()