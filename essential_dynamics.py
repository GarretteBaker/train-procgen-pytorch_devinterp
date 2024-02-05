import lambdahat_helpers as lah
import time
import os
import torch
import numpy as np
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--device", type=int, default="0")
args = parser.parse_args()

device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")

model_start = 0
model_end = 8000

dataloader, dataset, _ = lah.get_artifact_network_and_data(
    model_end, 
    datapoints = 2000, 
    batch_size = 1, 
    download = False, 
    device = device, 
    shuffle = True, 
    generate_dataset = True
)

if args.device == 0:
    model_start = 0
    model_end = model_end//4
elif args.device == 1:
    model_start = model_end//4
    model_end = model_end//2
elif args.device == 2:
    model_start = model_end//2
    model_end = model_end//4*3
elif args.device == 3:
    model_start = model_end//4*3
    model_end = model_end

outs = list()

os.makedirs("essential_dynamics/output_data", exist_ok=True)

for modelno in tqdm(range(model_start, model_end)):
    try:
        policy = lah.get_artifact_policy(
            modelno, 
            download = False, 
            device = device
        )
    except:
        continue
    policy.to(device)
    for batch_idx, batch in enumerate(dataloader):
        obs, rew = batch
        obs = obs.to(device)
        out, value = policy.forward_no_categorical(obs)
        break
    out = out.flatten()
    outs.append(out.cpu().detach().numpy())
    with open(f"essential_dynamics/output_data/{model_start}-{model_end}.pkl", "wb") as f:
        pickle.dump(outs, f)
    
