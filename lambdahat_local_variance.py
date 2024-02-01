import lambdahat_helpers as lah
import time
import os
import torch
import numpy as np
import pickle
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
# os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # or ":16:8" for the alternative configuration
# torch.use_deterministic_algorithms(True)

parser = argparse.ArgumentParser()
parser.add_argument("--device", type=str, default="cpu")
args = parser.parse_args()

device = torch.device(f"{args.device}" if torch.cuda.is_available() else "cpu")

artifact_start = 6000
num_artifacts = 10
llcs = []
results = []

for artifact_number in tqdm(range(artifact_start, artifact_start + num_artifacts)):
    os.makedirs("variance_data", exist_ok=True)
    timestamp = time.time()
    dataloader, dataset, value_network = lah.get_artifact_network_and_data(
        artifact_number = artifact_number, 
        datapoints = 400, 
        batch_size = 100, 
        download=False, 
        shuffle=True
    )
    print(f"Optimizing value network {artifact_number}")

    value_network = lah.optimize_value_network(value_network, dataloader, epochs=200)

    epsilon = 9.1818e-7
    gamma = 94000
    num_chains = 20
    num_draws = 700
    llc_estimator = lah.OnlineLLCEstimator(num_chains, num_draws, len(dataset), device=device)
    grad_norm = lah.GradientNorm(num_chains, num_draws, device=device)
    callbacks = [llc_estimator, grad_norm]
    torch.manual_seed(1)
    np.random.seed(1)
    torch.cuda.manual_seed(1)
    result = lah.run_callbacks(
        model=value_network,
        epsilon=epsilon,
        gamma=gamma,
        dataloader=dataloader,
        num_chains=num_chains,
        num_draws=num_draws,
        dataset=dataset,
        callbacks=callbacks,
        device=device
    )
    llcs.append(result['llc/means'][-1])
    results.append(result)

    with open(f"variance_data/{artifact_start}_{artifact_start+num_artifacts}_{timestamp}.pkl", "wb") as f:
        pickle.dump(result, f)

print(f"Spread of llcs: {max(llcs) - min(llcs)}")
plt.plot(llcs)
plt.savefig(f"variance_data/{artifact_start}_{artifact_start+num_artifacts}_{timestamp}_llcs.png")
plt.close()