import lambdahat_helpers as lah
import time
import os
import torch
import numpy as np
import pickle
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # or ":16:8" for the alternative configuration
parser = argparse.ArgumentParser()
parser.add_argument("--device", type=str, default="cpu")
args = parser.parse_args()

device = torch.device(f"{args.device}" if torch.cuda.is_available() else "cpu")

artifact_no = 8000
llcs = []
results = []
os.makedirs("variance_data", exist_ok=True)
timestamp = time.time()
dataloader, dataset, value_network = lah.get_artifact_network_and_data(
    artifact_number = artifact_no, 
    datapoints = 4000, 
    batch_size = 1000, 
    download=False, 
    shuffle=True
)
print(f"Optimizing value network {artifact_no}")


torch.manual_seed(1)
np.random.seed(1)
# value_network = lah.optimize_value_network(value_network, dataloader, epochs=200)

for i in tqdm(range(2)):
    epsilon = 9.1818e-7
    gamma = 94000
    num_chains = 1
    num_draws = 100
    # num_chains = 20
    # num_draws = 2000
    llc_estimator = lah.OnlineLLCEstimator(num_chains, num_draws, len(dataset), device=device)
    grad_norm = lah.GradientNorm(num_chains, num_draws, device=device)
    callbacks = [llc_estimator, grad_norm]
    torch.manual_seed(1)
    np.random.seed(1)
    torch.cuda.manual_seed(1)
    torch.use_deterministic_algorithms(True)
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
    with open(f"variance_data/result_{timestamp}_{artifact_no}.pkl", "wb") as f:
        pickle.dump(results, f)

    # check that the seeds result in the same llc
    if i > 0:
        same_llc = llcs[-1] == llcs[-2]
        print(f"Same llc: {same_llc}")

for n in range(10):
    print(llcs[0] - llcs[1])