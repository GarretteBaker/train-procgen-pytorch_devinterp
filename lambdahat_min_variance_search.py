import lambdahat_helpers as lah
import os
import time
import torch
import numpy as np
import pickle
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--device", type=int, default=0)
args = parser.parse_args()

device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")

if args.device == 0:
    artifact_no = 2000
elif args.device == 1:
    artifact_no = 4000
elif args.device == 2:
    artifact_no = 6000
elif args.device == 3:
    artifact_no = 8000
results = {}
timestamp = time.time()
dataloader, dataset, value_network = lah.get_artifact_network_and_data(
    artifact_number = artifact_no, 
    datapoints = 4000, 
    batch_size = 1000, 
    download=False
)
print(f"Optimizing value network {artifact_no}")

torch.manual_seed(1)
np.random.seed(1)
value_network = lah.optimize_value_network(value_network, dataloader, epochs=200)

# epsilons should be around 1e-6
# gammas around 1e5

epsilons = np.linspace(1e-6 - 1e-7, 1e-6 + 1e-7, 12)
gammas = np.linspace(1e5 - 1e4, 1e5 + 1e4, 16)
num_chains = 5 # making these too high so that we can vary them in case weird epsilons or gammas are found
num_draws = 700
os.makedirs(f"variance_data/{timestamp}", exist_ok=True)

total_iterations = len(epsilons) * len(gammas)
iteration_count = 0
start_time = time.time()

for epsilon in epsilons:
    for gamma in gammas:
        iteration_count += 1
        iter_start_time = time.time()
        llc_estimator = lah.OnlineLLCEstimator(num_chains, num_draws, len(dataset), device=device)
        grad_norm = lah.GradientNorm(num_chains, num_draws, device=device)
        callbacks = [llc_estimator, grad_norm]
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
        results[(artifact_no, epsilon, gamma)] = result
        with open(f"variance_data/{timestamp}/results_{artifact_no}.pkl", "wb") as f:
            pickle.dump(results, f)

        iter_end_time = time.time()
        iter_duration = iter_end_time - iter_start_time
        average_duration = (iter_end_time - start_time) / iteration_count
        remaining_iterations = total_iterations - iteration_count
        estimated_time_left = average_duration * remaining_iterations

        print(f"Iteration {iteration_count}/{total_iterations} complete. Estimated time remaining: {estimated_time_left:.2f} seconds.")