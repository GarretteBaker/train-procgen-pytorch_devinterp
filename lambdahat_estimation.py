import wandb
from tqdm import tqdm
import torch
import os
import lambdahat_helpers as lah
import argparse
import time

# Set up argument parsing
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='gpu number')
args = parser.parse_args()
device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

# Initial parameters (will be dynamically adjusted)
epsilon = 9.1818e-07
gamma = 94000.0

# Other settings
num_chains = 5
num_draws = 700
datapoints = 4000
batch_size = 1000
temperature = "adaptive"
noise_level = 1.0
num_burnin_steps = 0
num_steps_bw_draws = 1
num_epochs = 200
max_artifacts = 8000

# Determine artifact range based on GPU
if args.gpu == 0:
    artifact_start = 0
    artifact_end = 2000
elif args.gpu == 1:
    artifact_start = 2000
    artifact_end = 4000
elif args.gpu == 2:
    artifact_start = 4000
    artifact_end = 6000
elif args.gpu == 3:
    artifact_start = 6000
    artifact_end = 8000

# Initialize wandb
wandb.init(project="procgen-maze-aisc-lambdahat-estimation", name=f"artifact-{artifact_start}-{artifact_end}")

# Loop over artifacts
for artifact_number in tqdm(range(artifact_start, artifact_end)):
    # Dynamically adjust epsilon and gamma based on the current artifact_number
    epsilon, gamma = lah.interpolate_parameters(artifact_number)

    # Proceed with data loading, network optimization, and estimations
    torch.manual_seed(1)
    dataloader, dataset, value_network = lah.get_artifact_network_and_data(
        artifact_number=artifact_number, 
        datapoints=datapoints, 
        batch_size=batch_size, 
        download=False
    )

    value_network = lah.optimize_value_network(value_network, dataloader, epochs=num_epochs, wandbinit=False)

    llc_estimator = lah.LLCEstimator(num_chains, num_draws, len(dataset), device=device)
    grad_norms = lah.GradientNorm(num_chains, num_draws, len(dataset), device=device)
    callbacks = [llc_estimator, grad_norms]
    torch.manual_seed(1)
    results = lah.run_callbacks(
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
    
    # Log results to wandb
    trace = results['loss/trace']
    llc = results['llc/mean']
    llc_std = results['llc/std']
    grad_norm = results['gradient_norm/trace'].mean()
    time_log = str(time.time())
    lah.plot_trace_and_save(
        trace=trace,
        y_axis='L_n(w)',
        name=f"trace_plot_{time_log}.png",
        x_axis="step", 
        title="Learning Coefficient Trace",
        plot_mean=False,
        plot_std=False,
        fig_size=(12, 9),
        true_lc=None
    )

    wandb.log({
        'artifact_number': artifact_number,
        'epsilon': epsilon,
        'gamma': gamma,
        'llc': llc,
        'llc_std': llc_std,
        'learning_coeff_stats': results, 
        'mean grad_norm': grad_norm,
        "trace plot": wandb.Image(f"trace_plot_{time_log}.png")
    })
    
    os.remove(f"trace_plot_{time_log}.png")

wandb.finish()
