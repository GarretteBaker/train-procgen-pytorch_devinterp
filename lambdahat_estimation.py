#%%
import wandb
from tqdm import tqdm
import torch
import os
import shutil
import matplotlib.pyplot as plt
from devinterp.slt import estimate_learning_coeff, estimate_learning_coeff_with_summary
from devinterp.optim import SGLD
from common.env.procgen_wrappers import *
from common.logger import Logger
from common.storage import Storage
from common.model import NatureModel, ImpalaModel
from common.policy import CategoricalPolicy, CategoricalValueNetwork
from common import set_global_seeds, set_global_log_levels

import os, time, yaml, argparse
import gym
from procgen import ProcgenEnv
import random
import torch
from agents.ppo import PPO as AGENT
import random
import yaml
from devinterp.utils import plot_trace
import pickle
from matplotlib.collections import PatchCollection
import matplotlib.ticker as ticker

from devinterp.slt import OnlineLLCEstimator, sample, validate_callbacks, LLCEstimator
from devinterp.slt.norms import GradientNorm

plt.rcParams["figure.figsize"]=15,12  # note: this cell may need to be re-run after creating a plot to take effect

def plot_trace_and_save(trace, y_axis, name, x_axis='step', title=None, plot_mean=True, plot_std=True, fig_size=(12, 9), true_lc=None):
    num_chains, num_draws = trace.shape
    sgld_step = list(range(num_draws))

    plt.figure(figsize=fig_size)

    if true_lc:
        plt.axhline(y=true_lc, color='r', linestyle='dashed')
    
    # trace
    for i in range(num_chains):
        draws = trace[i]
        plt.plot(sgld_step, draws, linewidth=1, label=f'chain {i}')

    # mean
    if plot_mean:
        mean = np.mean(trace, axis=0)
        plt.plot(sgld_step, mean, color='black', linestyle='--', linewidth=2, label='mean', zorder=3)
    
    # std
    if plot_std:
        std = np.std(trace, axis=0)
        plt.fill_between(sgld_step, mean - std, mean + std, color='gray', alpha=0.3, zorder=2)

    if title is None:
        title = f'{y_axis} values over sampling draws'
    
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))    
    plt.title(title)
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    plt.tight_layout()
    plt.savefig(name)
    plt.show()
    plt.close()

# def estimate_learning_coeff(
#     model: torch.nn.Module,
#     loader: DataLoader,
#     criterion: Callable,
#     sampling_method: Type[torch.optim.Optimizer] = SGLD,
#     optimizer_kwargs: Optional[Dict[str, Union[float, Literal["adaptive"]]]] = None,
#     num_draws: int = 100,
#     num_chains: int = 10,
#     num_burnin_steps: int = 0,
#     num_steps_bw_draws: int = 1,
#     cores: int = 1,
#     seed: Optional[Union[int, List[int]]] = None,
#     device: torch.device = torch.device("cpu"),
#     verbose: bool = True,
#     callbacks: List[Callable] = [],
# ) -> float:

# og run made with:
# python train.py --exp_name hard-run --env_name maze_aisc --param_name easy --num_levels 0 --distribution_mode easy --num_timesteps 200000000 --num_checkpoints 1000

# Excess defaults from train.py for reference:
#     parser.add_argument('--exp_name',         type=str, default = 'test', help='experiment name')
#     parser.add_argument('--env_name',         type=str, default = 'starpilot', help='environment ID')
#     parser.add_argument('--start_level',      type=int, default = int(0), help='start-level for environment')
#     parser.add_argument('--num_levels',       type=int, default = int(0), help='number of training levels for environment')
#     parser.add_argument('--distribution_mode',type=str, default = 'easy', help='distribution mode for environment')
#     parser.add_argument('--param_name',       type=str, default = 'easy-200', help='hyper-parameter ID')
#     parser.add_argument('--device',           type=str, default = 'gpu', required = False, help='whether to use gpu')
#     parser.add_argument('--gpu_device',       type=int, default = int(0), required = False, help = 'visible device in CUDA')
#     parser.add_argument('--num_timesteps',    type=int, default = int(25000000), help = 'number of training timesteps')
#     parser.add_argument('--seed',             type=int, default = random.randint(0,9999), help='Random generator seed')
#     parser.add_argument('--log_level',        type=int, default = int(40), help='[10,20,30,40]')
#     parser.add_argument('--num_checkpoints',  type=int, default = int(1), help='number of checkpoints to store')

def get_model_number(model_name):
    # model is of format model_<number>:v<version>
    return int(model_name.split('_')[1].split(':')[0])

def gradient_single_plot(gradients, param_name: str, color='blue', plot_zero=True, chain: int = None, filename = None):
    grad_dist = gradients.grad_dists[param_name]
    if chain is not None:
        max_count = grad_dist[chain].max()
    else:
        max_count = grad_dist.sum(axis=0).max()

    def get_color_alpha(count):
        if count == 0:
            return torch.tensor(0).to(gradients.device)
        min_alpha = 0.35
        max_alpha = 0.85
        return (count / max_count) * (max_alpha - min_alpha) + min_alpha
    
    def build_rect(count, bin_min, bin_max, draw):
        alpha = get_color_alpha(count)
        pos = (draw, bin_min)
        height = bin_max - bin_min
        width = 1
        return plt.Rectangle(pos, width, height, color=color, alpha=alpha.cpu().numpy().item(), linewidth=0)
    
    _, ax = plt.subplots()
    patches = []
    for draw in range(gradients.num_draws):
        for pos in range(gradients.num_bins):
            bin_min = gradients.min_grad + pos * gradients.bin_size
            bin_max = bin_min + gradients.bin_size
            if chain is None:
                count = grad_dist[:, draw, pos].sum()
            else:
                count = grad_dist[chain, draw, pos]
            if count != 0:
                rect = build_rect(count, bin_min, bin_max, draw)
                patches.append(rect)
    patches = PatchCollection(patches, match_original=True)
    ax.add_collection(patches)

    # note that these y min/max values are relative to *all* gradients, not just the ones for this param
    y_min = gradients.min_grad
    y_max = gradients.max_grad
    # ensure the 0 line is visible
    y_min = y_min if y_min < 0 else -y_max
    y_max = y_max if y_max > 0 else -y_min
    plt.ylim(y_min, y_max)

    plt.xlim(0, gradients.num_draws)
    plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    if plot_zero:
        plt.axhline(color='black', linestyle=':', linewidth=1)

    plt.xlabel('Sampler steps')
    plt.ylabel('gradient distribution')
    plt.title(f'Distribution of {param_name} gradients at each sampler step')

    if filename is not None:
        plt.savefig(filename)
    else:
        plt.show()

def get_artifact_network_and_data(artifact_number, datapoints=100, batch_size=100):
    artifacts = run.logged_artifacts()

    artifact = artifacts[artifact_number]
    artifact_to_download = api.artifact(f"{project_name}/{artifact.name}", type="model")
    artifact_dir = artifact_to_download.download()
    # artifact_dir = "artifacts/model_160022528:v0"
    model_file = f"{artifact_dir}/{artifact.name[:-3]}.pth"

    hidden_state_dim = 0
    observation_space = env.observation_space
    observation_shape = observation_space.shape
    storage = Storage(observation_shape, hidden_state_dim, n_steps, n_envs, device)

    loaded_checkpoint = torch.load(model_file)
    model = ImpalaModel(in_channels = observation_shape[0])
    policy = CategoricalPolicy(model, False, env.action_space.n)
    if "state_dict" in loaded_checkpoint:
        policy.load_state_dict(loaded_checkpoint['state_dict'])
    elif "model_state_dict" in loaded_checkpoint:
        policy.load_state_dict(loaded_checkpoint['model_state_dict'])
    policy.to(device)
    agent = AGENT(env, policy, logger, storage, device, num_checkpoints, **hyperparameters)

    dataloader, dataset = agent.generate_data_loader(datapoints, batch_size)
    value_network = CategoricalValueNetwork(model, False)
    if "state_dict" in loaded_checkpoint:
        loaded_checkpoint['state_dict'] = {k: v for k,v in loaded_checkpoint['state_dict'].items() if "policy" not in k}
        value_network.load_state_dict(loaded_checkpoint['state_dict'])
    elif "model_state_dict" in loaded_checkpoint:
        loaded_checkpoint['model_state_dict'] = {k: v for k,v in loaded_checkpoint['model_state_dict'].items() if "policy" not in k}
        value_network.load_state_dict(loaded_checkpoint['model_state_dict'])
    # delete saved policy
    os.remove(model_file)
    return dataloader, dataset, value_network

def optimize_value_network(value_network, dataloader, epochs=50, lr=1e-5):
    value_network.to(device)
    optimizer = torch.optim.Adam(value_network.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()

    epochs_time_series = []
    batches = []
    value_losses = []
    grad_norms = []
    grad_means = []
    grad_stds = []
    gradient_hists = []

    for epoch in tqdm(range(epochs)):
        for batch_idx, batch in enumerate(dataloader):
            optimizer.zero_grad()
            observations, returns = batch
            observations = observations.to(device)
            returns = returns.to(device)
            values_pred = value_network(observations)
            value_loss = criterion(values_pred, returns)

            value_loss.backward()
            optimizer.step()

            # Calculate gradients statistics
            grad_norm = torch.norm(torch.stack([torch.norm(p.grad.detach()) for p in value_network.parameters() if p.grad is not None]))
            grad_mean = torch.mean(torch.stack([torch.mean(p.grad.detach()) for p in value_network.parameters() if p.grad is not None]))
            grad_std = torch.std(torch.stack([torch.std(p.grad.detach()) for p in value_network.parameters() if p.grad is not None]))

            epochs_time_series.append(epoch)
            batches.append(batch_idx)
            value_losses.append(value_loss.item())
            grad_norms.append(grad_norm.item())
            grad_means.append(grad_mean.item())
            grad_stds.append(grad_std.item())
            gradient_hists.append({name: p.grad.cpu().numpy() for name, p in value_network.named_parameters() if p.grad is not None})

    wandb.log({
        "epochs": epochs_time_series,
        "batches": batches,
        'value_loss': value_losses,
        'grad_norm': grad_norms,
        'grad_mean': grad_means,
        'grad_std': grad_stds,
    })

    return value_network    

def run_callbacks(model, epsilon, gamma, dataloader, num_chains, num_draws, dataset, callbacks, device, criterion):
    assert validate_callbacks(callbacks)
    optim_kwargs = {
        "lr": epsilon,
        "elasticity": gamma,
        "temperature": "adaptive",
        "num_samples": len(dataset), 
        "save_noise": True
    }
    
    if callbacks is None:
        llc_estimator = OnlineLLCEstimator(num_chains, num_draws, len(dataset), device=device)
        callbacks = [llc_estimator]

    sample(
        model=model, 
        loader=dataloader, 
        criterion = criterion, 
        optimizer_kwargs = optim_kwargs,
        sampling_method = SGLD, 
        num_chains = num_chains,
        num_draws = num_draws, 
        callbacks = callbacks, 
        device=device
    )
    
    results = {}

    for callback in callbacks:
        if hasattr(callback, "sample"):
            results.update(callback.sample())
    return results

#%%
# Set your specific run ID here
run_id = "jp9tjfzd"
project_name = "procgen"

# Initialize wandb API
api = wandb.Api()

# Fetch the run
run = api.run(f"{project_name}/{run_id}")

torch.manual_seed(1)

####################
## HYPERPARAMETERS #
#################### 
param_name = 'easy'
gpu_device = int(0)
env_name = "maze_aisc"
start_level = 0
num_levels = 0
distribution_mode = "easy"
exp_name = "hard-run"
seed = random.randint(0,9999)
num_checkpoints = 0

print('[LOADING HYPERPARAMETERS...]')
with open('hyperparams/procgen/config.yml', 'r') as f:
    hyperparameters = yaml.safe_load(f)[param_name]
for key, value in hyperparameters.items():
    print(key, ':', value)

############
## DEVICE ##
############
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_device)
device = torch.device('cuda')

#################
## ENVIRONMENT ##
#################
print('INITIALIZAING ENVIRONMENTS...')
n_steps = hyperparameters.get('n_steps', 256)
n_envs = hyperparameters.get('n_envs', 64)
# By default, pytorch utilizes multi-threaded cpu
# Procgen is able to handle thousand of steps on a single core
torch.set_num_threads(1)
env = ProcgenEnv(num_envs=n_envs,
                    env_name=env_name,
                    start_level=start_level,
                    num_levels=num_levels,
                    distribution_mode=distribution_mode, 
                    rand_region = 5) 
normalize_rew = hyperparameters.get('normalize_rew', True)
env = VecExtractDictObs(env, "rgb")
if normalize_rew:
    env = VecNormalize(env, ob=False) # normalizing returns, but not the img frames.
env = TransposeFrame(env)
env = ScaledFloatFrame(env)

############
## LOGGER ##
############
print('INITIALIZAING LOGGER...')
logdir = 'procgen/' + env_name + '/' + exp_name + '/' + 'seed' + '_' + \
            str(seed) + '_' + time.strftime("%d-%m-%Y_%H-%M-%S")
logdir = os.path.join('logs', logdir)
if not (os.path.exists(logdir)):
    os.makedirs(logdir)
logger = Logger(n_envs, logdir)

wandb.init(project="procgen", name="procgen-lambdahat-estimation")
criterion = torch.nn.MSELoss()
epsilon = 1e-6
gamma = 1e5
num_chains = 3
num_draws = 25
datapoints = 4000
batch_size = 1000
temperature = "adaptive"
noise_level = 1.0
num_burnin_steps = 0
num_steps_bw_draws = 1
num_epochs = 100
skip_every = 8*15
wandb.log({
    'epsilon': epsilon,
    'gamma': gamma,
    'num_chains': num_chains,
    'num_draws': num_draws,
    'datapoints': datapoints,
    'batch_size': batch_size,
    'temperature': temperature,
    'noise_level': noise_level,
    'num_burnin_steps': num_burnin_steps,
    'num_steps_bw_draws': num_steps_bw_draws, 
    'num_epochs': num_epochs
})

for artifact_number in tqdm(range(0, 8000, skip_every)):
    dataloader, dataset, value_network = get_artifact_network_and_data(
        artifact_number = artifact_number, 
        datapoints = datapoints, 
        batch_size = batch_size
    )

    value_network = optimize_value_network(value_network, dataloader, epochs=num_epochs)
    torch.save(value_network.state_dict(), 'value_network_local_min.pth')

    llc_estimator = LLCEstimator(num_chains, num_draws, len(dataset), device=device)
    grad_norms = GradientNorm(num_chains, num_draws, len(dataset), device=device)
    callbacks = [llc_estimator, grad_norms]
    results = run_callbacks(
        model = value_network,
        epsilon = epsilon,
        gamma = gamma,
        dataloader = dataloader,
        num_chains = num_chains,
        num_draws = num_draws,
        dataset = dataset,
        callbacks = callbacks,
        device = device,
        criterion = criterion
    )    
    trace = results['loss/trace']
    llc = results['llc/mean']
    llc_std = results['llc/std']
    grad_norm = results['gradient_norm/trace'].mean()

    plot_trace_and_save(
        trace = trace,
        y_axis = 'L_n(w)',
        name = "trace_plot.png",
        x_axis = "step", 
        title = " Learning Coefficient Trace",
        plot_mean=False,
        plot_std=False,
        fig_size=(12, 9),
        true_lc = None
    )
    # delete value network
    os.remove('value_network_local_min.pth')

    wandb.log({
        'artifact_number': artifact_number,
        'llc': llc,
        'llc_std': llc_std, 
        'learning_coeff_stats': results, 
        'mean grad_norm': grad_norm,
        "trace plot": wandb.Image("trace_plot.png")
    })

wandb.finish()