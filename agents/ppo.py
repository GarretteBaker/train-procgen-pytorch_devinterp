from .base_agent import BaseAgent
from common.misc_util import adjust_lr, get_n_params
import torch
import torch.optim as optim
import numpy as np
import imageio
from tqdm import tqdm
import wandb
import sys

def create_gif(observations, filename):
    with imageio.get_writer(filename, mode='I') as writer:
        for batch in observations:  # Iterate over each batch in observations
            first_obs = batch[0]  # Select the first observation from the batch
            # Convert to 8-bit RGB format
            first_obs = np.transpose(first_obs, (1, 2, 0))  # Rearrange to (64, 64, 3)
            first_obs = (first_obs * 255).astype(np.uint8)  # Convert to uint8
            writer.append_data(first_obs)

class PPO(BaseAgent):
    def __init__(self,
                 env,
                 policy,
                 logger,
                 storage,
                 device,
                 n_checkpoints,
                 n_steps=128,
                 n_envs=8,
                 epoch=3,
                 mini_batch_per_epoch=8,
                 mini_batch_size=32*8,
                 gamma=0.99,
                 lmbda=0.95,
                 learning_rate=2.5e-4,
                 grad_clip_norm=0.5,
                 eps_clip=0.2,
                 value_coef=0.5,
                 entropy_coef=0.01,
                 normalize_adv=True,
                 normalize_rew=True,
                 use_gae=True,
                 **kwargs):

        super(PPO, self).__init__(env, policy, logger, storage, device, n_checkpoints)

        self.n_steps = n_steps
        self.n_envs = n_envs
        self.epoch = epoch
        self.mini_batch_per_epoch = mini_batch_per_epoch
        self.mini_batch_size = mini_batch_size
        self.gamma = gamma
        self.lmbda = lmbda
        self.learning_rate = learning_rate
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate, eps=1e-5)
        self.grad_clip_norm = grad_clip_norm
        self.eps_clip = eps_clip
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.normalize_adv = normalize_adv
        self.normalize_rew = normalize_rew
        self.use_gae = use_gae

    def predict(self, obs, hidden_state, done):
        with torch.no_grad():
            obs = torch.FloatTensor(obs).to(device=self.device)
            hidden_state = torch.FloatTensor(hidden_state).to(device=self.device)
            mask = torch.FloatTensor(1-done).to(device=self.device)
            dist, value, hidden_state = self.policy(obs, hidden_state, mask)
            act = dist.sample()
            log_prob_act = dist.log_prob(act)

        return act.cpu().numpy(), log_prob_act.cpu().numpy(), value.cpu().numpy(), hidden_state.cpu().numpy()

    def optimize(self):
        pi_loss_list, value_loss_list, entropy_loss_list = [], [], []
        batch_size = self.n_steps * self.n_envs // self.mini_batch_per_epoch
        if batch_size < self.mini_batch_size:
            self.mini_batch_size = batch_size
        grad_accumulation_steps = batch_size / self.mini_batch_size
        grad_accumulation_cnt = 1

        self.policy.train()
        for e in range(self.epoch):
            recurrent = self.policy.is_recurrent()
            generator = self.storage.fetch_train_generator(mini_batch_size=self.mini_batch_size,
                                                           recurrent=recurrent)
            for sample in generator:
                obs_batch, hidden_state_batch, act_batch, done_batch, \
                    old_log_prob_act_batch, old_value_batch, return_batch, adv_batch = sample
                mask_batch = (1-done_batch)
                dist_batch, value_batch, _ = self.policy(obs_batch, hidden_state_batch, mask_batch)

                # Clipped Surrogate Objective
                log_prob_act_batch = dist_batch.log_prob(act_batch)
                ratio = torch.exp(log_prob_act_batch - old_log_prob_act_batch)
                surr1 = ratio * adv_batch
                surr2 = torch.clamp(ratio, 1.0 - self.eps_clip, 1.0 + self.eps_clip) * adv_batch
                pi_loss = -torch.min(surr1, surr2).mean()

                # Clipped Bellman-Error
                clipped_value_batch = old_value_batch + (value_batch - old_value_batch).clamp(-self.eps_clip, self.eps_clip)
                v_surr1 = (value_batch - return_batch).pow(2)
                v_surr2 = (clipped_value_batch - return_batch).pow(2)
                value_loss = 0.5 * torch.max(v_surr1, v_surr2).mean()

                # Policy Entropy
                entropy_loss = dist_batch.entropy().mean()
                loss = pi_loss + self.value_coef * value_loss - self.entropy_coef * entropy_loss
                loss.backward()

                # Let model to handle the large batch-size with small gpu-memory
                if grad_accumulation_cnt % grad_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.grad_clip_norm)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                grad_accumulation_cnt += 1
                pi_loss_list.append(pi_loss.item())
                value_loss_list.append(value_loss.item())
                entropy_loss_list.append(entropy_loss.item())

        summary = {'Loss/pi': np.mean(pi_loss_list),
                   'Loss/v': np.mean(value_loss_list),
                   'Loss/entropy': np.mean(entropy_loss_list)}
        return summary

    def generate_data_loader(self, num_samples):
        assert not self.policy.is_recurrent()
        observations = []
        rewards = []

        obs = self.env.reset()
        hidden_state = np.zeros((self.n_envs, self.storage.hidden_state_size))
        done = np.zeros(self.n_envs)
        episode_begin = 0
        for i in range(num_samples//self.n_envs):
            act, _, _, _ = self.predict(obs, hidden_state, done)
            next_obs, rew, done, _ = self.env.step(act)

            observations.append(next_obs)
            rewards.append(rew) 
            obs = next_obs
            
            if np.any(done):
                # propagate the reward backward to calculate the expected discounted reward for the episode
                # Each reward (rew) and observation (obs) in rewards and observations is a numpy array of batches
                # one of the batches succeeds, the others fail. We need to propagate the successful reward backward

                for j in range(i, episode_begin, -1):
                    rewards[j - 1] += self.gamma * rewards[j] * (1 - done[j])

                episode_begin = i+1
                # exit for troubleshooting
                # sys.exit()
        obs_tensor = torch.FloatTensor(np.array(observations)).to(device=self.device).squeeze(0)
        rew_tensor = torch.FloatTensor(np.array(rewards)).to(device=self.device).squeeze(0)
        # print(f"Observation tensor size: {obs_tensor.size()}")
        # print(f"Reward tensor size: {rew_tensor.size()}")

        dataset = torch.utils.data.TensorDataset(obs_tensor, rew_tensor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.mini_batch_size)

        return dataloader

    def train(self, num_timesteps, num_checkpoints):
        wandb.init(project="procgen")
        save_every = num_timesteps // num_checkpoints
        checkpoint_cnt = 0
        obs = self.env.reset()
        hidden_state = np.zeros((self.n_envs, self.storage.hidden_state_size))
        done = np.zeros(self.n_envs)
        observations_for_gif = []  # List to store observations for GIF
        video_frames = []

        with tqdm(total=num_timesteps, desc="Training Progress") as pbar:  # Initialize tqdm progress bar
            while self.t < num_timesteps:
                self.policy.eval()
                for _ in range(self.n_steps):
                    act, log_prob_act, value, next_hidden_state = self.predict(obs, hidden_state, done)
                    next_obs, rew, done, info = self.env.step(act)
                    self.storage.store(obs, hidden_state, act, rew, done, info, log_prob_act, value)

                    if self.t >= num_timesteps - self.n_steps * self.n_envs:
                        observations_for_gif.append(next_obs)  # Save observation for the final episode
                    video_frames.append(next_obs)
                    obs = next_obs
                    hidden_state = next_hidden_state

                _, _, last_val, hidden_state = self.predict(obs, hidden_state, done)
                self.storage.store_last(obs, hidden_state, last_val)
                self.storage.compute_estimates(self.gamma, self.lmbda, self.use_gae, self.normalize_adv)
                summary = self.optimize()
                self.t += self.n_steps * self.n_envs
                pbar.update(self.n_steps * self.n_envs)  # Update the progress bar
                rew_batch, done_batch = self.storage.fetch_log_data()
                self.logger.feed(rew_batch, done_batch)
                self.logger.write_summary(summary)
                wandb.log({"Reward": rew_batch.mean(), "Episode": self.t})
                # self.logger.dump()
                self.optimizer = adjust_lr(self.optimizer, self.learning_rate, self.t, num_timesteps)

                if len(video_frames) > 1000:
                    video_frames = video_frames[-1000:]

                # Save checkpoints at regular intervals
                if self.t >= ((checkpoint_cnt + 1) * save_every) and checkpoint_cnt < num_checkpoints:
                    # save video as gif
                    create_gif(video_frames, self.logger.logdir + f'/episode_{self.t}.gif')
                    # upload gif to wandb
                    wandb.log({"Video": wandb.Video(self.logger.logdir + f'/episode_{self.t}.gif', fps=4, format="gif")})
                    checkpoint_path = f"{self.logger.logdir}/model_{self.t}.pth"
                    torch.save({'state_dict': self.policy.state_dict()}, checkpoint_path)
                    wandb.save(checkpoint_path)  # Save checkpoint to wandb
                    checkpoint_cnt += 1

        create_gif(observations_for_gif, self.logger.logdir + '/final_episode.gif')
        print(f"Saved GIF to {self.logger.logdir + '/final_episode.gif'}")
        self.env.close()


    # def train(self, num_timesteps):
    #     save_every = num_timesteps // self.num_checkpoints
    #     checkpoint_cnt = 0
    #     obs = self.env.reset()
    #     hidden_state = np.zeros((self.n_envs, self.storage.hidden_state_size))
    #     done = np.zeros(self.n_envs)

    #     while self.t < num_timesteps:
    #         # Run Policy
    #         self.policy.eval()
    #         for _ in range(self.n_steps):
    #             act, log_prob_act, value, next_hidden_state = self.predict(obs, hidden_state, done)
    #             next_obs, rew, done, info = self.env.step(act)
    #             self.storage.store(obs, hidden_state, act, rew, done, info, log_prob_act, value)
    #             obs = next_obs
    #             hidden_state = next_hidden_state
    #         _, _, last_val, hidden_state = self.predict(obs, hidden_state, done)
    #         self.storage.store_last(obs, hidden_state, last_val)
    #         # Compute advantage estimates
    #         self.storage.compute_estimates(self.gamma, self.lmbda, self.use_gae, self.normalize_adv)

    #         # Optimize policy & valueq
    #         summary = self.optimize()
    #         # Log the training-procedure
    #         self.t += self.n_steps * self.n_envs
    #         rew_batch, done_batch = self.storage.fetch_log_data()
    #         self.logger.feed(rew_batch, done_batch)
    #         self.logger.write_summary(summary)
    #         self.logger.dump()
    #         self.optimizer = adjust_lr(self.optimizer, self.learning_rate, self.t, num_timesteps)
    #         # Save the model
    #         if self.t > ((checkpoint_cnt+1) * save_every):
    #             torch.save({'state_dict': self.policy.state_dict()}, self.logger.logdir +
    #                        '/model_' + str(self.t) + '.pth')
    #             checkpoint_cnt += 1
    #     self.env.close()
