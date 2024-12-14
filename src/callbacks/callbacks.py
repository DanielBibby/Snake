from stable_baselines3.common.callbacks import BaseCallback
import numpy as np


class ScoreLoggerCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(ScoreLoggerCallback, self).__init__(verbose)
        self.episode_scores = []
        self.episode_lengths = []
        self.all_time_high_score = 0
        self.metrics = []

    def _on_step(self) -> bool:
        # Access the environment
        info = self.locals.get("infos", [{}])
        if "episode" in info[-1]:
            episode_score = info[-1]["score"]
            episode_length = info[-1]["episode"]["l"]

            # Log rewards and lengths
            self.episode_scores.append(episode_score)
            self.episode_lengths.append(episode_length)

            # Update all-time high score
            if episode_score > self.all_time_high_score:
                self.all_time_high_score = episode_score

            # Compute metrics
            avg_score = np.mean(self.episode_scores[-100:])  # Last 100 episodes
            avg_length = np.mean(self.episode_lengths[-100:])
            if len(self.episode_lengths) < 2:
                self.metrics.append(
                    {
                        "avg_score": avg_score,
                        "avg_length": avg_length,
                        "all_time_high_score": self.all_time_high_score,
                        "num_timesteps": episode_length,
                    }
                )
            else:
                self.metrics.append(
                    {
                        "avg_score": avg_score,
                        "avg_length": avg_length,
                        "all_time_high_score": self.all_time_high_score,
                        "num_timesteps": episode_length
                        + self.metrics[-1]["num_timesteps"],
                    }
                )

        return True

    def get_metrics(self):
        return self.metrics


class RewardLoggerCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(RewardLoggerCallback, self).__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.all_time_high_score = float("-inf")
        self.metrics = []

    def _on_step(self) -> bool:
        # Access the environment
        info = self.locals.get("infos", [{}])
        if "episode" in info[-1]:
            episode_reward = info[-1]["episode"]["r"]
            episode_length = info[-1]["episode"]["l"]

            # Log rewards and lengths
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_length)

            # Update all-time high score
            if episode_reward > self.all_time_high_score:
                self.all_time_high_score = episode_reward

            # Compute metrics
            avg_reward = np.mean(self.episode_rewards[-100:])  # Last 100 episodes
            avg_length = np.mean(self.episode_lengths[-100:])
            self.metrics.append(
                {
                    "avg_reward": avg_reward,
                    "avg_length": avg_length,
                    "all_time_high_score": self.all_time_high_score,
                }
            )

        return True

    def get_metrics(self):
        return self.metrics
