import matplotlib.pyplot as plt
import numpy as np


def get_performance(model, env, n_episodes=100, seed=None):
    np.random.seed(seed)

    # Simulat e 100 steps in the environment with rendering
    scores = []

    obs, _ = env.reset()  # Reset the environment and get the initial observation
    for i in range(n_episodes):
        episode_done = False
        while not episode_done:
            # Use the model to predict the next action
            action, _states = model.predict(obs, deterministic=True)
            # Take the action in the environment
            obs, reward, done, truncated, info = env.step(action)

            if done or truncated:
                scores.append(info["score"])
                episode_done = True
                obs, _ = env.reset()

    return scores


def produce_plots(metrics):
    # Extract metrics
    avg_rewards = [m["avg_score"] for m in metrics]
    avg_lengths = [m["avg_length"] for m in metrics]
    high_scores = [m["all_time_high_score"] for m in metrics]

    # Create plots
    plt.figure(figsize=(16, 6))

    # Average Reward Plot
    plt.subplot(1, 3, 1)
    plt.plot(avg_rewards, label="Average Reward")
    plt.xlabel("Training Steps")
    plt.ylabel("Reward")
    plt.title("Average Reward Over Time")
    plt.legend()

    # Average Game Length Plot
    plt.subplot(1, 3, 2)
    plt.plot(avg_lengths, label="Average Game Length", color="orange")
    plt.xlabel("Training Steps")
    plt.ylabel("Game Length")
    plt.title("Average Game Length Over Time")
    plt.legend()

    # All-Time High Score Plot
    plt.subplot(1, 3, 3)
    plt.plot(high_scores, label="All-Time High Score", color="green")
    plt.xlabel("Training Steps")
    plt.ylabel("High Score")
    plt.title("All-Time High Score Over Time")
    plt.legend()

    # Adjust layout and display the plots
    plt.tight_layout()
    plt.show()
