import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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
    """
    Produce plots of average episode length, average reward and high score during training.
    """


    df = pd.DataFrame(metrics)

    # Create plots
    plt.figure(figsize=(16, 6))

    # Average Reward Plot
    plt.subplot(1, 3, 1)
    plt.plot(df.num_timesteps, df.avg_score, label="Average Score")
    plt.xlabel("Training Steps")
    plt.ylabel("Score")
    plt.title("Average Score Through Training")
    plt.legend()

    # Average Game Length Plot
    plt.subplot(1, 3, 2)
    plt.plot(
        df.num_timesteps, df.avg_length, label="Average Game Length During Training", color="orange"
    )
    plt.xlabel("Training Steps")
    plt.ylabel("Game Length")
    plt.title("Average Game Length Over Time")
    plt.legend()

    # All-Time High Score Plot
    plt.subplot(1, 3, 3)
    plt.plot(
        df.num_timesteps,
        df.all_time_high_score,
        label="High Score",
        color="green",
    )
    plt.xlabel("Training Steps")
    plt.ylabel("High Score")
    plt.title("All-Time High Score During Training")
    plt.legend()

    # Adjust layout and display the plots
    plt.tight_layout()
    plt.show()


def watch_agent_play(model, env, num_timesteps=150):
    """
    Renders the given model interacting with the given environment for the total number of provided timesteps.

    :param model: policy be observed
    :param env: environment agent is interacting with. Using an incompatible env will lead to issues
    :param num_timesteps: number of steps to be observed
    :return: rendering of agent making decisions
    """
    print("Interrupt kernel to stop rendering")

    obs, _ = env.reset()  # Reset the environment and get the initial observation
    for step in range(num_timesteps):
        action, _states = model.predict(obs, deterministic=True)

        obs, reward, done, truncated, info = env.step(action)
        if env._time_since_food_eaten > 15 * len(env._snake):
            # stop watching if agent is going in circles
            truncated = True

        if done or truncated:
            obs, _ = env.reset()

        # Render the environment
        env.render()

