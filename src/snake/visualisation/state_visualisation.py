import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def visualize_game_state(state):
    if len(state) != 100:
        raise ValueError("State list must have a length of 100.")

    # Map the state list to a 10x10 grid
    grid = np.array(state).reshape(10, 10)

    # Define colors for each state
    cmap = ListedColormap(
        ["white", "green", "blue", "red"]
    )  # Corresponds to 0, 1, 2, 3

    # Plot the grid
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(grid, cmap=cmap, aspect="equal")

    # Set up the gridlines and labels
    ax.set_xticks(np.arange(-0.5, 10, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 10, 1), minor=True)
    ax.grid(which="minor", color="black", linestyle="-", linewidth=0.5)
    ax.tick_params(
        which="both", bottom=False, left=False, labelbottom=False, labelleft=False
    )

    # Annotate cells (optional, for debugging)
    for i in range(10):
        for j in range(10):
            ax.text(
                j,
                i,
                int(grid[i, j]),
                color="black",
                ha="center",
                va="center",
                fontsize=10,
            )

    plt.show()
