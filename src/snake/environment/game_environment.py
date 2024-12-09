import gymnasium as gym
from gymnasium.spaces.discrete import Discrete
from gymnasium.spaces import MultiDiscrete
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from IPython.display import display, clear_output


class SnakeEnv(gym.Env):
    def __init__(self):
        super(SnakeEnv, self).__init__()

        self.action_space = Discrete(3)  # {0: 'Left', 1: 'Straight', 2: 'Right'}

        # [head loc x, head loc y, food loc x, food loc y, danger left, danger straight, danger right]
        # for each grid loc there is either head, tail, food or nothing
        self.observation_space = MultiDiscrete(
            [4] * 10 * 10
        )  # {0: 'Nothing', 1: 'food', 2: 'tail', 3: 'head'}

        self.state = [0] * 10 * 10
        self.state[40:43] = [2, 2, 3]
        self.state[47] = 1
        self.done = False

        self._state_lag = None
        self._snake = [40, 41, 42]
        self._time_since_food_eaten = 0
        self.score = 0

    def step(self, action):
        """
        This method applies an action and returns the next state, reward, done, and any additional info.

        :param action: Action taken by the agent
        :return: tuple (next_state, reward, done, info)
        """
        self.done, self.state, food_eaten = self._update_state(action)

        # Give reward only if food is eaten
        if food_eaten:
            reward = 1
            self._time_since_food_eaten = 0
            self.score += 1
        elif self.done:
            reward = -1
        else:
            reward = 0
            self._time_since_food_eaten += 1

        truncated = False
        if self._time_since_food_eaten > 25 * len(self._snake):
            truncated = True

        info = {"score": self.score}

        return self.state, reward, self.done, truncated, info

    def reset(self, seed=None):
        """
        This method resets the environment to its initial state.
        :return: Initial state
        """
        super().reset(seed=seed)

        self.state = [0] * 10 * 10
        self.state[40:43] = [2, 2, 3]
        self.state[47] = 1
        self.done = False

        # set the hidden variables appropriately
        self._state_lag = None
        self._snake = [40, 41, 42]
        self._time_since_food_eaten = 0
        self.score = 0

        return self.state, {}

    def render(self, mode="human"):
        """
        This method renders the environment with a progressive animation of the game.
        """
        # If a figure already exists, do not create a new one; just update the plot
        if not hasattr(self, "fig"):
            self.fig, self.ax = plt.subplots(figsize=(6, 6))

        # Call the helper function to visualize the state
        self.visualize_game_state(self.state)

        if mode == "human":
            # Update the display in the notebook
            clear_output(wait=True)  # Clear previous output to keep it dynamic
            display(self.fig)  # Display the current figure in the notebook
            plt.pause(0.01)  # Adjust for visualization speed

    def visualize_game_state(self, state):
        if len(state) != 100:
            raise ValueError("State list must have a length of 100.")

        # Map the state list to a 10x10 grid
        grid = np.array(state).reshape(10, 10)

        # Define colors for each state
        cmap = ListedColormap(
            ["white", "green", "blue", "red"]
        )  # Corresponds to 0, 1, 2, 3

        # Plot the grid on the existing axes
        self.ax.clear()  # Clear the previous plot to avoid overlaps
        self.ax.imshow(grid, cmap=cmap, aspect="equal")

        # Set up the gridlines and labels
        self.ax.set_xticks(np.arange(-0.5, 10, 1), minor=True)
        self.ax.set_yticks(np.arange(-0.5, 10, 1), minor=True)
        self.ax.grid(which="minor", color="black", linestyle="-", linewidth=0.5)
        self.ax.tick_params(
            which="both", bottom=False, left=False, labelbottom=False, labelleft=False
        )

        # Annotate cells (optional, for debugging)
        for i in range(10):
            for j in range(10):
                self.ax.text(
                    j,
                    i,
                    int(grid[i, j]),
                    color="black",
                    ha="center",
                    va="center",
                    fontsize=10,
                )

        self.fig.canvas.draw()  # Explicitly draw the canvas to update the plot

    def close(self):
        """
        Any cleanup code for the environment can be added here (e.g., closing windows).
        """
        pass

    def _update_state(
        self, action: Discrete(3)
    ) -> (bool, MultiDiscrete([4] * 10 * 10), bool):
        """
        Sse self.state and self._state_lag to figure out the next state
        """

        head = self.state.index(3)
        if self._state_lag:
            head_lag = self._state_lag.index(3)
        else:
            head_lag = head - 1  # snake always start moving right
        food_index = self.state.index(1)

        head_change = head - head_lag
        # head_change = +/- 1 => snake moving right / left
        # head_change = +/- 10 => snake moving down / up

        direction_action_tuple = (head_change, action)
        head_lag = head
        if direction_action_tuple in [(1, 2), (-1, 0), (10, 1)]:
            # head moves down
            head += 10
        elif direction_action_tuple in [(-10, 2), (10, 0), (1, 1)]:
            # head moves right
            head += 1
        elif direction_action_tuple in [(10, 2), (-10, 0), (-1, 1)]:
            # head moves left
            head -= 1
        elif direction_action_tuple in [(-1, 2), (1, 0), (-10, 1)]:
            # head moves up
            head -= 10

        # update the snake positions
        self._snake.append(head)

        food_eaten = False
        if food_index == head:
            food_eaten = True
            food_index = self._generate_new_food()
        else:
            self._snake.pop(0)

        if self._terminal_reached():
            return True, self.state, False

        # create new game state
        new_state = [0] * 10 * 10
        new_state[food_index] = 1
        for i in self._snake[0:-1]:
            new_state[i] = 2
        new_state[head] = 3

        self._state_lag = self.state

        return False, new_state, food_eaten

    def _generate_new_food(self):
        valid_locations = [i for i in range(10 * 10) if i not in self._snake]

        return np.random.choice(valid_locations)

    def _terminal_reached(self) -> bool:
        """
        Method to figure out if an action results in a terminal action
        """
        if self._snake[-2] % 10 == 0 and self._snake[-2] - self._snake[-1] == 1:
            # leftmost wall has been hit
            return True
        elif (self._snake[-2] + 1) % 10 == 0 and self._snake[-1] - self._snake[-2] == 1:
            # rightmost wall has been hit
            return True
        elif self._snake[-1] < 0 or self._snake[-1] >= 100:
            # either upper or lower most wall has been hit
            return True
        elif self._snake[-1] in self._snake[:-1]:
            return True

        return False


class SnakeEnvNewStateRep(SnakeEnv):
    def __init__(self):
        super(SnakeEnvNewStateRep, self).__init__()

    def step(self, action):
        """
        This method applies an action and returns the next state, reward, done, and any additional info.

        :param action: Action taken by the agent
        :return: tuple (next_state, reward, done, info)
        """
        self.state, reward, self.done, truncated, info = super().step(action)

        # perform transformation on the state
        rotated_state = self._transform_state(self.state)

        return rotated_state, reward, self.done, truncated, info

    def _transform_state(self, state):
        """
        Turn the game state into a matrix and rotate it so the snake is always facing up then return it back to a list.

        :param state:
        :return:
        """

        matrix = np.array(state).reshape(10, 10)

        if self._snake[-1] - self._snake[-2] == 1:
            # snake going right
            return list(np.rot90(matrix, 1).flatten())
        elif self._snake[-1] - self._snake[-2] == 10:
            # snake going down
            return list(np.rot90(matrix, 2).flatten())
        elif self._snake[-1] - self._snake[-2] == -1:
            # snake going left
            return np.rot90(matrix, 3).flatten()
        elif self._snake[-1] - self._snake[-2] == -10:
            return list(matrix.flatten())


class SnakeEnvNewReward(SnakeEnv):
    def __init__(self):
        super(SnakeEnvNewReward, self).__init__()

    def step(self, action):
        """
        This method applies an action and returns the next state, reward, done, and any additional info.

        :param action: Action taken by the agent
        :return: tuple (next_state, reward, done, info)
        """
        self.state, reward, self.done, truncated, info = super().step(action)

        if self.done:
            pass
        if reward == 0:
            head = self._snake[-1]
            old_head = self._state_lag.iloc[3]
            head_change = head - old_head
            food_loc = self.state.iloc[1]

            # if head moves towards food give a reward of 0.2
            if head_change == 10:
                if food_loc < head:
                    reward = 0.2
                else:
                    reward = -0.2
            elif head_change == 1:
                if food_loc % 10 > head % 10:
                    reward = 0.2
                else:
                    reward = -0.2
            elif head_change == -1:
                if food_loc % 10 < head % 10:
                    reward = 0.2
                else:
                    reward = -0.2
            elif head_change == 10:
                if food_loc > head:
                    reward = 0.2
                else:
                    reward = -0.2
            else:
                raise Exception("Something has gone wrong with head locations")

        return self.state, reward, self.done, truncated, info


class SnakeEnvRandS(SnakeEnv):
    def __init__(self):
        super(SnakeEnvRandS, self).__init__()

    def step(self, action):
        """
        This method applies an action and returns the next state, reward, done, and any additional info.

        :param action: Action taken by the agent
        :return: tuple (next_state, reward, done, info)
        """

        self.state, reward, self.done, truncated, info = super().step(action)

        if self.done:
            new_reward = -1
        elif truncated:
            new_reward = 0
        elif reward == 1:
            new_reward = 1

        elif reward == 0:
            head = self.state.index(3)
            old_head = self._state_lag.index(3)
            food_loc = self._state_lag.index(1)

            head_food_dist = self._get_distance(head, food_loc)
            old_head_food_dist = self._get_distance(old_head, food_loc)

            if head_food_dist < old_head_food_dist:
                new_reward = 0.2
            elif head_food_dist > old_head_food_dist:
                new_reward = -0.2
            else:
                new_reward = 0
                print(head_food_dist)
                print(old_head_food_dist)
                print("====")
                print(head)
                print(old_head)
                print(food_loc)
                raise Exception("Something has gone wrong.")

        rotated_state = self._transform_state(self.state)

        return rotated_state, new_reward, self.done, truncated, info

    def reset(self, seed=None):
        """
        This method resets the environment to its initial state.
        :return: Initial state
        """
        super().reset(seed=seed)

        self.state, self._state_lag, self._snake = self._create_random_state()
        self.done = False

        # set the hidden variables appropriately
        self._time_since_food_eaten = 0
        self.score = 0

        return self.state, {}

    def _get_distance(self, pos1, pos2):
        x_diff = abs(pos2 % 10 - pos1 % 10)
        y_diff = abs(np.floor(pos2 / 10) - np.floor(pos1 / 10))

        return x_diff + y_diff

    def _transform_state(self, state):
        """
        Turn the game state into a matrix and rotate it so the snake is always facing up then return it back to a list.

        :param state:
        :return:
        """

        matrix = np.array(state).reshape(10, 10)

        if self._snake[-1] - self._snake[-2] == 1:
            # snake going right
            return list(np.rot90(matrix, 1).flatten())
        elif self._snake[-1] - self._snake[-2] == 10:
            # snake going down
            return list(np.rot90(matrix, 2).flatten())
        elif self._snake[-1] - self._snake[-2] == -1:
            # snake going left
            return np.rot90(matrix, 3).flatten()
        elif self._snake[-1] - self._snake[-2] == -10:
            return list(matrix.flatten())

    def _get_valid_moves(self, x: int):
        if x < 0 or x > 99:
            raise Exception(f"x shoudl be from 0 to 99, got {x}")

        if x == 0:
            return [1, 10]
        elif x == 9:
            return [-1, 10]
        elif x == 90:
            return [1, -10]
        elif x == 99:
            return [-1, -10]
        elif np.floor(x / 10) == 0:
            return [-1, 1, 10]
        elif np.floor(x / 10) == 9:
            return [-1, 1, -10]
        elif x % 10 == 0:
            return [1, 10, -10]
        elif x % 10 == 9:
            return [-1, 10, -10]
        else:
            return [-1, 10, -10, 1]

    def _create_random_state(self):
        """
        returns a random valid state.

        """
        state = [0] * 100
        state_lag = [0] * 100

        snake = [np.random.choice(100)]

        tail_length = np.random.choice(8) + 2

        for i in range(tail_length):
            done = False
            end = snake[-1]
            valid_moves = self._get_valid_moves(end)
            while not done:
                try:
                    proposal_move = np.random.choice(len(valid_moves))
                    proposal_loc = end + valid_moves[proposal_move]

                    if proposal_loc not in snake:
                        snake.append(proposal_loc)
                        done = True
                    else:
                        valid_moves.pop(proposal_move)

                except ValueError:
                    done = True

        for i in snake[:-1]:
            state[i] = 2
        state[snake[-1]] = 3

        for i in snake[:-2]:
            state_lag[i] = 2
        state_lag[snake[-2]] = 3

        done = False
        while not done:
            proposal_food_loc = np.random.choice(100)
            if proposal_food_loc not in snake:
                done = True
                state[proposal_food_loc] = 1
                state_lag[proposal_food_loc] = 1

        return state, state_lag, snake
