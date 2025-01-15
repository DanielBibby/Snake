import numpy as np
from .utils import rotate_state, proximity_reward, flip_state
from .reset_utils import (
    create_random_coil_state,
    create_random_line_state,
)
from .base_game_environment import BaseSnakeEnv



class SnakeEnvTransformedState(BaseSnakeEnv):
    """
    This child of BaseSnakeEnv transformed the state representation so that, during training, the algorithm would
    be shown states that are rotated and flipped so that
        1. The snake is always moving up
        2. The head is always on the right side of the grid.
    This provides an eight-to-one mapping of the state space, leading to more efficient training.
    """

    _flip_occured: bool # indicates the state was mirrored, meaning left / right movements should also be mirrored.

    def __init__(self):
        super(SnakeEnvTransformedState, self).__init__()

    def step(self, action):
        if self._flip_occured:
            flipped_action = {
                0: 2, # maps a left movement to a right movement
                1: 1, # straight movements are unchanged
                2: 0, # maps a right movement to a left movement
            }
            if isinstance(action, np.ndarray) and action.size == 1:
                action = action.item()  # Extract the single element as an integer

            action = flipped_action[action]

        self.state, reward, self.done, truncated, info = super().step(action)

        # rotate snake so it is moving upwards
        rotated_state = rotate_state(state=self.state, snake=self._snake)
        # mirror state if the snake head is in the left half of the grid
        flipped_state, self._flip_occured = flip_state(state=rotated_state)

        return flipped_state, reward, self.done, truncated, info

    def reset(self, seed=None):
        state, info = super().reset()

        rotated_state = rotate_state(state, self._snake)
        flipped_state, self._flip_occured = flip_state(state=rotated_state)

        return flipped_state, info


class SnakeEnvProximityReward(BaseSnakeEnv):
    """
    This class introduces a small reward of +/- 0.2 for moving towards / away from food, making rewards less sparse
    during training.
    """

    def __init__(self):
        super(SnakeEnvProximityReward, self).__init__()

    def step(self, action):
        self.state, reward, self.done, truncated, info = super().step(action)

        if self.done:
            pass
        # new reward logic
        if reward == 0:
            # if agents moves towards / away from food it gets a reward of +/- 0.2
            head = self._snake[-1]
            old_head = self._snake[-2]
            food_loc = self.state.iloc[1]
            reward = proximity_reward(head=head, old_head=old_head, food_loc=food_loc)

        return self.state, reward, self.done, truncated, info


class SnakeEnvTranformedProximityReward(BaseSnakeEnv):
    """
    This class implements the state transormation discussed in SnakeEnvTransformedState and the new reward
    function discuessed in SnakeEnvProximityReward.
    """

    def __init__(self):
        super(SnakeEnvTranformedProximityReward, self).__init__()

    def step(self, action):
        if self._flip_occured:
            flipped_action = {
                0: 2,
                1: 1,
                2: 0,
            }
            if isinstance(action, np.ndarray) and action.size == 1:
                action = action.item()  # Extract the single element as an integer

            action = flipped_action[action]

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

            new_reward = proximity_reward(
                head=head, old_head=old_head, food_loc=food_loc
            )

        rotated_state = rotate_state(state=self.state, snake=self._snake)
        flipped_state, self._flip_occured = flip_state(state=rotated_state)

        return flipped_state, new_reward, self.done, truncated, info

    def reset(self, seed=None):
        state, info = super().reset()

        rotated_state = rotate_state(state=self.state, snake=self._snake)
        flipped_state, self._flip_occured = flip_state(state=rotated_state)

        return flipped_state, info


class SnakeEnvCoilReset(SnakeEnvTranformedProximityReward):
    """
    Environment with new reward function and state transformation.

    reset method now resets to a random snake based on the following logic.

    Snake head is randomly placed on grid and tail length is sampled randomly from [2,...,max_reset_length (default = 10)], working
    backwards the tail is formed by a random walk until either there are no valid squares or the max tail length is
    reached.
    """

    def __init__(self, max_reset_length: int = 10):
        super(SnakeEnvCoilReset, self).__init__()
        self.max_reset_length = max_reset_length

    def reset(self, seed=None):
        super().reset(seed=seed)  # this is needed to run gym.Env.reset

        self.state, self._state_lag, self._snake = create_random_coil_state(
            max_len=self.max_reset_length
        )
        self.done = False
        self.score = 0
        self._time_since_food_eaten = 0

        rotated_state = rotate_state(state=self.state, snake=self._snake)
        flipped_state, self._flip_occured = flip_state(state=rotated_state)

        return flipped_state, {}


class SnakeEnvLineReset(SnakeEnvTranformedProximityReward):
    """
    Environment with new reward function and state transformation.

    reset method now resets snake to a random position based on the following logic.

    Snake head is randomly placed on grid and a random direction is chosen tail length is sampled randomly from
    [2, ..., min(max_reset_length (default = 10), distance to wall in chosen direction)] a straight tail is made of
    the sampled length.
    """

    def __init__(self, max_reset_length: int = 10):
        super(SnakeEnvLineReset, self).__init__()

        self.max_reset_length = max_reset_length

    def reset(self, seed=None):
        super().reset(
            seed=seed
        )

        self.state, self._state_lag, self._snake = create_random_line_state(
            max_len=self.max_reset_length
        )
        self.done = False

        # set the hidden variables appropriately
        self._time_since_food_eaten = 0
        self.score = 0

        rotated_state = rotate_state(state=self.state, snake=self._snake)
        flipped_state, self._flip_occured = flip_state(state=rotated_state)

        return flipped_state, {}
