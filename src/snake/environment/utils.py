import numpy as np


def rotate_state(state, snake):
    """
    Rotate the game state so that it is always facing up, this provides a many to one mapping, so
    the number of possible states is reduced to 25% of the original size.
    """

    matrix = np.array(state).reshape(10, 10)
    head_change = snake[-1] - snake[-2]

    if head_change == 1:
        # snake going right
        return list(np.rot90(matrix, 1).flatten())
    elif head_change == 10:
        # snake going down
        return list(np.rot90(matrix, 2).flatten())
    elif head_change == -1:
        # snake going left
        return list(np.rot90(matrix, 3).flatten())
    elif head_change == -10:
        # snake going up
        return list(matrix.flatten())
    else:
        raise ValueError(
            f"Invalid snake, {snake} debug by making sure snake is being updated properly during steps."
        )


# transform states so head is always on the right.
def invert_loc(i):
    """
    returns the inverted position for i
    """
    row = np.ceil((i + 1) / 10) - 1
    new_col = 9 - (i % 10)

    return int(10 * row + new_col)


def flip_state(state):
    snake_head = state.index(3)

    if snake_head % 10 < 5:
        new_snake_tail = []
        snake_tail = [i for i, value in enumerate(state) if value == 2]
        for i in snake_tail:
            new_snake_tail.append(invert_loc(i))

        new_snake_head = invert_loc(snake_head)
        new_food_loc = invert_loc(state.index(1))

        new_state = [0] * 100

        for i in new_snake_tail:
            new_state[i] = 2

        new_state[new_snake_head] = 3
        new_state[new_food_loc] = 1

        return new_state

    else:
        return state


def proximity_reward(head, old_head, food_loc):
    head_food_dist = get_distance(head, food_loc)
    old_head_food_dist = get_distance(old_head, food_loc)

    if head_food_dist < old_head_food_dist:
        reward = 0.2
    elif head_food_dist > old_head_food_dist:
        reward = -0.2
    else:
        raise Exception(
            "Something has gone wrong when calculating reward. Likely caused by updating of states during "
            "step."
        )

    return reward


def get_distance(pos1, pos2):
    """
    :param pos1: first array location
    :param pos2: second array locaiton
    :return: the distance between two arrays points on the grid
    """
    x_diff = abs(pos2 % 10 - pos1 % 10)
    y_diff = abs(np.floor(pos2 / 10) - np.floor(pos1 / 10))

    return x_diff + y_diff
