# This script implements functions to reset the state in a different ways


import numpy as np


def get_valid_moves(x: int):
    if x < 0 or x > 99:
        raise Exception(f"x should be from 0 to 99, got {x}")

    if x == 0:
        return [1, 10]
    elif x == 9:
        return [-1, 10]
    elif x == 90:
        return [1, -10]
    elif x == 99:
        return [-1, -10]
    elif x // 10 == 0:  # Top row
        return [-1, 1, 10]
    elif x // 10 == 9:  # Bottom row
        return [-1, 1, -10]
    elif x % 10 == 0:  # Left column
        return [1, 10, -10]
    elif x % 10 == 9:  # Right column
        return [-1, 10, -10]
    else:
        return [-1, 1, 10, -10]


def generate_food_loc(snake):
    # Place food in a random location not occupied by the snake
    done = False
    while not done:
        proposal_food_loc = np.random.choice(100)
        if proposal_food_loc not in snake:
            done = True

    return proposal_food_loc


def create_random_coil_state(max_len: int = 10):
    state = [0] * 100
    state_lag = [0] * 100

    # Initialize the snake with a random starting position
    snake = [np.random.choice(100)]

    # Generate a random tail length (minimum 2, maximum 9)
    tail_length = np.random.choice(max_len - 2) + 3

    # Build the snake
    for _ in range(tail_length):
        done = False
        end = snake[-1]
        valid_moves = get_valid_moves(end)
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

    snake.reverse()
    # Update state and state_lag with snake positions
    for pos in snake[:-1]:
        state[pos] = 2
    state[snake[-1]] = 3

    for pos in snake[:-2]:
        state_lag[pos] = 2
    state_lag[snake[-2]] = 3

    proposal_food_loc = generate_food_loc(snake)

    state[proposal_food_loc] = 1
    state_lag[proposal_food_loc] = 1

    return state, state_lag, snake


def create_random_line_state(max_len: int = 10):
    state = [0] * 100
    state_lag = [0] * 100

    snake_length = np.random.choice(max_len - 2) + 3

    if (snake_length >= 1) & (snake_length <= 10):
        d = np.random.choice(4)  # 0: down, 1: up, 2: right, 3: left

        done = False
        while not done:
            proposal = np.random.choice(100)

            if (d == 0) & (np.floor(proposal / 10) <= 10 - snake_length):
                snake = [proposal + 10 * i for i in range(snake_length)]
                done = True

            elif (d == 1) & (np.floor(proposal / 10) >= snake_length - 1):
                snake = [proposal - 10 * i for i in range(snake_length)]
                done = True

            elif (d == 2) & (proposal % 10 <= 10 - snake_length):
                snake = [proposal + i for i in range(snake_length)]
                done = True

            elif (d == 3) & (proposal % 10 >= snake_length - 1):
                snake = [proposal - i for i in range(snake_length)]
                done = True
    else:
        raise ValueError(f"max_length should be between 1 and 10, got {snake_length}")

    snake.reverse()

    state[snake[-1]] = 3
    for i in snake[:-1]:
        state[i] = 2

    state_lag[snake[-2]] = 3
    for i in snake[:-2]:
        state_lag[i] = 2

    # Place food in a random location not occupied by the snake
    done = False
    while not done:
        proposal_food_loc = np.random.choice(100)
        if proposal_food_loc not in snake:
            done = True
            state[proposal_food_loc] = 1
            state_lag[proposal_food_loc] = 1

    return state, state_lag, snake
