from itertools import product
from typing import Callable, List
from dataclasses import dataclass


@dataclass(frozen=True)
class State:
    name: str
    is_terminal: bool


@dataclass(frozen=True)
class Action:
    name: str


class Environment:
    def __init__(self) -> None:
        pass

    def take_action(state, action):
        pass


def epsilon_greedy(state, q_values, epsilon):
    pass


def qlearning(
    states: List[State],
    actions: List[Action],
    initial_state:State,
    env:Environment,
    alpha:float=0.1,
    epsilon:float=0.1,
    gamma:float=0.9,
    initialization_value:float=-2,
    num_episode:int=1000,
    num_steps:int=100,
    choose_action: Callable = epsilon_greedy,
):

    # initialization
    q_values = {}
    for e in product(states, actions):
        q_values[e] = initialization_value if not e[0].is_terminal else 0

    for i in num_episode:
        current_state = initial_state
        for j in num_steps:
            if current_state.is_terminal:
                break
            action = choose_action(current_state, q_values)
            next_state, reward = env.take_action(current_state, action)
            q_values[(current_state, action)] += alpha * (
                reward
                + gamma * max([q_values[(next_state, a)] for a in actions])
                - q_values[(current_state, action)]
            )
            current_state = next_state
