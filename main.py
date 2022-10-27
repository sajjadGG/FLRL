import gym
from qlearning import qlearning,Environment, State, Action
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree

class GYMAcroENV(Environment):
    def __init__(self) -> None:
        super().__init__()
        self.env = gym.make('Acrobot-v1',render_mode="human")
        self.env.reset()
            
        low = np.array([ 1., 1., 1., 1., 12.57, 28.27])
        high = np.array([ -1., -1., -1., -1., -12.57, -28.27])
        self.state_values = np.linspace(low,high)
        self.tree = cKDTree(self.state_values)

    def _observe2state(self,observe):
        return self.tree.query(np.array(observe),k=1)[1] 

    def take_action(self,state, action):
        observe, reward, terminated, truncated , info = self.env.step(int(action.name))
        self.env.render()
        next_state = self._observe2state(observe)+1
        return State(str(next_state),terminated),reward

states = [State('inital',False)]
low = np.array([ 1., 1., 1., 1., 12.57, 28.27])
high = np.array([ -1., -1., -1., -1., -12.57, -28.27])
state_values = np.linspace(low,high)
states += [State(str(i),False) for i in range(len(state_values))]
actions = [Action('0'),Action('1'),Action('2')]

r = []
env = GYMAcroENV()
qlearning(states,actions,states[0],env)

# for _ in range(500):
#     next_state, reward, terminated, truncated , info = env.step(env.action_space.sample())
#     # env.render()

#     r.append(reward)
# plt.plot(range(500),r)
# plt.show()