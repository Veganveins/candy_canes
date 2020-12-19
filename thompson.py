# pip install kaggle-environments --upgrade -q
from collections import defaultdict, namedtuple
import random
import numpy as np

alpha_p = 1 # TODO: dynamically update this as we pull more times
beta_p = 1 # TODO: dynamically update this as we pull more times
class Arm:
    # An arm needs to remember the relevant history:
    # - when my opponent selected it
    # - when I selected it, with the reward I got
    def __init__(self, i):
        self.i = i
        self.my_actions_and_rewards = {} # step number to reward
        self.opponent_actions = set() # step number when my opponent chose it
    def __repr__(self):
        if len(self.my_actions_and_rewards) == 0 and len(self.opponent_actions) == 0:
            return ''
        return repr(dict(
            me = list(sorted(self.my_actions_and_rewards.items())),
            op = list(self.opponent_actions),
            ))
    def simple_score_from_my_pulls(self):
        # this is really naive.
        # Maybe better to 'discount' the old successes, as we know the arms
        # loses 3% each time it's pull (50%-to-47%, or 50%-to-48.5%, I'm not sure)
        how_often_have_I_pulled_this = len(self.my_actions_and_rewards)
        reward_I_got_from_this = sum(self.my_actions_and_rewards.values())
        return how_often_have_I_pulled_this, reward_I_got_from_this
    def compute_params(self):
        alpha = alpha_p + self.simple_score_from_my_pulls()[1]
        beta = beta_p + self.simple_score_from_my_pulls()[0] - self.simple_score_from_my_pulls()[1]
        return alpha, beta
    def evaluate_arm(self):
        alpha, beta = self.compute_params()
        arm_rank = np.random.beta(alpha, beta)
        return arm_rank


O = namedtuple('O', 'remainingOverageTime step agentIndex reward lastActions')

allActions = []
my_total_reward = 0
my_rewards = []

arms = [Arm(i) for i in range(100)]

def random_agent(observation, configuration):
    o = O(**observation)
    if o.step == 0:
        # No option on the first round really
        return random.randrange(configuration.banditCount)

    global my_total_reward

    allActions.append(o.lastActions)
    my_last_action = allActions[-1][o.agentIndex]
    opponents_last_action = allActions[-1][1-o.agentIndex]

    my_last_reward = o.reward - my_total_reward
    my_rewards.append(my_last_reward)
    my_total_reward = o.reward
    arms[my_last_action].my_actions_and_rewards[o.step-1]=my_last_reward
    arms[opponents_last_action].opponent_actions.add(o.step-1)

    # Start taking action

    # If the last one worked, try it again
    arm_status = [a.evaluate_arm() for a in arms]

    best_arm = max(arm_status) # returns the "expected value" as %
    best_arm_location = arm_status.index(best_arm) # returns int 0 - 99
    
    return best_arm_location

    # Give up, random choice
    #return random.randrange(configuration.banditCount)

"""
This competition is modeled after the "multi-armed bandit problem," a classic probability-based, reinforcement learning problem that examines the exploration-exploitation tradeoff dilemma.

In this problem, both participants will work with the same set of 100 vending machines (bandits).
Each bandit provides a random reward based on a probability distribution specific to that machine.
Every round each player selects ("pulls") a bandit,
the likelihood of a reward decreases by 3%.

Each agent can see the move of the other agent, but will not see whether a reward was gained in their respective bandit pull.

This episode continues for 2000 rounds per player (4000 pulls total).


An Agent will receive an observation containing their total reward,
the bandits pulled by both players in the previous turn (lastActions),
the current step of the competition,
and the remainingOverageTime.
"""
