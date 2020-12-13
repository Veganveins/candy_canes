# pip install kaggle-environments --upgrade -q
from collections import namedtuple
import random

O = namedtuple('O', 'remainingOverageTime step agentIndex reward lastActions')

allActions = []

def random_agent(observation, configuration):
    o = O(**observation)
    if o.step == 0:
        return random.randrange(configuration.banditCount)
    allActions.append(o.lastActions)
    my_opponents_last_action = allActions[-1][1-o.agentIndex]
    return my_opponents_last_action


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
