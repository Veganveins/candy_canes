# pip install kaggle-environments --upgrade -q
from collections import defaultdict, namedtuple
import random

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

    #print(my_total_reward, my_rewards)
    assert 0 <= my_last_reward <= 1
    #print()

    assert my_total_reward == sum(r for a in arms for r in a.my_actions_and_rewards.values())


    # Start taking action

    # If the last one worked, try it again
    if my_last_reward == 1:
        return my_last_action

    # If my opponent repeated himself, copy him
    if len(allActions) >= 2:
        op2 = allActions[-2][1-o.agentIndex]
        op1 = allActions[-1][1-o.agentIndex]
        assert op1 == opponents_last_action
        if op1 == op2:
            return op1


    # Anything I haven't selected yet
    unselected_by_me = [a.i for a in arms if len(a.my_actions_and_rewards) == 0]
    if len(unselected_by_me) > 0:
        return unselected_by_me[0]

    # Give up, random choice
    return random.randrange(configuration.banditCount)

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
