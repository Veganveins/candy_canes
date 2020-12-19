## You'll probably need to install this package first:
##   pip install --user kaggle-environments

from kaggle_environments import make


# 'mab' stands for Multi-armed-bandit https://www.kaggle.com/c/santa-2020
env = make("mab", debug=True)


# Run two of our agents against each other.
# Or you can just use "random" to play against a bot that just selects randomly
agent_name1 = './thompson.py'
agent_name2 = './basic_v2.py'
steps = env.run([agent_name1, agent_name2])

#   Render an html ipython replay of the tictactoe game.
#   This looks cool in a Jupyter notebook, but it's pretty slow actually!
# env.render(mode="ipython", width=800, height=500)

# Print the status at the end of the game:
final_status = steps[-1]
assert len(final_status) ==2 

print('\nFinal scores:')
print(agent_name1, '\t', final_status[0]['reward'])
print(agent_name2, '\t', final_status[1]['reward'])
