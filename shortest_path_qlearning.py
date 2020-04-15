#finding the shortest path between two points (0-7) using RL Q-Learning Algorithm

import numpy as np
import pylab as plt
import networkx as nx

points_list = [(0,1), (1,5), (5,6), (5,4), (1,2), (2,3), (2,7)]
goal = 7


'''
#creating graph
G=nx.Graph()
G.add_edges_from(points_list)
pos = nx.spring_layout(G)
nx.draw_networkx_nodes(G,pos)
nx.draw_networkx_edges(G,pos)
nx.draw_networkx_labels(G,pos)
plt.show()'''

#first step is to creat the reward graph or reward matrix initializing each choice with a value (-1)

#how many points in graph? 8 POINTS

MATRIX_SIZE = 8

#then create matrix x*y

R = np.matrix(np.ones(shape=(MATRIX_SIZE, MATRIX_SIZE)))
R *= -1

'''now we have to change the values of reward to be 0 if it is a viable path and 100
if it is a goal path. 
- Viable Path is 0 if it's in the point list and your reverse is also 0. e.g: R[0,1]=0 and R[1,0]=0
- Looking for our point list we see that only the points (2,7) and (7,7) will be 100
'''

for point in points_list:
    #print(point)
    if point[1] == goal:
        R[point] = 100
    else:
        R[point] = 0

    if point[0] == goal:
        R[point[::-1]] = 100
    else:
        # reverse of point
        R[point[::-1]]= 0

# add goal point round trip
R[goal,goal]= 100

''' It's important to mention that our algorithm reads the reward matrix and will looking for the path
which gives the highest reward, It means that the RL algorithm don't try to understand the problem
it only moves forward to the path which gives the best solution.

To read the above matrix, it's important to sau that the y-axis is the state where we are located right now
and the x-axis is our possiblie next actions. Then we build the Q-Learning matrix which will
store all lessons learned from our bot. The Q-Learning model uses a transitional rule formula
and a gamma is the learning parameter.'''


Q = np.matrix(np.zeros([MATRIX_SIZE,MATRIX_SIZE]))

# learning parameter
gamma = 0.8 #learning rate

initial_state = 1

def available_actions(state): #return all the potential moves the actual state can move
    current_state_row = R[state,]
    av_act = np.where(current_state_row >= 0)[1]
    return av_act

available_act = available_actions(initial_state)

def sample_next_action(available_actions_range): #select an action randomly
    next_action = int(np.random.choice(available_act,1))
    return next_action

action = sample_next_action(available_act)

def update(current_state, action, gamma):

  max_index = np.where(Q[action,] == np.max(Q[action,]))[1]

  if max_index.shape[0] > 1:
      max_index = int(np.random.choice(max_index, size = 1))
  else:
      max_index = int(max_index)
  max_value = Q[action, max_index]
  #look to reward matrix 
  Q[current_state, action] = R[current_state, action] + gamma * max_value #QLearning function
  #print('max_value', R[current_state, action] + gamma * max_value)

  if (np.max(Q) > 0):
    return(np.sum(Q/np.max(Q)*100))
  else:
    return (0)

update(initial_state, action, gamma)


#then we can train our model, it will make this proces 700 times allowing the Q-Learning figure out the most efficient path

# Training
scores = []
for i in range(700):
    current_state = np.random.randint(0, int(Q.shape[0]))
    available_act = available_actions(current_state)
    action = sample_next_action(available_act)
    score = update(current_state,action,gamma)
    scores.append(score)
    #print ('Score:', str(score))

print("Trained Q matrix:")
print(Q/np.max(Q)*100)


#after the training we can test and the algorithm will show the best solution

# Testing
current_state = 0
steps = [current_state]

while current_state != 7:

    next_step_index = np.where(Q[current_state,] == np.max(Q[current_state,]))[1]

    if next_step_index.shape[0] > 1:
        next_step_index = int(np.random.choice(next_step_index, size = 1))
    else:
        next_step_index = int(next_step_index)

    steps.append(next_step_index)
    current_state = next_step_index

print("Most efficient path:")
print(steps)

plt.plot(scores) #print the behaviour of our scores
plt.show()

#conclusion: by 400 iteration it converges to the best solution