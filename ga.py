import numpy as np
import gym
import matplotlib.pyplot as plt


def init(input_len,output_len,pop_size):
    nets = []
    for _ in range(pop_size):
        nets.append(2*np.random.random((output_len,input_len))-1)

    return nets



def get_action(net,observation):

    def sigmoid(x):
        return 1/(1+np.exp(-x))

    out = sigmoid(np.dot(net,observation))

    return np.argmax(out)



def train(net,train_len,env,render=False):
    observation = env.reset()

    if render:
        env.render()

    fit = 0
    done = False
    i = 0
    while not done:
        action = get_action(net,observation)
        print(i)
        observation, reward, done, info = env.step(action)
        if render:
            env.render()

        fit += reward 
        i += 1

        if i > train_len:
            break

    return fit




def breed(parents):
    children = []
    for i in range(len(parents)):
        p0 = parents[i][0]
        p1 = parents[i][1]
        child = np.zeros(p0.shape)

        for j in range(p0.shape[0]):
            if np.random.uniform() >= 0.5:
                child[j,:] = p0[j,:]
            else:
                child[j,:] = p1[j,:]

        children.append(child)
    return children


def mutate(children,random_percentage,mutation_rate):
    mutated_children = []
    for i in range(len(children)):
        child = children[i]
        for j in range(child.shape[0]):
            rand = np.random.random(child[j].shape)
            y = np.where(rand > random_percentage)

            rand = 2*rand-1
            rand = rand*np.abs(rand)
            rand[y] = 0

            child[j] += rand

        mutated_children.append(child)
    return mutated_children


env_name = "CartPole-v0"
env = gym.make(env_name)

observation = env.reset()
input_len = len(observation) 


action_len = env.action_space.n


random_percentage = 0.2 
max_change_rate = 0.01 
train_length = 1000
population_size = 100 
parents_portion = 0.8 


epochs = 20

nets = init(input_len,action_len,population_size)


for ch in range(epoch):
    fits = np.zeros((len(nets)))

    
    for i in range(len(nets)):
        fits[i] = train(nets[i],train_length,env)

    sorted_fits = np.argsort(fits)[::-1]
    sorted_nets = [nets[i] for i in sorted_fits]

    parents = []
    for i in range(int(population_size*parents_portion)):
        r0 = np.power(np.random.random(),2)
        r1 = np.power(np.random.random(),2)
        parents.append((sorted_nets[int(np.floor(population_size*r0))], sorted_nets[int(np.floor(population_size*r1))]))


    children = breed(parents)


    mutated_children = mutate(children,random_percentage,max_change_rate)

    next_gen = mutated_children
    next_gen.extend(sorted_nets[0:population_size-len(mutated_children)])

    best_fit = fits[sorted_fits[0]]

    print("epoch {}/{}: best_fit is {}".format(ch+1,epochs,best_fit))


    nets = next_gen


plt.plot(fits)
plt.show()
#train(nets[sorted_fits[0]],train_length,env,render = True)