import gym
import random
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

class DQNAgent:

	def __init__(self,state_size,action_size):
		self.state_size=state_size
		self.action_size=action_size
		self.gamma=0.95
		self.epsilon=1.0
		self.epsilon_min=0.01
		self.epsilon_decay=0.995
		self.learning_rate=0.001
		self.memory=deque(maxlen=20000)
		self.model=self._build_model()

	def _build_model(self):
		model=Sequential()
		model.add(Dense(24,input_dim=self.state_size,activation='relu'))
		model.add(Dense(24,activation='relu'))
		model.add(Dense(self.action_size,activation='linear'))
		model.compile(loss='mse',optimizer=Adam(lr=self.learning_rate))
		return model

	def act(self,state):
		if np.random.rand()<=self.epsilon:
			return random.randrange(self.action_size)
		act_values=self.model.predict(state)
		return np.argmax(act_values[0])

	def remember(self,state,action,reward,next_state,done):
		self.memory.append((state,action,reward,next_state,done))

	def replay(self,batch_size):
		batch=random.sample(self.memory,batch_size)
		for state,action,reward,next_state,done in batch:
			target=reward
		if not done:
			target=reward+self.gamma*(np.amax(self.model.predict(next_state)[0]))
		target_f=self.model.predict(state)
		target_f[0][action]=target
		self.model.fit(state,target_f,epochs=1,verbose=0)

		if self.epsilon>self.epsilon_min:
			self.epsilon*=self.epsilon_decay


if __name__=='__main__':
	env=gym.make('CartPole-v0')
	agent=DQNAgent(4,2)
	episodes=100000
	for e in range(episodes):
		state=env.reset()
		#print(state)
		state=np.reshape(state,[1,4])
		#print(state)

		for time_t in range(500):
			#env.render()
			action=agent.act(state)
			next_state,reward,done,_=env.step(action)
			next_state=np.reshape(next_state,[1,4])
			agent.remember(state,action,reward,next_state,done)
			state=next_state

			if done:
				print('episode: {}/{}, score: {}'.format(e+1,episodes,time_t))
				break
		agent.replay(10)



	state=env.reset()
	#print(state)
	state=np.reshape(state,[1,4])
	#print(state)
	for time_t in range(500):
		env.render()
		action=agent.act(state)
		next_state,reward,done,_=env.step(action)
		next_state=np.reshape(next_state,[1,4])
		state = next_state
		if done:
			print(time_t)
			break

