# MiniProjects
Small and Fun projects to enhance my understanding

Contains:
1. Story Extender
2. Image Compression using Kmeans
3. Sentiment Analysis on IMDb data
4. OpenAI Gym experiments

# 1. Story Extender
• Implemented a Recurrent Neural Network from scratch in numpy that can read in a sequence of words
  and then generate words in that style, the architecture uses a single hidden layer with tanh activations

• Implemented the backpropagation with Adagrad as the optimization technique from scratch in numpy

• Cross-entropy was implemented as the loss function and the model predicts the next words char by char

Any suggestions are welcome.
rnn.py is the script. This code was inspired from the youtube series "The Math of Intelligence" by Siraj Raval.

# 2. Image Compression using Kmeans
• Implemented the Kmeans Clustering algorithm from scratch using numpy to detect major color clusters

• Achieved a compression ratio of 6 with decent quality by detecting the major 16 colors since 4 bits are
  sufficient for a pixel after compression while compared to 24 bits (8 for each of R,G&B) before compression
  
kmeans.py is the script. This code was inspired from Assignment 7 of Machine Learning course by Andrew NG in Coursera

# 3. Sentiment Analysis on IMDb data
• Implemented a 1D convnet in Keras to predict the review of a movie to be positive or negative

• The model was trained on the IMDb data after converting the words into word vectors using word2vec

• Achieved an accuracy of 87% on the test data from IMDb while the state of the art was at 89.9%

sentiment_analysis.py is the script. This Code was inspired from the blog https://machinelearningmastery.com/predict-sentiment-movie-reviews-using-deep-learning/ by Jason Brownlee

# 4. OpenAI Gym Experiments
• Implemented Q-learning agents using TensorFlow for both ‘Frozenlake’ and ‘Cartpole’ environments

• Implemented a Genetic Algorithm from scratch to train a neural network for the ‘Cartpole’ environment

frozenlake-ql.py is the script containing Qlearning agent for Frozenlake
cartpole-ql.py is the script containing Qlearning agent for Cartpole
ga.py contains the genetic algorithm for Frozenlake

Not great results but atleast I tried. The codes are inspired from the OpenAI submissions for the respective environments


