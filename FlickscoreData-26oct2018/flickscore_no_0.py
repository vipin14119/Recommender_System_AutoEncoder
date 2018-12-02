import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.contrib import layers
from tensorflow.contrib.learn import *
import math
import re
import random
import json

def sigm(NumpyMat):	
	NumpyMat = 1 / (1 + np.exp(-NumpyMat))

	return NumpyMat

def softmax(NumpyMat):
	e_x = np.exp(NumpyMat - np.max(NumpyMat))
	return e_x / e_x.sum()

MOVIES_PATH = 'movies.csv'
MOVIES_HEADER = ['movie_id', 'description', 'language', 'released', 'rating', 'writer', 'director', 'cast', 'genre', 'name']
movies_df = pd.read_csv(MOVIES_PATH, sep=',')

unique_movies = list(movies_df.movie_id)
movies_dict = {}
for i in range(len(unique_movies)):
    movies_dict[unique_movies[i]] = i+1


def create_id_mapping(id):
    return movies_dict[id]

def string_to_list(genre_string):
    l = genre_string.lower().replace('music"', "musical").replace("[", "").replace("]", "").replace(" ", "").replace("\"", "").split(",")
    return l

all_genres = []
for i in movies_df['genre']:
    s = string_to_list(i)
    all_genres += s
all_genres = sorted(list(set(all_genres)))[1:]
num_genres = len(all_genres)



def encode_genre(g_list):
    encoded_genres = np.zeros((len(g_list), num_genres))
    for i in range(len(g_list)):
        g_avail = string_to_list(g_list[i])
        for g in g_avail:
            if g in all_genres:
                encoded_genres[i,all_genres.index(g)] = 1
    return encoded_genres

movies_df_dup = movies_df.copy()
movies_df_dup['movie_id'] = movies_df_dup.movie_id.apply(create_id_mapping)

ItemEncoding = encode_genre(movies_df_dup.genre)

inputSize = 924
learning_rate = 0.002
logs_path = './logs'
lambdaR = 0.018
hiddenLayer1 = 150
hiddenLayer2 = 100
hiddenLayer3 = 50

mapping = tf.placeholder("float", [2850,inputSize]) 
X = tf.placeholder("float", [2850,inputSize])
ItemSide = tf.placeholder("float", [2850, 20])
Xnew = tf.concat([X,ItemSide],1)

V1 = tf.Variable(tf.random_uniform([inputSize + 20,hiddenLayer1],-1.0 / math.sqrt(inputSize + 20),1.0 / math.sqrt(inputSize + 20)),trainable=True)
V2 = tf.Variable(tf.random_uniform([hiddenLayer1 + 20 ,hiddenLayer2],-1.0 / math.sqrt(hiddenLayer1+ 20 ),1.0 / math.sqrt(hiddenLayer1+ 20 )),trainable=True)
V3 = tf.Variable(tf.random_uniform([hiddenLayer2 + 20,hiddenLayer3],-1.0 / math.sqrt(hiddenLayer2 + 20),1.0 / math.sqrt(hiddenLayer2 + 20)),trainable=True)
# bias of first three encoders
mu1 = tf.Variable(tf.zeros([hiddenLayer1]),trainable=True)
mu2 = tf.Variable(tf.zeros([hiddenLayer2]),trainable=True)
mu3 = tf.Variable(tf.zeros([hiddenLayer3]),trainable=True)
W3 = tf.Variable(tf.random_uniform([hiddenLayer3 ,hiddenLayer2],-1.0 / math.sqrt(hiddenLayer3 ),1.0 / math.sqrt(hiddenLayer3 )),trainable=True)
W2 = tf.Variable(tf.random_uniform([hiddenLayer2 ,hiddenLayer1],-1.0 / math.sqrt(hiddenLayer2 ),1.0 / math.sqrt(hiddenLayer2 )),trainable=True)
W1 = tf.Variable(tf.random_uniform([hiddenLayer1 ,inputSize],-1.0 / math.sqrt(hiddenLayer1 ),1.0 / math.sqrt(hiddenLayer1 )),trainable=True)
b3 = tf.Variable(tf.zeros([hiddenLayer2]),trainable=True)
b2 = tf.Variable(tf.zeros([hiddenLayer1]),trainable=True)
b1 = tf.Variable(tf.zeros([inputSize]),trainable=True)
S1 = tf.Variable(tf.random_uniform([hiddenLayer1 + 20,inputSize],-1.0 / math.sqrt(hiddenLayer1 + 20),1.0 / math.sqrt(hiddenLayer1 + 20)),trainable=True)
S2 = tf.Variable(tf.random_uniform([hiddenLayer2 + 20,inputSize],-1.0 / math.sqrt(hiddenLayer2 + 20),1.0 / math.sqrt(hiddenLayer2 + 20)),trainable=True)
S3 = tf.Variable(tf.random_uniform([hiddenLayer3 + 20,inputSize],-1.0 / math.sqrt(hiddenLayer3 + 20),1.0 / math.sqrt(hiddenLayer3 + 20)),trainable=True)
S1I = tf.Variable(tf.random_uniform([hiddenLayer1 + 20,20],-1.0 / math.sqrt(hiddenLayer1 + 20),1.0 / math.sqrt(hiddenLayer1 + 20)),trainable=True)
S2I = tf.Variable(tf.random_uniform([hiddenLayer2 + 20,20],-1.0 / math.sqrt(hiddenLayer2 + 20),1.0 / math.sqrt(hiddenLayer2 + 20)),trainable=True)
S3I = tf.Variable(tf.random_uniform([hiddenLayer3 + 20,20],-1.0 / math.sqrt(hiddenLayer3 + 20),1.0 / math.sqrt(hiddenLayer3 + 20)),trainable=True)
pi1 = tf.Variable(tf.zeros([inputSize]),trainable=True)
pi2 = tf.Variable(tf.zeros([inputSize]),trainable=True)
pi3 = tf.Variable(tf.zeros([inputSize]),trainable=True)
pi1I = tf.Variable(tf.zeros([20]),trainable=True)
pi2I = tf.Variable(tf.zeros([20]),trainable=True)
pi3I = tf.Variable(tf.zeros([20]),trainable=True)

E1 = tf.nn.sigmoid(tf.matmul(Xnew,V1) + mu1)
E1 = tf.concat([E1,ItemSide],1)
E2 = tf.nn.sigmoid(tf.add(tf.matmul(E1,V2),mu2))
E2 = tf.concat([E2,ItemSide],1)
E3 = tf.nn.sigmoid(tf.add(tf.matmul(E2,V3),mu3))
E3 = tf.concat([E3,ItemSide],1)

YS1 = tf.multiply(tf.identity(tf.add(tf.matmul(E1,S1),pi1)),mapping)
YS2 = tf.multiply(tf.identity(tf.add(tf.matmul(E2,S2),pi2)),mapping)
YS3 = tf.multiply(tf.identity(tf.add(tf.matmul(E3,S3),pi3)),mapping)
YS1I = tf.identity(tf.add(tf.matmul(E1,S1I),pi1I))
YS2I = tf.identity(tf.add(tf.matmul(E2,S2I),pi2I))
YS3I = tf.identity(tf.add(tf.matmul(E3,S3I),pi3I))
YpoolI = (YS1I + YS2I + YS3I)/3
Ypool = (YS1 + YS2 + YS3)/3


regularize = layers.apply_regularization(layers.l2_regularizer(scale=lambdaR),weights_list=[V1,V2,V3,S1,S2,S3,S1I,S2I,S3I])



difference1NM = X - YS1
difference2NM = X - YS2
difference3NM = X - YS3
differencePool = X - Ypool


Loss1NM = tf.reduce_mean(tf.reduce_sum(tf.square(difference1NM), 1, keep_dims=True))
Loss2NM = tf.reduce_mean(tf.reduce_sum(tf.square(difference2NM), 1, keep_dims=True))
Loss3NM = tf.reduce_mean(tf.reduce_sum(tf.square(difference3NM), 1, keep_dims=True))
LossPool = tf.reduce_mean(tf.reduce_sum(tf.square(differencePool), 1, keep_dims=True))

difference1NMI = ItemSide - YS1I
difference2NMI = ItemSide - YS2I
difference3NMI = ItemSide - YS3I
differencePoolI = ItemSide - YpoolI


Loss1NMI = tf.reduce_mean(tf.square(difference1NMI))
Loss2NMI = tf.reduce_mean(tf.square(difference2NMI))
Loss3NMI = tf.reduce_mean(tf.square(difference3NMI))
LossPoolI = tf.reduce_mean(tf.square(differencePoolI))

loss = Loss1NM + Loss2NM + Loss3NM + LossPool+ regularize + Loss1NMI + Loss2NMI + Loss3NMI + LossPoolI

optimizer = layers.optimize_loss(loss=loss,global_step=tf.train.get_global_step(),learning_rate=learning_rate,optimizer=tf.train.AdamOptimizer,summaries=["learning_rate","loss","gradients","gradient_norm",])

saver = tf.train.Saver()
tf.summary.scalar("loss",loss)
merged_summary_op = tf.summary.merge_all()

MAE = np.zeros([5,1])
RMSE = np.zeros([5,1])
PRECISION = np.zeros([5,1])
RECALL = np.zeros([5,1])
BEST_MAE = 999999999
for i in range(0,5):
	trainIndex = np.array(random.sample(range(0,10181),9163))
	indexT = np.array(range(0,10181))
	testIndex = np.setdiff1d(indexT,trainIndex)
	ItemMatrix = np.zeros([2850,924])
	ItemCompletion = np.zeros([2850,924])
	PredictionMatrix = np.zeros([2850,924])
	PredictionCompletion = np.zeros([2850,924])
	traininSet = open('ratings.csv')
	print('training indexes: ', trainIndex.shape)
	cc = 0
	for line in traininSet.readlines():
		x = [int(t) for t in line.split(',')]
		# print(cc)
		if(cc in trainIndex):
			# UserMatrix[x[0]-1,x[1]-1] = x[2]
			ItemMatrix[x[2]-1,x[1]-1] = x[3]
			# UserCompletion[x[0]-1,x[1]-1] = 1
			ItemCompletion[x[2]-1,x[1]-1] = 1
		else:
			PredictionMatrix[x[2]-1,x[1]-1] = x[3]
			PredictionCompletion[x[2]-1,x[1]-1] = 1
		cc = cc + 1
	print('DATA LOADED.........')

	with tf.Session() as sess:
		init = tf.global_variables_initializer()
		sess.run(init)
		summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())
		for k in range(0,500):
			print('Iteration: ', k)
			for q in range(0,1):
				Xinput = np.zeros([2850,924])
				Xinput[:,:] = ItemMatrix[q*2850:(q+1)*2850,:]
				A = np.zeros([2850,924])
				A[:,:] = ItemCompletion[q*2850:(q+1)*2850,:]
				itemInfo = ItemEncoding[q*2850:(q+1)*2850,:]
				opt,lossP,summary,V_1,V_2,V_3,mu_1,mu_2,mu_3,W_3,W_2,W_1,b_3,b_2,b_1,S_1,S_2,S_3,pi_3,pi_2,pi_1= sess.run([optimizer,loss,merged_summary_op,V1,V2,V3,mu1,mu2,mu3,W3,W2,W1,b3,b2,b1,S1,S2,S3,pi3,pi2,pi1],feed_dict={X:Xinput,mapping:A,ItemSide:itemInfo})
				summary_writer.add_summary(summary,  k*1 + q)
			print(lossP)


		newInp = np.array(np.concatenate((ItemMatrix,ItemEncoding),axis = 1))
		E_1 = sigm(np.matmul(newInp,V_1) + mu_1)
		E_1 = np.array(np.concatenate((E_1,ItemEncoding),axis = 1))
		E_2 = sigm(np.matmul(E_1,V_2) + mu_2)
		E_2 = np.array(np.concatenate((E_2,ItemEncoding),axis = 1))
		E_3 = sigm(np.matmul(E_2,V_3) + mu_3)
		E_3 = np.array(np.concatenate((E_3,ItemEncoding),axis = 1))
		Y_S1 = np.matmul(E_1,S_1) + pi_1
		Y_S2 = np.matmul(E_2,S_2) + pi_2
		Y_S3 = np.matmul(E_3,S_3) + pi_3
		Y_pool_1 = (Y_S1 +Y_S2+Y_S3)/3
		Y_pool_3 = Y_S1
		p = np.multiply(Y_pool_1,PredictionCompletion)
		MAE[i,0] = np.sum(np.absolute(PredictionMatrix - p))/np.sum(PredictionCompletion)
		RMSE[i,0] = np.sqrt(np.sum(np.square(PredictionMatrix - p))/np.sum(PredictionCompletion))

		if MAE[i, 0] < BEST_MAE:
			BEST_MAE = MAE[i, 0]
			save_path = saver.save(sess, "./logs/model_no_0_lr_"+str(learning_rate)+".ckpt")
			print("Model saved in path: %s" % save_path)

		binPRed = PredictionMatrix
		binPRed[binPRed<3] = 0
		binPRed[binPRed > 3] = 1
		binP = p
		binP[binP<3] = 0
		binP[binP > 3] = 1
		tp = 0
		fp = 0
		fn = 0
		for elem1 in range(0,2850):
			for elem2 in range(0,924):
				if(binP[elem1,elem2] == 1 and binP[elem1,elem2] == binPRed[elem1,elem2]):
					tp = tp +1
				elif(binP[elem1,elem2] == 1 and binP[elem1,elem2] != binPRed[elem1,elem2]):
					fp = fp +1
				elif(binP[elem1,elem2] == 0 and binP[elem1,elem2] != binPRed[elem1,elem2]):
					fn = fn +1
		preicison = tp/(tp+fp)
		recall = tp/(tp + fn)
		PRECISION[i,0] = preicison
		RECALL[i,0] = recall

print(MAE)
print(RMSE)
print(PRECISION)
print(RECALL)
print(np.sum(MAE)/5,np.sum(RMSE)/5,np.sum(PRECISION)/5,np.sum(RECALL)/5)