import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.contrib import layers
from tensorflow.contrib.learn import *
import math
import re
import random
import pandas as pd

def sigm(NumpyMat):
	NumpyMat = 1 / (1 + np.exp(-NumpyMat))

	return NumpyMat

def softmax(NumpyMat):
	e_x = np.exp(NumpyMat - np.max(NumpyMat))
	return e_x / e_x.sum()


with open('user_ratedmovies.dat') as dat:
    next(dat)
    dat_df = pd.DataFrame(l.rstrip().split() for l in dat)

with open('movie_genres.dat') as dat:
    next(dat)
#     lines = dat.readlines()
    gen_df = pd.DataFrame(l.rstrip().split() for l in dat)

all_movies = sorted(list(set(gen_df[0])))
num_movies = len(all_movies)
movie_dict = {}

for i in range(len(all_movies)):
    movie_dict[all_movies[i]] = i+1

def movie_id_map(id):
    return movie_dict[id]

df = dat_df[[0, 1, 2]]
df = df.sample(frac=1)
# rated_movies = sorted(list(set(df[1])))
users = sorted(list(set(df[0])))


user_dict = {}
for i in range(len(users)):
    user_dict[users[i]] = i+1

def user_id_map(id):
    return user_dict[id]

df[0] = df[0].apply(user_id_map)
df[1] = df[1].apply(movie_id_map)

gen_df_dup = gen_df.copy()
gen_df_dup[0] = gen_df_dup[0].apply(movie_id_map)
encoded_genres = np.zeros((num_movies, 20))
all_genres = sorted(list(set(gen_df_dup[1])))
for _, row in gen_df_dup.iterrows():
    encoded_genres[row[0]-1, all_genres.index(row[1])] = 1

ItemEncoding = encoded_genres

NUM_USERS = 2113
NUM_MOVIES = 10197
NUM_RATINGS = 855598
NUM_RATINGS_90 = 770039
NUM_ENCODED_SIZE = 20

inputSize = NUM_USERS # number of users
NUM_ITERATIONS = 1000
learning_rate = 0.005
# logs_path = 'D:\loss'
logs_path = './logs'
lambdaR = 0.02
hiddenLayer1 = 400
hiddenLayer2 = 200
hiddenLayer3 = 100

mapping = tf.placeholder("float", [NUM_MOVIES,inputSize])
X = tf.placeholder("float", [NUM_MOVIES,inputSize])
ItemSide = tf.placeholder("float", [NUM_MOVIES,NUM_ENCODED_SIZE])
Xnew = tf.concat([X,ItemSide],1)
# first 3 encoders
V1 = tf.Variable(tf.random_uniform([inputSize + NUM_ENCODED_SIZE,hiddenLayer1],-1.0 / math.sqrt(inputSize + NUM_ENCODED_SIZE),1.0 / math.sqrt(inputSize + NUM_ENCODED_SIZE)),trainable=True)
V2 = tf.Variable(tf.random_uniform([hiddenLayer1 + NUM_ENCODED_SIZE ,hiddenLayer2],-1.0 / math.sqrt(hiddenLayer1+ NUM_ENCODED_SIZE ),1.0 / math.sqrt(hiddenLayer1+ NUM_ENCODED_SIZE )),trainable=True)
V3 = tf.Variable(tf.random_uniform([hiddenLayer2 + NUM_ENCODED_SIZE,hiddenLayer3],-1.0 / math.sqrt(hiddenLayer2 + NUM_ENCODED_SIZE),1.0 / math.sqrt(hiddenLayer2 + NUM_ENCODED_SIZE)),trainable=True)
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
S1 = tf.Variable(tf.random_uniform([hiddenLayer1 + NUM_ENCODED_SIZE,inputSize],-1.0 / math.sqrt(hiddenLayer1 + NUM_ENCODED_SIZE),1.0 / math.sqrt(hiddenLayer1 + NUM_ENCODED_SIZE)),trainable=True)
S2 = tf.Variable(tf.random_uniform([hiddenLayer2 + NUM_ENCODED_SIZE,inputSize],-1.0 / math.sqrt(hiddenLayer2 + NUM_ENCODED_SIZE),1.0 / math.sqrt(hiddenLayer2 + NUM_ENCODED_SIZE)),trainable=True)
S3 = tf.Variable(tf.random_uniform([hiddenLayer3 + NUM_ENCODED_SIZE,inputSize],-1.0 / math.sqrt(hiddenLayer3 + NUM_ENCODED_SIZE),1.0 / math.sqrt(hiddenLayer3 + NUM_ENCODED_SIZE)),trainable=True)
S1I = tf.Variable(tf.random_uniform([hiddenLayer1 + NUM_ENCODED_SIZE,NUM_ENCODED_SIZE],-1.0 / math.sqrt(hiddenLayer1 + NUM_ENCODED_SIZE),1.0 / math.sqrt(hiddenLayer1 + NUM_ENCODED_SIZE)),trainable=True)
S2I = tf.Variable(tf.random_uniform([hiddenLayer2 + NUM_ENCODED_SIZE,NUM_ENCODED_SIZE],-1.0 / math.sqrt(hiddenLayer2 + NUM_ENCODED_SIZE),1.0 / math.sqrt(hiddenLayer2 + NUM_ENCODED_SIZE)),trainable=True)
S3I = tf.Variable(tf.random_uniform([hiddenLayer3 + NUM_ENCODED_SIZE,NUM_ENCODED_SIZE],-1.0 / math.sqrt(hiddenLayer3 + NUM_ENCODED_SIZE),1.0 / math.sqrt(hiddenLayer3 + NUM_ENCODED_SIZE)),trainable=True)
pi1 = tf.Variable(tf.zeros([inputSize]),trainable=True)
pi2 = tf.Variable(tf.zeros([inputSize]),trainable=True)
pi3 = tf.Variable(tf.zeros([inputSize]),trainable=True)
pi1I = tf.Variable(tf.zeros([NUM_ENCODED_SIZE]),trainable=True)
pi2I = tf.Variable(tf.zeros([NUM_ENCODED_SIZE]),trainable=True)
pi3I = tf.Variable(tf.zeros([NUM_ENCODED_SIZE]),trainable=True)

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

#Loss1NM = tf.reduce_mean(tf.nn.l2_loss(difference1NM))
#Loss2NM = tf.reduce_mean(tf.nn.l2_loss(difference2NM))
#Loss3NM = tf.reduce_mean(tf.nn.l2_loss(difference3NM))
#LossPool = tf.reduce_mean(tf.nn.l2_loss(differencePool))

loss = Loss1NM + Loss2NM + Loss3NM + LossPool+ regularize + Loss1NMI + Loss2NMI + Loss3NMI + LossPoolI

optimizer = layers.optimize_loss(loss=loss,global_step=tf.train.get_global_step(),learning_rate=learning_rate,optimizer=tf.train.AdamOptimizer,summaries=["learning_rate","loss","gradients","gradient_norm",])
#optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

saver = tf.train.Saver()
tf.summary.scalar("loss",loss)
merged_summary_op = tf.summary.merge_all()

MAE = np.zeros([5,1])
RMSE = np.zeros([5,1])
PRECISION = np.zeros([5,1])
RECALL = np.zeros([5,1])
BEST_MAE = 999999999999
# BEST_F1 = -1.0
for i in range(0,1):
	trainIndex = np.array(random.sample(range(0,NUM_RATINGS),NUM_RATINGS_90))
	trainIndex.sort()
	index = np.array(range(0,NUM_RATINGS))
	testIndex = np.setdiff1d(index,trainIndex)
	testIndex.sort()
	# UserMatrix = np.zeros([NUM_USERS,NUM_MOVIES])
	ItemMatrix = np.zeros([NUM_MOVIES,NUM_USERS])
	# UserCompletion = np.zeros([NUM_USERS,NUM_MOVIES])
	ItemCompletion = np.zeros([NUM_MOVIES,NUM_USERS])
	# PredictionMatrix = np.zeros([NUM_MOVIES,NUM_USERS])
	# PredictionCompletion = np.zeros([NUM_MOVIES,NUM_USERS])
	traininSet = open('ratings_full.csv')
	print('training indexes: ', trainIndex.shape)
	cc = 0
	tindex = 0
	for line in traininSet.readlines():
		x = [float(t) for t in line.replace("\n", "").split(',')]
		# print(cc)
		if(tindex < NUM_RATINGS_90 and cc == trainIndex[tindex]):
			# UserMatrix[x[0]-1,x[1]-1] = x[2]
			ItemMatrix[int(x[2]-1),int(x[1]-1)] = x[3]
			# UserCompletion[x[0]-1,x[1]-1] = 1
			ItemCompletion[int(x[2]-1),int(x[1]-1)] = 1
			tindex = tindex + 1
		# else:
		# 	PredictionMatrix[x[1]-1,x[0]-1] = x[2]
		# 	PredictionCompletion[x[1]-1,x[0]-1] = 1
		cc = cc + 1
	print('DATA LOADED.........')

	with tf.Session() as sess:
		init = tf.global_variables_initializer()
		sess.run(init)
		summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())

		for k in range(0,NUM_ITERATIONS):
			print('Iteration: ', k)
			for q in range(0,1):
				# Xinput = np.zeros([NUM_MOVIES,NUM_USERS])
				# Xinput[:,:] = ItemMatrix[q*NUM_MOVIES:(q+1)*NUM_MOVIES,:]
				# A = np.zeros([NUM_MOVIES,NUM_USERS])
				# A[:,:] = ItemCompletion[q*NUM_MOVIES:(q+1)*NUM_MOVIES,:]
				# itemInfo = ItemEncoding[q*NUM_MOVIES:(q+1)*NUM_MOVIES,:]
				opt,lossP,summary,V_1,V_2,V_3,mu_1,mu_2,mu_3,W_3,W_2,W_1,b_3,b_2,b_1,S_1,S_2,S_3,pi_3,pi_2,pi_1,y_pool_I,y_pool= sess.run([optimizer,loss,merged_summary_op,V1,V2,V3,mu1,mu2,mu3,W3,W2,W1,b3,b2,b1,S1,S2,S3,pi3,pi2,pi1,YpoolI,Ypool],feed_dict={X:ItemMatrix[0:NUM_MOVIES, :],mapping: ItemCompletion[0:NUM_MOVIES, :],ItemSide:ItemEncoding[0:NUM_MOVIES, :]})
				summary_writer.add_summary(summary,  k*1 + q)
			print(lossP)

		newInp = np.array(np.concatenate((ItemMatrix,ItemEncoding),axis = 1))
		E_1 = sigm(np.matmul(newInp,V_1) + mu_1)
		E_1 = np.array(np.concatenate((E_1,ItemEncoding),axis = 1))
		E_2 = sigm(np.matmul(E_1,V_2) + mu_2)
		E_2 = np.array(np.concatenate((E_2,ItemEncoding),axis = 1))
		E_3 = sigm(np.matmul(E_2,V_3) + mu_3)
		E_3 = np.array(np.concatenate((E_3,ItemEncoding),axis = 1))
		#D_2 = sigm(np.matmul(E_3,W_3) + b_3)
		#D_2 = np.array(np.concatenate((D_2,ItemEncoding),axis = 1))
		#D_1 = sigm(np.matmul(D_2,W_2) + b_2)
		#D_1 = np.array(np.concatenate((D_1,ItemEncoding),axis = 1))
		#Y_last = np.matmul(D_1,W_1) + b_1
		Y_S1 = np.matmul(E_1,S_1) + pi_1
		Y_S2 = np.matmul(E_2,S_2) + pi_2
		Y_S3 = np.matmul(E_3,S_3) + pi_3
		Y_pool_1 = (Y_S1 +Y_S2+Y_S3)/3
		Y_pool_3 = Y_S1

		testSet = open('ratings_full.csv')
		print('testing indexes: ', trainIndex.shape)
		cc = 0
		tindex = 0
		ItemMatrix = np.zeros([NUM_MOVIES,NUM_USERS])
		ItemCompletion = np.zeros([NUM_MOVIES,NUM_USERS])
		for line in testSet.readlines():
			x = [float(t) for t in line.replace("\n", "").split(',')]
			# print(cc)
			if(tindex < (NUM_RATINGS-NUM_RATINGS_90) and cc == testIndex[tindex]):
				# UserMatrix[x[0]-1,x[1]-1] = x[2]
				ItemMatrix[int(x[2]-1),int(x[1]-1)] = x[3]
				# UserCompletion[x[0]-1,x[1]-1] = 1
				ItemCompletion[int(x[2]-1),int(x[1]-1)] = 1
				tindex = tindex + 1
			cc = cc + 1
		print('TESTING DATA LOADED.........')
		p = np.multiply(Y_pool_1,ItemCompletion)
		MAE[i,0] = np.sum(np.absolute(ItemMatrix - p))/np.sum(ItemCompletion)
		RMSE[i,0] = np.sqrt(np.sum(np.square(ItemMatrix - p))/np.sum(ItemCompletion))

		if MAE[i, 0] < BEST_MAE:
			BEST_MAE = MAE[i, 0]
			save_path = saver.save(sess, "./logs/cold_model_lr"+str(learning_rate)+".ckpt")
			print("Model saved in path: %s" % save_path)

		binPRed = ItemMatrix
		binPRed[binPRed<3] = 0
		binPRed[binPRed > 3] = 1
		binP = p
		binP[binP<3] = 0
		binP[binP > 3] = 1
		tp = 0
		fp = 0
		fn = 0
		for elem1 in range(0,NUM_MOVIES):
			for elem2 in range(0,NUM_USERS):
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
		# f1 = 2.0*preicison*recall / (preicison + recall)
		# if f1 > BEST_F1:
		# 	BEST_F1 = f1
		# 	save_path = saver.save(sess, "./logs/model.ckpt")
		# 	print("Model saved in path: %s" % save_path)

print(MAE)
print(RMSE)
print(PRECISION)
print(RECALL)
np.save('cold_mae.npy', MAE)
np.save('cold_rmse.npy', RMSE)
np.save('cold_preicison.npy', PRECISION)
np.save('cold_recall.npy', RECALL)
print('Best MAE: ', BEST_MAE)
print(np.sum(MAE)/1,np.sum(RMSE)/1,np.sum(PRECISION)/1,np.sum(RECALL)/1)
