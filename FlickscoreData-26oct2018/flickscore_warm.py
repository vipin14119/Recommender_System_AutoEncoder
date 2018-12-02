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

user_col = []
user_id = []
movie_col = []
rating_col = []
json_file = open('ratings.json', 'r')
data = json_file.readlines()
zusers = 0
anyusers = 0
u = 0

movies_stats = {}
u_id = 1
dropped_user_dict = {}

for d in data:
    d = json.loads(d)
#     print(d)
    rated = d['rated']
    del rated['submit']
    # print(rated.keys())
    flag = False
    for key, values in rated.items():
        flag = True
        r = int(values[0])
        if r != 0:
            
            if r < 0:
                r += 2
            elif r == 1:
                r += 1
            user_col.append(d['_id'])
            movie_col.append(key)
            rating_col.append(r)
            if key in movies_stats.keys():
                movies_stats[key] += 1
            else:
                movies_stats[key] = 1
    if flag == False:
        zusers += 1
    else:
        anyusers += 1
        dropped_user_dict[d['_id']] = u_id
        u_id += 1

dropped_movie_dict = {}
for i in range(len(list(movies_stats.keys()))):
    dropped_movie_dict[list(movies_stats.keys())[i]] = i+1

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
    
    l = genre_string.lower()
    if "music" in l:
        l = l.replace('music"', "musical")
    l = l.replace("[", "").replace("]", "").replace(" ", "").replace("\"", "").split(",")
    return l


movies_df_dup = movies_df.copy()

movies_df_dup = movies_df_dup.loc[movies_df_dup['movie_id'].isin(dropped_movie_dict)]

all_genres = []
for i in movies_df_dup['genre']:
    s = string_to_list(i)
    all_genres += s
all_genres = sorted(list(set(all_genres)))[1:]
num_genres = len(all_genres)
# print(num_genres)
# print(all_genres)

def encode_genre(g_list):
    encoded_genres = np.zeros((len(g_list), num_genres))
    for i in range(len(g_list)):
        
        g_avail = string_to_list(g_list[i])

        for g in g_avail:
            if g in all_genres:
                encoded_genres[i,all_genres.index(g)] = 1
    return encoded_genres


movies_df_dup['movie_id'] = movies_df_dup.movie_id.apply(create_id_mapping)
ItemEncoding = encode_genre(list(movies_df_dup.genre))

inputSize = 757
learning_rate = 0.008
logs_path = './logs'
lambdaR = 0.018
hiddenLayer1 = 200
hiddenLayer2 = 150
hiddenLayer3 = 100

mapping = tf.placeholder("float", [1027,inputSize]) 
X = tf.placeholder("float", [1027,inputSize])
ItemSide = tf.placeholder("float", [1027, 19])
Xnew = tf.concat([X,ItemSide],1)

V1 = tf.Variable(tf.random_uniform([inputSize + 19,hiddenLayer1],-1.0 / math.sqrt(inputSize + 19),1.0 / math.sqrt(inputSize + 19)),trainable=True)
V2 = tf.Variable(tf.random_uniform([hiddenLayer1 + 19 ,hiddenLayer2],-1.0 / math.sqrt(hiddenLayer1+ 19 ),1.0 / math.sqrt(hiddenLayer1+ 19 )),trainable=True)
V3 = tf.Variable(tf.random_uniform([hiddenLayer2 + 19,hiddenLayer3],-1.0 / math.sqrt(hiddenLayer2 + 19),1.0 / math.sqrt(hiddenLayer2 + 19)),trainable=True)
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
S1 = tf.Variable(tf.random_uniform([hiddenLayer1 + 19,inputSize],-1.0 / math.sqrt(hiddenLayer1 + 19),1.0 / math.sqrt(hiddenLayer1 + 19)),trainable=True)
S2 = tf.Variable(tf.random_uniform([hiddenLayer2 + 19,inputSize],-1.0 / math.sqrt(hiddenLayer2 + 19),1.0 / math.sqrt(hiddenLayer2 + 19)),trainable=True)
S3 = tf.Variable(tf.random_uniform([hiddenLayer3 + 19,inputSize],-1.0 / math.sqrt(hiddenLayer3 + 19),1.0 / math.sqrt(hiddenLayer3 + 19)),trainable=True)
S1I = tf.Variable(tf.random_uniform([hiddenLayer1 + 19,19],-1.0 / math.sqrt(hiddenLayer1 + 19),1.0 / math.sqrt(hiddenLayer1 + 19)),trainable=True)
S2I = tf.Variable(tf.random_uniform([hiddenLayer2 + 19,19],-1.0 / math.sqrt(hiddenLayer2 + 19),1.0 / math.sqrt(hiddenLayer2 + 19)),trainable=True)
S3I = tf.Variable(tf.random_uniform([hiddenLayer3 + 19,19],-1.0 / math.sqrt(hiddenLayer3 + 19),1.0 / math.sqrt(hiddenLayer3 + 19)),trainable=True)
pi1 = tf.Variable(tf.zeros([inputSize]),trainable=True)
pi2 = tf.Variable(tf.zeros([inputSize]),trainable=True)
pi3 = tf.Variable(tf.zeros([inputSize]),trainable=True)
pi1I = tf.Variable(tf.zeros([19]),trainable=True)
pi2I = tf.Variable(tf.zeros([19]),trainable=True)
pi3I = tf.Variable(tf.zeros([19]),trainable=True)

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
	ItemMatrix = np.zeros([1027,757])
	ItemCompletion = np.zeros([1027,757])
	PredictionMatrix = np.zeros([1027,757])
	PredictionCompletion = np.zeros([1027,757])
	traininSet = open('ratings_warm.csv')
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
		for k in range(0,800):
			print('Iteration: ', k)
			for q in range(0,1):
				Xinput = np.zeros([1027,757])
				Xinput[:,:] = ItemMatrix[q*1027:(q+1)*1027,:]
				A = np.zeros([1027,757])
				A[:,:] = ItemCompletion[q*1027:(q+1)*1027,:]
				itemInfo = ItemEncoding[q*1027:(q+1)*1027,:]
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
		binPRed[binPRed<1.5] = 0
		binPRed[binPRed > 1.5] = 1
		binP = p
		binP[binP<1.5] = 0
		binP[binP > 1.5] = 1
		tp = 0
		fp = 0
		fn = 0
		for elem1 in range(0,1027):
			for elem2 in range(0,757):
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
