import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.contrib import layers
from tensorflow.contrib.learn import *
import math
import re
import random

def sigm(NumpyMat):	
	NumpyMat = 1 / (1 + np.exp(-NumpyMat))

	return NumpyMat

def softmax(NumpyMat):
	e_x = np.exp(NumpyMat - np.max(NumpyMat))
	return e_x / e_x.sum()

inputSize = 6040
learning_rate = 0.01
# logs_path = 'D:\loss'
logs_path = './logs'
lambdaR = 0.02
hiddenLayer1 = 900
hiddenLayer2 = 450
hiddenLayer3 = 150

mapping = tf.placeholder("float", [3952,inputSize]) 
X = tf.placeholder("float", [3952,inputSize])
# ItemSide = tf.placeholder("float", [3952,18])
# Xnew = tf.concat([X,ItemSide],1)
V1 = tf.Variable(tf.random_uniform([inputSize ,hiddenLayer1],-1.0 / math.sqrt(inputSize ),1.0 / math.sqrt(inputSize )),trainable=True)
V2 = tf.Variable(tf.random_uniform([hiddenLayer1 ,hiddenLayer2],-1.0 / math.sqrt(hiddenLayer1 ),1.0 / math.sqrt(hiddenLayer1 )),trainable=True)
V3 = tf.Variable(tf.random_uniform([hiddenLayer2 ,hiddenLayer3],-1.0 / math.sqrt(hiddenLayer2 ),1.0 / math.sqrt(hiddenLayer2 )),trainable=True)
mu1 = tf.Variable(tf.zeros([hiddenLayer1]),trainable=True)
mu2 = tf.Variable(tf.zeros([hiddenLayer2]),trainable=True)
mu3 = tf.Variable(tf.zeros([hiddenLayer3]),trainable=True)
W3 = tf.Variable(tf.random_uniform([hiddenLayer3 ,hiddenLayer2],-1.0 / math.sqrt(hiddenLayer3 ),1.0 / math.sqrt(hiddenLayer3 )),trainable=True)
W2 = tf.Variable(tf.random_uniform([hiddenLayer2 ,hiddenLayer1],-1.0 / math.sqrt(hiddenLayer2 ),1.0 / math.sqrt(hiddenLayer2 )),trainable=True)
W1 = tf.Variable(tf.random_uniform([hiddenLayer1 ,inputSize],-1.0 / math.sqrt(hiddenLayer1 ),1.0 / math.sqrt(hiddenLayer1 )),trainable=True)
b3 = tf.Variable(tf.zeros([hiddenLayer2]),trainable=True)
b2 = tf.Variable(tf.zeros([hiddenLayer1]),trainable=True)
b1 = tf.Variable(tf.zeros([inputSize]),trainable=True)
S1 = tf.Variable(tf.random_uniform([hiddenLayer1 ,inputSize],-1.0 / math.sqrt(hiddenLayer1 ),1.0 / math.sqrt(hiddenLayer1 )),trainable=True)
S2 = tf.Variable(tf.random_uniform([hiddenLayer2 ,inputSize],-1.0 / math.sqrt(hiddenLayer2 ),1.0 / math.sqrt(hiddenLayer2 )),trainable=True)
S3 = tf.Variable(tf.random_uniform([hiddenLayer3 ,inputSize],-1.0 / math.sqrt(hiddenLayer3 ),1.0 / math.sqrt(hiddenLayer3 )),trainable=True)
pi1 = tf.Variable(tf.zeros([inputSize]),trainable=True)
pi2 = tf.Variable(tf.zeros([inputSize]),trainable=True)
pi3 = tf.Variable(tf.zeros([inputSize]),trainable=True)

E1 = tf.nn.sigmoid(tf.matmul(X,V1) + mu1)
E2 = tf.nn.sigmoid(tf.add(tf.matmul(E1,V2),mu2))
E3 = tf.nn.sigmoid(tf.add(tf.matmul(E2,V3),mu3))
YS1 = tf.multiply(tf.identity(tf.add(tf.matmul(E1,S1),pi1)),mapping)
YS2 = tf.multiply(tf.identity(tf.add(tf.matmul(E2,S2),pi2)),mapping)
YS3 = tf.multiply(tf.identity(tf.add(tf.matmul(E3,S3),pi3)),mapping)
Ypool = (YS1 + YS2 + YS3)/3

regularize = layers.apply_regularization(layers.l2_regularizer(scale=lambdaR),weights_list=[V1,V2,V3,S1,S2,S3])

difference1NM = X - YS1
difference2NM = X - YS2
difference3NM = X - YS3
differencePool = X - Ypool

Loss1NM = tf.reduce_sum(tf.square(difference1NM))
Loss2NM = tf.reduce_sum(tf.square(difference2NM))
Loss3NM = tf.reduce_sum(tf.square(difference3NM))
LossPool = tf.reduce_sum(tf.square(differencePool))

loss = Loss1NM + Loss2NM + Loss3NM + LossPool+ regularize

optimizer = layers.optimize_loss(loss=loss,global_step=tf.train.get_global_step(),learning_rate=learning_rate,optimizer=tf.train.AdamOptimizer,summaries=["learning_rate","loss","gradients","gradient_norm",])
#optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

saver = tf.train.Saver()
tf.summary.scalar("loss",loss)
merged_summary_op = tf.summary.merge_all()

MAE = np.zeros([5,1])
RMSE = np.zeros([5,1])
PRECISION = np.zeros([5,1])
RECALL = np.zeros([5,1])
BEST_F1 = -1.0
for i in range(0,5):
	trainIndex = np.array(random.sample(range(0,1000209),900209))
	index = np.array(range(0,1000209))
	testIndex = np.setdiff1d(index,trainIndex)
	UserMatrix = np.zeros([6040,3952])
	ItemMatrix = np.zeros([3952,6040])
	UserCompletion = np.zeros([6040,3952])
	ItemCompletion = np.zeros([3952,6040])
	PredictionMatrix = np.zeros([3952,6040])
	PredictionCompletion = np.zeros([3952,6040])
	traininSet = open('ratings.dat')
	print('training indexes: ', trainIndex.shape)
	cc = 0
	for line in traininSet.readlines():
		x = [int(t) for t in line.split('::')]
		# print(cc)
		if(cc in trainIndex):
			UserMatrix[x[0]-1,x[1]-1] = x[2]
			ItemMatrix[x[1]-1,x[0]-1] = x[2]
			UserCompletion[x[0]-1,x[1]-1] = 1
			ItemCompletion[x[1]-1,x[0]-1] = 1
		else:
			PredictionMatrix[x[1]-1,x[0]-1] = x[2]
			PredictionCompletion[x[1]-1,x[0]-1] = 1
		cc = cc + 1
	print('DATA LOADED.........')

	with tf.Session() as sess:
		init = tf.global_variables_initializer()
		sess.run(init)
		summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())

		for k in range(0,1000):
			print('Iteration: ', k)
			for q in range(0,1):
				Xinput = np.zeros([3952,6040])
				Xinput[:,:] = ItemMatrix[q*3952:(q+1)*3952,:]
				A = np.zeros([3952,6040])
				A[:,:] = ItemCompletion[q*3952:(q+1)*3952,:]
				opt,lossP,summary,V_1,V_2,V_3,mu_1,mu_2,mu_3,W_3,W_2,W_1,b_3,b_2,b_1,S_1,S_2,S_3,pi_3,pi_2,pi_1= sess.run([optimizer,loss,merged_summary_op,V1,V2,V3,mu1,mu2,mu3,W3,W2,W1,b3,b2,b1,S1,S2,S3,pi3,pi2,pi1],feed_dict={X:Xinput,mapping:A})#,ItemSide:ItemEncoding})
				summary_writer.add_summary(summary,  k*1 + q)
			print(lossP)

		E_1 = sigm(np.matmul(ItemMatrix,V_1) + mu_1)
		E_2 = sigm(np.matmul(E_1,V_2) + mu_2)
		E_3 = sigm(np.matmul(E_2,V_3) + mu_3)
		Y_S1 = np.matmul(E_1,S_1) + pi_1
		Y_S2 = np.matmul(E_2,S_2) + pi_2
		Y_S3 = np.matmul(E_3,S_3) + pi_3
		Y_pool_1 = (Y_S1 +Y_S2+Y_S3)/3
		Y_pool_3 = Y_S1
		p = np.multiply(Y_pool_1,PredictionCompletion)
		MAE[i,0] = np.sum(np.absolute(PredictionMatrix - p))/np.sum(PredictionCompletion)
		RMSE[i,0] = np.sqrt(np.sum(np.square(PredictionMatrix - p))/np.sum(PredictionCompletion))

		binPRed = PredictionMatrix
		binPRed[binPRed<3] = 0
		binPRed[binPRed > 3] = 1
		binP = p
		binP[binP<3] = 0
		binP[binP > 3] = 1
		tp = 0
		fp = 0
		fn = 0
		for elem1 in range(0,3952):
			for elem2 in range(0,6040):
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
		f1 = 2.0*preicison*recall / (preicison + recall)
		if f1 > BEST_F1:
			BEST_F1 = f1
			save_path = saver.save(sess, "./logs/model.ckpt")
			print("Model saved in path: %s" % save_path)

print(MAE)
print(RMSE)
print(PRECISION)
print(RECALL)
np.save('mae.npy', MAE)
np.save('rmse.npy', RMSE)
np.save('preicison.npy', PRECISION)
np.save('recall.npy', RECALL)
print('Best f1 score: ', BEST_F1)
print(np.sum(MAE)/5,np.sum(RMSE)/5,np.sum(PRECISION)/5,np.sum(RECALL)/5)