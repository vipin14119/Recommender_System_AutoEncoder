import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.contrib import layers
from tensorflow.contrib.learn import *
import math
import re
import random
#import bigfloat



def sigm(NumpyMat):	
	NumpyMat = 1 / (1 + np.exp(-NumpyMat))

	return NumpyMat

def softmax(NumpyMat):
	e_x = np.exp(NumpyMat - np.max(NumpyMat))
	return e_x / e_x.sum()



inputSize = 73421
learning_rate = 0.01
logs_path = './logs'
lambdaR = 0.002
# hiddenLayer1 = 400
# hiddenLayer2 = 250
# hiddenLayer3 = 100
hiddenLayer1 = 1000
hiddenLayer2 = 500
hiddenLayer3 = 200


mapping = tf.placeholder("float", [100,inputSize]) 

X = tf.placeholder("float", [100,inputSize])


V1 = tf.Variable(tf.random_uniform([inputSize ,hiddenLayer1],-1.0 / math.sqrt(inputSize ),1.0 / math.sqrt(inputSize )),trainable=True)
V2 = tf.Variable(tf.random_uniform([hiddenLayer1 ,hiddenLayer2],-1.0 / math.sqrt(hiddenLayer1 ),1.0 / math.sqrt(hiddenLayer1 )),trainable=True)
V3 = tf.Variable(tf.random_uniform([hiddenLayer2 ,hiddenLayer3],-1.0 / math.sqrt(hiddenLayer2 ),1.0 / math.sqrt(hiddenLayer2 )),trainable=True)
mu1 = tf.Variable(tf.zeros([hiddenLayer1]),trainable=True)
mu2 = tf.Variable(tf.zeros([hiddenLayer2]),trainable=True)
mu3 = tf.Variable(tf.zeros([hiddenLayer3]),trainable=True)
#W3 = tf.Variable(tf.random_uniform([hiddenLayer3 ,hiddenLayer2],-1.0 / math.sqrt(hiddenLayer3 ),1.0 / math.sqrt(hiddenLayer3 )),trainable=True)
#W2 = tf.Variable(tf.random_uniform([hiddenLayer2 ,hiddenLayer1],-1.0 / math.sqrt(hiddenLayer2 ),1.0 / math.sqrt(hiddenLayer2 )),trainable=True)
#W1 = tf.Variable(tf.random_uniform([hiddenLayer1 ,inputSize],-1.0 / math.sqrt(hiddenLayer1 ),1.0 / math.sqrt(hiddenLayer1 )),trainable=True)
#b3 = tf.Variable(tf.zeros([hiddenLayer2]),trainable=True)
#b2 = tf.Variable(tf.zeros([hiddenLayer1]),trainable=True)
#b1 = tf.Variable(tf.zeros([inputSize]),trainable=True)
S1 = tf.Variable(tf.random_uniform([hiddenLayer1 ,inputSize],-1.0 / math.sqrt(hiddenLayer1 ),1.0 / math.sqrt(hiddenLayer1 )),trainable=True)
S2 = tf.Variable(tf.random_uniform([hiddenLayer2 ,inputSize],-1.0 / math.sqrt(hiddenLayer2 ),1.0 / math.sqrt(hiddenLayer2 )),trainable=True)
S3 = tf.Variable(tf.random_uniform([hiddenLayer3 ,inputSize],-1.0 / math.sqrt(hiddenLayer3 ),1.0 / math.sqrt(hiddenLayer3 )),trainable=True)
pi1 = tf.Variable(tf.zeros([inputSize]),trainable=True)
pi2 = tf.Variable(tf.zeros([inputSize]),trainable=True)
pi3 = tf.Variable(tf.zeros([inputSize]),trainable=True)

Ypool = (tf.identity(tf.add(tf.matmul(tf.nn.sigmoid(tf.matmul(X,V1) + mu1),S1),pi1)) + tf.identity(tf.add(tf.matmul(tf.nn.sigmoid(tf.add(tf.matmul(tf.nn.sigmoid(tf.matmul(X,V1) + mu1),V2),mu2)),S2),pi2)) + tf.identity(tf.add(tf.matmul(tf.nn.sigmoid(tf.add(tf.matmul(tf.nn.sigmoid(tf.add(tf.matmul(tf.nn.sigmoid(tf.matmul(X,V1) + mu1),V2),mu2)),V3),mu3)),S3),pi3)))/3

loss = tf.reduce_mean(tf.reduce_sum(tf.square(X - tf.multiply(tf.identity(tf.add(tf.matmul(tf.nn.sigmoid(tf.matmul(X,V1) + mu1),S1),pi1)),mapping)), 1, keep_dims=True)) + tf.reduce_mean(tf.reduce_sum(tf.square(X - tf.multiply(tf.identity(tf.add(tf.matmul(tf.nn.sigmoid(tf.add(tf.matmul(tf.nn.sigmoid(tf.matmul(X,V1) + mu1),V2),mu2)),S2),pi2)),mapping)), 1, keep_dims=True)) + tf.reduce_mean(tf.reduce_sum(tf.square(X - tf.multiply(tf.identity(tf.add(tf.matmul(tf.nn.sigmoid(tf.add(tf.matmul(tf.nn.sigmoid(tf.add(tf.matmul(tf.nn.sigmoid(tf.matmul(X,V1) + mu1),V2),mu2)),V3),mu3)),S3),pi3)),mapping)), 1, keep_dims=True)) + tf.reduce_mean(tf.reduce_sum(tf.square(X - (tf.multiply(tf.identity(tf.add(tf.matmul(tf.nn.sigmoid(tf.matmul(X,V1) + mu1),S1),pi1)),mapping) + tf.multiply(tf.identity(tf.add(tf.matmul(tf.nn.sigmoid(tf.add(tf.matmul(tf.nn.sigmoid(tf.matmul(X,V1) + mu1),V2),mu2)),S2),pi2)),mapping) + tf.multiply(tf.identity(tf.add(tf.matmul(tf.nn.sigmoid(tf.add(tf.matmul(tf.nn.sigmoid(tf.add(tf.matmul(tf.nn.sigmoid(tf.matmul(X,V1) + mu1),V2),mu2)),V3),mu3)),S3),pi3)),mapping))/3), 1, keep_dims=True))+ layers.apply_regularization(layers.l2_regularizer(scale=lambdaR),weights_list=[V1,V2,V3,S1,S2,S3])

optimizer = layers.optimize_loss(loss=loss,global_step=tf.train.get_global_step(),learning_rate=learning_rate,optimizer=tf.train.AdamOptimizer,summaries=["learning_rate","loss","gradients","gradient_norm",])

saver = tf.train.Saver()
tf.summary.scalar("loss",loss)
merged_summary_op = tf.summary.merge_all()

MAE = np.zeros([5,1])
RMSE = np.zeros([5,1])
PRECISION = np.zeros([5,1])
RECALL = np.zeros([5,1])
# BEST_F1 = -1.0
BEST_MAE = 999999999999
jesterData = open('jester.txt')
newData = np.zeros([7342100,3])
cc3 = 0
for line in jesterData.readlines():
	x = re.split("[,\\n]+",line)
	newData[cc3*100:(cc3+1)*100,0] = cc3
	newData[cc3*100:(cc3+1)*100,1] = np.array(range(0,100))
	newData[cc3*100:(cc3+1)*100,2] = list(map(float, x[0:100])) # np.array(list(map(float, x[0:100]))) + 10
	cc3 = cc3+1
#print(newData)
index = np.where(newData[:,2] == 99)[0]
newData = np.delete(newData,index,axis = 0)


for i in range(0,1):
	trainIndex = np.array(random.sample(range(0,4136360),3722724))
	indexT = np.array(range(0,100000))
	testIndex = np.setdiff1d(indexT,trainIndex)
	UserMatrix = np.zeros([100,73421])
	UserCompletion = np.zeros([100,73421])
	PredictionMatrix = np.zeros([100,73421])
	PredictionCompletion = np.zeros([100,73421])
	cc = 0
	for row in range(0,4136360):
		
		if(row%10!=i):
			# print(row)
			UserMatrix[int(newData[row,1]),int(newData[row,0])] = newData[row,2]
			UserCompletion[int(newData[row,1]),int(newData[row,0])] = 1
		else:
			PredictionMatrix[int(newData[row,1]),int(newData[row,0])] = newData[row,2]
			PredictionCompletion[int(newData[row,1]),int(newData[row,0])] = 1
		cc = cc + 1



	
	print('STARTING TRAINING.........')
	with tf.Session() as sess:
		init = tf.global_variables_initializer()
		sess.run(init)
		summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())

		#inputLastIND = np.random.randint(2, size=(58,943))
		for k in range(0,800):
			print('Iteration: ', k)
			for q in range(0,1):
				
				opt,lossP,summary,V_1,V_2,V_3,mu_1,mu_2,mu_3,S_1,S_2,S_3,pi_3,pi_2,pi_1,Y= sess.run([optimizer,loss,merged_summary_op,V1,V2,V3,mu1,mu2,mu3,S1,S2,S3,pi3,pi2,pi1,Ypool],feed_dict={X:UserMatrix,mapping:UserCompletion})
				summary_writer.add_summary(summary,  k*1 + q)
			print(lossP)

		Y_pool_1 = Y
		#Y_pool_1 = Y
		p = np.multiply(Y_pool_1,PredictionCompletion)
		MAE[i,0] = np.sum(np.absolute(PredictionMatrix - p))/np.sum(PredictionCompletion)
		RMSE[i,0] = np.sqrt(np.sum(np.square(PredictionMatrix - p))/np.sum(PredictionCompletion))
		if MAE[i, 0] < BEST_MAE:
			BEST_MAE = MAE[i, 0]
			save_path = saver.save(sess, "./logs/model_jester3_with_0_lr_"+str(learning_rate)+".ckpt")
			print("Model saved in path: %s" % save_path)
		binPRed = PredictionMatrix
		binPRed[binPRed<5] = 0
		binPRed[binPRed >5] = 1
		binP = p
		binP[binP<5] = 0
		binP[binP >5] = 1
		tp = 0
		fp = 0
		fn = 0
		for elem1 in range(0,100):
			for elem2 in range(0,73421):
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
np.save('mae.npy', MAE)
np.save('rmse.npy', RMSE)
np.save('preicison.npy', PRECISION)
np.save('recall.npy', RECALL)
print('Best MAE: ', BEST_MAE)
print(np.sum(MAE)/1,np.sum(RMSE)/1,np.sum(PRECISION)/1,np.sum(RECALL)/1)