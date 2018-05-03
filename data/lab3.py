
# Code from Chapter 4 of Machine Learning: An Algorithmic Perspective (2nd Edition)
# by Stephen Marsland (http://stephenmonika.net)

# You are free to use, change, or redistribute the code in any way you wish for
# non-commercial purposes, but please maintain the name of the original author.
# This code comes with no warranty of any kind.

# Stephen Marsland, 2008, 2014

# The sinewave regression example

import pylab as pl
import numpy as np

low = np.pi/4
up = np.pi/2
#low = 50
#up = 100
#low = 5
#up = 10
file_name = "/home/nick/Projects/2_Spring_18/CS457/project3/CS457_Lab3/data/sine_results"
#file_name = "/home/nick/Projects/2_Spring_18/CS457/project3/CS457_Lab3/data/inverse_results"
#file_name = "/home/nick/Projects/2_Spring_18/CS457/project3/CS457_Lab3/data/e_results"
#file_name = "/home/nick/Projects/2_Spring_18/CS457/project3/CS457_Lab3/data/log_results"
path = '/home/nick/Projects/2_Spring_18/CS457/project3/CS457_Lab3/data/images_sine/'
#path = '/home/nick/Projects/2_Spring_18/CS457/project3/CS457_Lab3/data/images_inverse/'
#path = '/home/nick/Projects/2_Spring_18/CS457/project3/CS457_Lab3/data/images_e/'
#path = '/home/nick/Projects/2_Spring_18/CS457/project3/CS457_Lab3/data/images_log/'


for num_neurons in range(1,11):
    path_ = str(path)
    path_ += str(num_neurons)	
    path_ += '/sine_'
    #path_ += '/inverse_'
    #path_ += '/e_'
    #path_ += '/log_'
    for train_iterations in [100]:
        path__ = str(path_)
        path__ += str(train_iterations)
        for num in range(20,101,20):
            path___ = str(path__)
            path___ += '_'
            path___ += str(num)
            path___ += '.png'

            # Set up the data
            x = np.linspace(low,up,num).reshape((num,1))
            t = np.sin(x)
            #t = np.reciprocal(x)
            #t = np.exp(-x)
            #t = np.log10(x)
            x = (x-up/2)*2

            res = open(file_name, "a")
            res.write("\n\nRange: [")
            res.write(str(low))
            res.write(", ")
            res.write(str(up))
            res.write("]\nN=")
            res.write(str(num))
            res.write("\nHidden neurons: ")
            res.write(str(num_neurons))
            res.write("\nTrain iterations: ")
            res.write(str(train_iterations))
            res.write("\n")
            res.close()

            # Split into training, testing, and validation sets
            train = x[0::2,:]
            test = x[1::4,:]
            valid = x[3::4,:]
            traintarget = t[0::2,:]
            testtarget = t[1::4,:]
            validtarget = t[3::4,:]


        	 # Plot the data
            pl.plot(testtarget,'bo')
            pl.xlabel('x')
            pl.ylabel('t')

		# Perform basic training with a small MLP
            import mlp
            net = mlp.mlp(train,traintarget,num_neurons,outtype='linear')
            net.mlptrain(train,traintarget,0.25,train_iterations)

    		# Use early stopping
            net.earlystopping(train,traintarget,valid,validtarget,0.25,file_name)
            test = np.concatenate((test,np.ones((np.shape(test)[0],1))),axis=1)
            outputs = net.mlpfwd(test)
            pl.plot(outputs,'r+')
            pl.savefig(path___)
            pl.clf()
            res = open(file_name, "a")
            res.write("\nTest Error: ")
            res.write(str((0.5*sum((outputs-testtarget)**2))))
            res.close()

			# Test out different sizes of network
			#count = 0
			#out = np.zeros((10,7))
			#for nnodes in [1,2,3,5,10,25,50]:
			#    for i in range(10):
			#        net = mlp.mlp(train,traintarget,nnodes,outtype='linear')
			#        out[i,count] = net.earlystopping(train,traintarget,valid,validtarget,0.25)
			#    count += 1
		
			#print out
			#print out.mean(axis=0)
			#print out.var(axis=0)
			#print out.max(axis=0)
			#print out.min(axis=0)

			#pl.show()

print 'done'
