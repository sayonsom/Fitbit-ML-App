# -*- coding: utf-8 -*-
"""
Created on Sat Jan 14 19:17:03 2017

@author: Sayonsom Chanda
"""

import numpy as np
import matplotlib.pyplot as plt

class Neutral_Network(object):
    def __init__(self):
    # Define the Hyper Parameters
    
        self.inputLayerSize = 2
        self.outputLayerSize = 1
        self.hiddenLayerSize = 3
    
    #Define the weights
    
        self.W1 = np.random.randn(self.inputLayerSize,self.hiddenLayerSize)
        self.W2 = np.random.randn(self.hiddenLayerSize,self.outputLayerSize)
    
    
    ## Using Matrix Notation to pass multiple input data at once. 
    
    def forward(self,X):
        #Propagate the inputs through the networks
    
        self.z2 = np.dot(X, self.W1)
        self.a2 = self.sigmoid(self.z2)
        self.z3 = np.dot(self.a2,self.W2)
        yHat = self.sigmoid(self.z3)
        
        return yHat
        
    def sigmoid(self, z):
        return 1/(1+np.exp(-z))
        
    def sigmoidPrime(z):
        #Derivative of Sigmoid Function
    
        return np.exp(-z)/((1+np.exp(-z))**2)
        
    def costFunction(self,X,y):
        #Compute the error for given X,y by using the weights already known
    
        self.yHat = self.forward(X)
        J = 0.5*sum((y-self.yHat)**2)
        
        return J
        
    def costFunctionPrime(self, X, y):
        #Compute derivative with respect to W1 and W2
    
        self.yHat = self.forward(X)
        
        delta3 = np.multiply(-y(y-self.yHat),self.sigmoidPrime(self.z3))
        dJdW2 = np.dot(self.a2.T, delta3)
        
        delta2 = np.dot(delta3, self.W2.T)*self.sigmoidPrime(self.z2)
        dJdW1 =  np.dot(X.T, delta2)

        return dJdW1, dJdW2
        
    ### NUMERICAL GRADIENT CHECKING --- HELPER FUNCTIONS
        
    def getParams(self):
        #Combining W1 and W2 into a single vector -- helps in finding norm later
    
        params = np.concatenate((self.W1.ravel(), self.W2.ravel()))
        return params
        
    def setParams(self,params):
        W1_start = 0
        W1_end = self.hiddenLayerSize*self.inputLayerSize
        self.W1 = np.reshape(params[W1_start:W1_end],(self.inputLayerSize, self.hiddenLayerSize))
        W2_end = W1_end + self.hiddenLayerSize*self.outputLayerSize
        self.W2 = np.reshape(params[W1_end:W2_end],(self.hiddenLayerSize, self.outputLayerSize))
        
    def computeGradients(self, X, y):
        dJdW1, dJdW2 = self.costFunctionPrime(X,y):
            return np.concatenate((dJW1.ravel(), dJW2.ravel()))
            
             
        
    def computeNumericalGradient(N,X,y):
        paramsInitial = N.getParams()
        numgrad = np.zeros(paramsInitial.shape)
        perturb = np.zeros(paramsInitial.shape)
        e = 1e-4
        
        for p in range(len(paramsInitial)):
            # Check for small disturbances in data
        
            perturb[p] = e
            N.setParams(paramsInitial + perturb)
            loss2 = N.costFunction(X,y)
            N.setParams(paramsInitial - perturb)
            loss1 = N.costFunction(X,y)
            
            #Compute Numerial Gradient 
            
            numgrad[p] = (loss2 - loss1)/(2*e)
            
            # Reset the value we changed to original value

            perturb[p] = 0   
            
        #Reset the Params Value
            
        N.setParams(paramsInitial)
        
        return numgrad
            
#    testValues = np.arange(-5,5,0.01)
#    plt.plot(testValues, sigmoid(testValues), linewidth=2)
#    plt.plot(testValues, sigmoidPrime(testValues), linewidth=2)
#    plt.grid(1)
#    plt.legend(['sigmoid','Sigmoid Prime'])
    
        
        
        