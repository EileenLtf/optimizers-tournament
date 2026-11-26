import numpy as np 
np.random.seed(3)
#import data 
import optimizers 


# initilization(simple random)
def initialize_parameter(layer_dimension):
    parameters = {}
    L = len(layer_dimension)
    for l in range(1, L):
        parameters["W" + str(l)] = np.random.randn(layer_dimension[l], layer_dimension[l-1])
        parameters["b"+ str(l)] = np.zeros((layer_dimension[l], 1))
    
        assert(parameters["W" + str(l)].shape == (layer_dimension[l], layer_dimension[l-1]))
        assert(parameters["b" + str(l)].shape == (layer_dimension[l], 1))
    return parameters


def activation(A_prev, W, b, activation):
    if activation == "sigmoid":
        Z = np.dot(W, A_prev) + b 
        linear_cache = (A_prev, W, b)
        A  = 1 / (1 + np.exp(-Z))
        activation_cache = (A)
    
    elif activation == "relu":
        Z = np.dot(W, A_prev) + b
        linear_cache = (A_prev, W, b)
        A = np.maximum(0, Z)
        activation_cache = (A)
    
    cache = (linear_cache,activation_cache)
    
    return A, cache


def forward_prop(X, parameters):
    caches = []
    A = X
    L = len(parameters) // 2
    
    for l in range (1,L):
        A_prev = A 
        A, cache = activation(A_prev, parameters["W" + str(l)], parameters["b" + str(l)], "relu")
        caches.append(cache)
        
    last_layer, cache_L = activation (A, parameters["W" + str(L)], parameters["b" + str(L)], "sigmoid")
    caches.append(cache_L)

    return last_layer, caches 


def cost_function(labels, predicts):
    m = labels.shape[0]
    cost = np.sum((labels.T * np.log(predicts)) + ((1-labels.T)*np.log(1- predicts)))
    cost /= -m
    cost = np.squeeze(cost)
    return cost



def backward_prop(last_layer, labels, cache):
    gradients = {}
    L = len(cache)
    m = last_layer.shape[1]
    last_layer = last_layer.reshape(1, -1)
    labels = labels.reshape(1, -1)
    dAL = -np.divide(labels, last_layer) + np.divide((1-labels), (1-last_layer))
    
    #sigmoid layer 
    dZ = dAL * last_layer * (1 - last_layer)
    dA_prevL = np.dot(cache[L-1][0][1].T,  dZ) #cache ((A-prev, w, b), nonlinearity )
    dW = 1/m * np.dot(dZ, cache[L-1][0][0].T)   #A[l-1]
    db =1/m * (np.sum(dZ, axis=1, keepdims=True))
    gradients["dA" + str(L-1)] = dA_prevL
    gradients["dW" + str(L)] = dW
    gradients["db" + str(L)] = db
   
   # relu layers 
    for l in reversed(range(L-1)) : 
        current_cache = cache[l]
        dZ = dA_prevL * (current_cache[1] > 0)
        dA_prevL = np.dot(current_cache[0][1].T, dZ)
        dW = 1/m * np.dot(dZ, current_cache[0][0].T)
        db = 1/m * (np.sum(dZ))
        
        gradients["dA" + str(L-1)] = dA_prevL
        gradients["dW" + str(L)] = dW
        gradients["db" + str(L)] = db
    
    return gradients
   