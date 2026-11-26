def Gradient_decent(parameters, gradinets, learning_rate):
    L = len(parameters) // 2
    for l in range(L):
        parameters["W " + str(l+1)] -= (learning_rate * gradinets["W" + str(l+1)])
        parameters["b" + str(l+1)] -= (learning_rate * gradinets["b" + str(l+1)])

    return parameters  
