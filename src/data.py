import numpy as np 
from numpy.random import Generator as gen 
from numpy.random import PCG64 as pcg
import matplotlib.pyplot as plt 

seed = 101 

def Generate_dots(a): # tool/arz square
    array = gen(pcg(seed))
    return array.uniform(-a, a, size=(2000, 2))
    

def labeling_circle_boundry(r, data):
    labels = (data[:, 0]**2 + data[:, 1] **2 < r**2).astype(int)
    return labels 


def circle_visualization(X, labels):
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(X[:, 0], X[:, 1], c=labels )
    circle = plt.Circle((0, 0), 1, fill=False, color="red",linewidth=2.5) 
    ax.add_patch(circle)
    ax.set_title("in cirlcle or out of it?")
    ax.set_xlabels("x1")
    ax.set_ylabels("x2")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    plt.show()

    return fig, ax