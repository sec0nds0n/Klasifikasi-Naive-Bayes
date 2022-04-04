import numpy as np
import matplotlib.pyplot as plt

def visualize_classifier(classifier, x, y):
    min_x, max_x = x[:,0].min() - 1.0, x[:,0].max() + 1.0
    min_y, max_y = x[:,1].min() - 1.0, x[:,1].max() + 1.0
    
    #ukuran jarak data arange
    mesh_step_size = 0.01
    #buat distribusi nilai sesuai dengan ukuran sesuai min_x, max_x dan min_y, max_y
    x_vals, y_vals = np.meshgrid(np.arange(min_x, max_x, mesh_step_size),
                                 np.arange(min_y, max_y, mesh_step_size))
    #ravel : 1-D array
    output = classifier.predict(np.c_[x_vals.ravel(), y_vals.ravel()])
    #reshape sesual x_vals
    output = output.reshape(x_vals.shape)
    
    plt.figure()
    # pseudocolor plot. cmap:colormap
    plt.pcolormesh(x_vals, y_vals, output, cmap=plt.cm.Greens)
    # scatter plot. c=list warna. s=shape
    plt.scatter(x[:,0], x[:,1], c=y, s=75, edgecolors='black', linewidths=1, cmap=plt.cm.Paired)
    
    plt.xlim(x_vals.min(), x_vals.max())
    plt.ylim(y_vals.min(), y_vals.max())
    #ketebalan
    plt.xticks((np.arange(int(x[:, 0].min() - 1), int(x[:, 0].max() + 1), 1.0)))
    plt.yticks((np.arange(int(x[:, 1].min() - 1), int(x[:, 1].max() + 1), 1.0)))
    
    plt.show()