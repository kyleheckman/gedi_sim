import os
import numpy as np
import matplotlib.pyplot as plt

def display_2d(x, x_hat):
    for i in range(x.shape[0]):
        fig, axes = plt.subplots(1,2,figsize=(2,1))
        axes[0].imshow(x[i])
        axes[1].imshow(x_hat[i])
        plt.show()
        plt.close()

def display_3d(x, x_hat):
    fig = plt.figure()
    axes = fig.add_subplot(1,2,1, projection='3d')
    axes.scatter(x[:,0:1], x[:,1:2], x[:,2:3])
    # axes = fig.add_subplot(1,2,2, projection='3d')
    # axes.scatter(x_hat)
    plt.show()
    plt.close()

def get_pairs(fn_list):
    pairs = []
    for i in range(0, len(fn_list)//2):
        x = None
        x_hat = None
        for fn in fn_list:
            x = fn if str(i) in fn and 'original' in fn else x
            x_hat = fn if str(i) in fn and 'reconstructed' in fn else x_hat
        pairs.append((x, x_hat))
    return pairs

if __name__ == '__main__':
    data_path = '../samples'
    fn_list = os.listdir(data_path)

    pairs = get_pairs(fn_list)
    print(pairs)
    for pair in pairs[1:]:
        print(pair)
        x = np.load(f'{data_path}/{pair[0]}')
        x_hat = np.load(f'{data_path}/{pair[1]}')
        print(f'Min: {np.min(x)} | Max: {np.max(x)} | Mean: {np.mean(x)}')
        print(f'Min: {np.min(x_hat)} | Max: {np.max(x_hat)} | Mean: {np.mean(x_hat)}')

        display_2d(x, x_hat)
