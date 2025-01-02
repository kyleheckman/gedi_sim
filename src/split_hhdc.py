import numpy as np
import os
import argparse

def get_new_hhdc(hhdc):
    new = []
    for i in range(2):
        for j in range(2):
            tmp = hhdc[:, 32*i:32*(i+1), 32*i:32*(i+1)]
            new.append(tmp)
    return new

if __name__ == '__main__':
    parser = argparse.ArgumentParser
    parser.add_argument('-d', '--data', type=str, required=True)
    parser.add_argument('-o', '--output', type=str, required=True)

    args = parser.parse_args
    data_dir = args.data
    output_dir = args.output

    entries = os.listdir(data_dir)

    for entry in entries:
        hhdc = np.load({data_dir}/{entry})
        new_hhdc = get_new_hhdc(hhdc)

        for i in range(len(new_hhdc)):
            tmp = new_hhdc[i]
            count = np.count_nonzero(tmp, axis=0)
            count[count != 0] = 1
            count = np.sum(count)

            if count < 768:
                continue

            new_fn = f'{output_dir}/{entry.split('.')[0]}_sub-block_{i}.npy'
            np.save(new_fn, tmp)