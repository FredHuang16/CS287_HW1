
#!/usr/bin/env python

"""Text Classification Preprocessing
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse
import h5py
import sys

if __name__ == "__main__":
    arguments = sys.argv[1:]

# get the name of the dataset that was passed in
parser = argparse.ArgumentParser(
    description=__doc__,
    formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('dataset', help="Data set",
                    type=str)
args = parser.parse_args(arguments)
dataset = args.dataset

# get the files
with h5py.File("output."+dataset+".hdf5", "r") as f:
    if "alpha" in f.keys():
        alpha = np.array(f["alpha"])
    if "accuracy" in f.keys():
        accuracy = np.array(f["accuracy"])

    y_hat = np.array(f["output"],dtype = np.int32)

if "alpha" in locals() and "accuracy" in locals():
    plt.plot(alpha,accuracy)
    plt.xlabel("alpha")
    plt.ylabel("Pct Correct")
    plt.show()

# make IDs
ids = np.arange(len(y_hat)) + 1
output = np.vstack((ids,y_hat)).transpose()

# save the output
np.savetxt("results."+dataset+".csv",output,"%d"
            ,header="ID,Category",delimiter=",",comments="")
