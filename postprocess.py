
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
parser.add_argument('filename', help="hdf5 formatted file from which to extract data",
                    type=str)
args = parser.parse_args(arguments)
filename = args.filename

# get the files
with h5py.File(filename, "r") as f:
    if "alpha" in f.keys():
        alpha = np.array(f["alpha"])
    if "accuracy" in f.keys():
        accuracy = np.array(f["accuracy"])
    if "trainLoss" in f.keys():
        trainLoss = np.array(f["trainLoss"])

    y_hat = np.array(f["output"],dtype = np.int32)

if "alpha" in locals() and "accuracy" in locals():
    plt.plot(alpha,accuracy)
    plt.xlabel("alpha")
    plt.ylabel("Pct Correct")
    plt.savefig("nbAlphaGS.png")
    plt.show()

if "trainLoss" in locals():
    plt.plot(trainLoss)
    plt.xlabel("iteration")
    plt.ylabel("Loss")
    plt.savefig("loss.png")
    plt.show()
# make IDs
ids = np.arange(len(y_hat)) + 1
output = np.vstack((ids,y_hat)).transpose()

# save the output
np.savetxt(filename+".csv",output,"%d"
            ,header="ID,Category",delimiter=",",comments="")
