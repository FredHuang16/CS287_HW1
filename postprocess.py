
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

print(dataset)

# get the files
with h5py.File("output."+dataset+".hdf5", "r") as f:
    alpha = np.array(f["alpha"])
    accuracy = np.array(f["accuracy"])
    y_hat = np.array(f["output"],dtype = np.int32)

plt.plot(alpha,accuracy)
plt.show()

# save the output
np.savetxt("results."+dataset+".csv",y_hat,"%d")
