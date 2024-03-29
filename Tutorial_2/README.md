# Tutorial 2 - Introduction to Regressor NN
This tutorial is developed for the Offshell Workshop - Wednesday 15:00-16:00.

[Introduction Slides](https://indico.cern.ch/event/1375252/timetable/#13-machine-learning)


## Getting Setup

```
cd Tutorial_2
```

## Imports
The first step is to import the packages we will need to create the NN.
```
# General Utilities
import math
import random
import numpy as np
import uproot

# Keras and TensorFlow
from sklearn.model_selection import cross_val_score, KFold
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from keras.utils import np_utils
from sklearn.metrics import accuracy_score
import tensorflow as tf

# Sklearn Utilities
import sklearn
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

# Plotting Utilities
import matplotlib
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
import matplotlib.pyplot as plt
```
## Initializing Variables
We also need to initialize some variables we want to use in our script.

```
### Initializing Input File Information

INPUT_FILE_NAME = "../../EMTF_ntuple_slimmed_v2.root"

### Initializing Input Variable Information

INPUT_VAR_NAMES = ["theta", "st1_ring2", "dPhi_12", "dPhi_23", "dPhi_34", "dPhi_13", "dPhi_14", "dPhi_24", "FR_1", "bend_1", "dPhiSum4", "dPhiSum4A", "dPhiSum3", "dPhiSum3A", "outStPhi", "dTh_14", "RPC_1", "RPC_2", "RPC_3", "RPC_4"]

### Initializing empty X, Y, W

X = []
Y = []
W = []
```

## Reading Root Files using Uproot
We will be using Uproot to read our input root files. Uproot is very useful because it makes the interface with ML tools much easier since branches are converted to numpy arrays. This makes the process of reading in our data very straight forward - we can simply loop through each file and pulling the input variables directly. One thing to be mindful of is that we need our `X` array to be in the proper format, so it should be that each element in the 2D array should correspond to the input layer but when we get these variables our of the root file using `f["tree/" + var].array()` each element will be for a given variable. Thus, we will need to take the transpose of the `input_var` array.

Example:
```
input_var_directly = [ [pt1, pt2, pt3, pt4 ...],
                       [mass1, mass2, mass3, mass4 ...]
                       [eta1, eta2, eta3, eta4 ...] ]

input_var_transposed  = [ [pt1, mass1, eta1],
                          [pt2, mass2, eta2],
                          [pt3, mass3, eta3],
                          [pt4, mass4, eta4], ...]
```

```

### Reading Root Files and Filling X, Y, W

f = uproot.open(INPUT_FILE_NAME)
input_vars = [f["tree/" + var].array() for var in INPUT_VAR_NAMES]

X = np.transpose(input_vars)
Y = f["tree/GEN_pt"].array()
W = [1]*len(input_vars[0])
```

## Defining the Keras Model

We can create our NN model using Keras. In a function, here called `baseline_model`, we can set how many layers we want, how many nodes in each layer, the type of layer, each layers activiation function, and the model's loss function, optimizer algorithm, and metrics.

```

### Defining Baseline Model

def baseline_model():
    model = Sequential() # 20, 60, 30, 15, 1
    model.add(Dense(60, input_dim=len(INPUT_VAR_NAMES), activation='relu'))
    model.add(Dense(30, activation='relu'))
    model.add(Dense(15, activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))
    # Compile model
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

### Splitting into Training and Testing Sets
X_train, X_test, Y_train, Y_test, W_train, W_test = train_test_split(X, Y, W, test_size=.5, random_state=123)


### Defining our Keras Model
estimator = KerasRegressor(build_fn=baseline_model, epochs=200, batch_size=128, verbose=1)
estimator.fit(np.array(X_train), np.array(Y_train))
```

## Plotting Resolution
Resolution is a good metric to see how well the regressor is performing. The resolution plot is generated by the proportion of error and provides a snapshot into how accurate the regresor is compared to the true value. The resolution plot also shows the expected spread of values and can indicate biases in the training.

```
predictions = estimator.predict(X_test)

resolution = [(Y_test[i] - predictions[i])/Y_test[i] for i in range(0,len(Y_test))]
res_binned, res_bins = np.histogram(resolution, 100, (-2,2))

pdf_pages = PdfPages("./dnn_history_tutorial2.pdf")
fig, ax = plt.subplots(1)
fig.suptitle("Model Resolution")
ax.errorbar([res_bins[i]+(res_bins[i+1]-res_bins[i])/2 for i in range(0, len(res_bins)-1)],
                    res_binned, xerr=[(res_bins[i+1] - res_bins[i])/2 for i in range(0, len(res_bins)-1)],
                    linestyle="", marker=".", markersize=3, elinewidth = .5)
ax.set_ylabel('$N_{events}$')
ax.set_xlabel("$(p_T^{GEN} - p_T^{NN})/(p_T^{GEN})$")
fig.set_size_inches(6,6)


pdf_pages.savefig(fig)
pdf_pages.close()
```


