The network_config file contains the following:

The first number (denoted I for the rest of the document) is the number of input nodes.
The next number (denoted H for the rest of the document) denotes the number of hidden layers in the network.
Each of the H numbers after denotes the number of nodes in each hidden layer.
The next number (also the (H+3)th number) in the file and denotes the number of nodes in the output layer. This number is also denoted as O for the rest of the document.
Each of these numbers are positive integers.

The last 2 doubles denote the minimum and maximum doubles that the weights can be, respectively



The training file contains the following:

The first number is a positive integer that denotes the maximum amount of times (iterations) the network will be trained.
The second number is a positive double that denotes the initial lambda value, which is the initial learning factor
The third number is a positive double that denotes the error threshold, which denotes the minimum error that would end training
The fourth number is a double that denotes the factor that lambda changes by during training.


The targetInOut file contains the following:

The first number contains 1 positive integer (denoted C for the rest of the document) that denotes the number of training cases there are
The next C*I doubles denote the inputs for each training case.
The first group of I doubles makes up the first training case. The following group of I doubles makes up the next training case. This continues until the last training case.
The next (and final) C*O doubles denote the target outputs for each training case.
Similarly to the inputs, The first group of O doubles makes up the first training case. The following group of O doubles makes up the next training case. This continues until the last training case.
