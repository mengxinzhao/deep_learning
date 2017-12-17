import numpy as np


def sigmoid(x):
    """
    Calculate sigmoid
    """
    return 1 / (1 + np.exp(-x))

def sigmoid_prime(x):
    """
    Calculate sigmoid
    """
    return sigmoid(x) * (1 - sigmoid(x))

x = np.array([0.5, 0.1, -0.2,0.9])
target = np.array([0.6, 0.2])
learnrate = 0.5

weights_input_hidden = np.array([[0.5, -0.6, 0.4 ],
                                 [0.1, -0.2, 0.4 ],
                                 [0.1, 0.7, 0.4],
                                 [0.4, 0.2, 0.1]])

weights_hidden_output = np.array([[0.1, -0.3],
                                [0.2, -0.1],
                                [0.2,0.4]])

#x = np.array([0.5, 0.1, -0.2])
#target = 0.6
#learnrate = 0.5

#weights_input_hidden = np.array([[0.5, -0.6],
#                                 [0.1, -0.2],
#                                 [0.1, 0.7]])

#weights_hidden_output = np.array([0.1, -0.3])

## Forward pass
hidden_layer_input = np.dot(x, weights_input_hidden)


hidden_layer_output = sigmoid(hidden_layer_input)
print("hidden_layer_output",hidden_layer_output)
output_layer_in = np.dot(hidden_layer_output, weights_hidden_output)

output = sigmoid(output_layer_in)

print("output",output)
## Backwards pass
## TODO: Calculate output error
error = target - output 

# TODO: Calculate error term for output layer
output_error_term = error * sigmoid_prime( output_layer_in)
print("output_error_term", output_error_term)
# TODO: Calculate error term for hidden layer

hidden_error_term = np.dot(output_error_term, weights_hidden_output.T) * sigmoid_prime(hidden_layer_input)
print("hidden_error_term",hidden_error_term)
# TODO: Calculate change in weights for hidden layer to output layer
delta_w_h_o = learnrate * hidden_layer_output.reshape(-1,1) *output_error_term 
print("delta_w_h_o", delta_w_h_o)
# TODO: Calculate change in weights for input layer to hidden layer
delta_w_i_h = learnrate * x.reshape(-1,1)* hidden_error_term
#delta_w_i_h = learnrate * hidden_error_term * x[:, None]
print("delta_w_i_h",delta_w_i_h)

print('Change in weights for hidden layer to output layer:')
print(delta_w_h_o)
print('Change in weights for input layer to hidden layer:')
print(delta_w_i_h)

