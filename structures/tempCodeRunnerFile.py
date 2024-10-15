# Update weights / biases
    #       self.weights -= learning_rate * d_L_d_w.T
    #       self.biases -= learning_rate * d_L_d_b
    #       return d_L_d_inputs.reshape(self.last_input_shape)