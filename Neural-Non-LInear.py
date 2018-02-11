from numpy import exp, array, random, dot

class Smartthinking():

    def __init__(self):
        #random.seed()

        #Creating a (3 input x 1 output) matrix with random values in between (-1 to 1)
        self.weights = random.random((3, 1))

    # Adjusting the weights each time.
    def training(self, input_set, output_set, iterations):
        for value in range(iterations):
            output = self.think(input_set)

            # Calculating the difference
            difference = output_set - output
            #print(difference)
            
            # This means inputs, which are zero, do not cause changes to the weights.
            adjustment = dot(input_set.T, difference)
            #print(adjustment) 

            # Adjust the weights.
            self.weights += adjustment

    def think(self, inputs):
        # Pass inputs to the neural.
        print(inputs, self.weights)
        matrix_multiplication = dot(inputs, self.weights) 
        return matrix_multiplication

if __name__ == "__main__":

    smart_thinking = Smartthinking()

print ("Random initial weights: ",smart_thinking.weights)

input_set = array([[1,0,0], [0,1,0], [1,0,0], [1,1,0]])
output_set = array([[1,0,1,1]]).T

smart_thinking.training(input_set, output_set, 50000)

print ("New weighting factor: ")
print (smart_thinking.weights)

# Test the neural network with a new situation.
print ("Testing other set of [1, 0, 0]:")
print (smart_thinking.think(array([0, 0, 0])))

