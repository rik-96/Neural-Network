# Neural Network class definition
import numpy
import scipy.special
class neuralNetwork():
    #initialize the neural ntework
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        #set no of nodes in each input, output and hidden layers
        self.inodes=inputnodes
        self.hnodes=hiddennodes
        self.onodes=outputnodes

        #set the learning rate
        self.lr=learningrate

        #initialize the weights
        #wih is weight between input and hidden layer
        self.wih=numpy.random.normal(0.0,pow(self.hnodes,-0.5),(self.hnodes,self.inodes))
        #who is the wights betwnn hidden layer and output layer
        self.who=numpy.random.normal(0.0,pow(self.onodes,-0.5),(self.onodes,self.inodes))

        #activation function is sigmoid function
        self.activation_function=lambda x: scipy.special.expit(x)
        pass

    #train the neural network
    def train(self,inputs_list,targets_list):
        inputs=numpy.array(inputs_list,ndmin=2).T
        targets=numpy.array(targets_list,ndmin=2).T

        #signals into hidden layer calculation
        hidden_inputs=numpy.dot(self.wih,inputs)
        #calculate signals emerging from hidden layer
        hidden_outputs=self.activation_function(hidden_inputs)

        #calculate input into final output layer
        final_inputs=numpy.dot(self.who,hidden_outputs)
        #calculate signals emerging from final output layer
        final_outputs=self.activation_function(final_inputs)

        #error = target-actual
        output_errors=targets-final_outputs

        #hidden layer error isoutput_error split y weights and added to hidden nodes
        hidden_errors=numpy.dot(self.who.T,output_errors)

        #update the weights for the links between hidden and output layer
        self.who+=self.lr*numpy.dot((output_errors*final_outputs*(1.0-final_outputs)),numpy.transpose(hidden_outputs))

        #update the weights for the links between input and hidden layers
        self.wih+=self.lr*numpy.dot((hidden_errors*hidden_outputs*(1.0-hidden_outputs)),numpy.transpose(hidden_outputs))
        
        pass

    #query the neural network
    def query(self,inputs_list):
        #convert inputs into 2d array
        inputs=numpy.array(inputs_list,ndmin=2).T

        #calculate signals into hidden layer
        hidden_inputs=numpy.dot(self.wih,inputs)
        #calculate the signals emerging from hidden layer
        hidden_outputs=self.activation_function(hidden_inputs)
        #calculate signals into final layer
        final_inputs=numpy.dot(self.who,hidden_outputs)
        #calculate the signals emerging from final output layer
        final_outputs=self.activation_function(final_inputs)

        return final_outputs
    
        pass
    
