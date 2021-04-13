import torch;
import torchvision;



# The Neural Network class!
class Neural_Network(torch.nn.Module):
    def __init__(self, num_hidden_layers = 3, nodes_per_layer = 20):
        # Note: we assume that num_hidden_layers and nodes_per_layer are
        # positive integers.

        # Call the superclass initializer.
        super(Neural_Network, self).__init__();

        # Define Layers ModuleList.
        self.Layers = torch.nn.ModuleList();

        # Append the first hidden layer. The domain of this layer is the input
        # domain, which means that its in_features is different from its
        # out_features
        self.Layers.append(
            torch.nn.Linear(
                in_features = 28*28,
                out_features = nodes_per_layer,
                bias = True
            )
        );

        # Now append the rest of the hidden layers. Each of these layers maps
        # within the same space, which means thatin_features = out_features.
        for i in range(1, num_hidden_layers):
            self.Layers.append(
                torch.nn.Linear(
                    in_features = nodes_per_layer,
                    out_features = nodes_per_layer,
                    bias = True
                )
            );

        # Now, append the Output Layer, which has nodes_per_layer input
        # features, but only 10 output features.
        self.Layers.append(
            torch.nn.Linear(
                in_features = nodes_per_layer,
                out_features = 10,
                bias = True
            )
        );

        # Set the number of layers
        self.Num_Layers = num_hidden_layers + 1;

        # Initialize the weight matricies in the network.
        for i in range(self.Num_Layers):
            torch.nn.init.xavier_uniform_(self.Layers[i].weight);

        # Finally, set the activation function, and Softmax (to turn the output
        # of the final layer into a probability distribution).
        self.Activation_Function = torch.nn.Tanh();
        self.Softmax = torch.nn.Softmax(0);


    def forward(self, x):
        # Note: we assume that x is a FLATTENED 28x28 tensor.

        # Pass x through the layers.
        for i in range(self.Num_Layers - 1):
            x = self.Activation_Function(self.Layers[i](x));

        # Pass through the last layer and apply softmax.
        x = self.Softmax(self.Layers[self.Num_Layers - 1](x));

        # Return the final result.
        return x;



# Data loaders
def Load_Data(batch_size):
    # Set up the training dataset, dataloader
    training_dataset = torchvision.datasets.FashionMNIST(
        root = "data",
        train = True,
        download = True,
        transform = torchvision.transforms.ToTensor()
    );
    training_dataloader = torch.utils.data.DataLoader(training_dataset, batch_size = batch_size, shuffle = True);

    # Set up the test dataset.
    testing_dataset = torchvision.datasets.FashionMNIST(
        root = "data",
        train = False,
        download = True,
        transform = torchvision.transforms.ToTensor()
    );

    return training_dataloader, testing_dataset;



# Loss function
def Loss_Function(prediction, target_index):
    return -torch.log(prediction[target_index]);



# Test loop function.
def Training_Loop(test_dataloader, Network, optimizer):
    # Iterate over the batches
    for samples in test_dataloader:
        # The “samples” returned by test_dataloader consist of two variables:
        # images and the targets.
        images, targets = samples;
        batch_size = images.size()[0];

        # Zero out the network gradients.
        optimizer.zero_grad();

        # Evaluate the network for the images in this batch
        for i in range(batch_size):
            # We think of images as a batch_size array of 1x28x28 tensors.
            # Thus, image[i] gets the ith element of this array, which is a
            # 1x28x28 tensor. We then use the flatten method to turn this into
            # a 1 dimensional 28*28 tensor, which we can feed into the network!
            image = images[i].flatten();

            # Now extract the target index (the value of this target)
            target_index = targets[i].item();

            # Determine the prediction, loss
            prediction = Network.forward(image);
            sample_loss = Loss_Function(prediction, target_index);

            # Perform backward automatic differentiation.
            sample_loss.backward();

        # Now that we've accumulated a batch's worth of gradients, perform back
        # propagation using our optimizer.
        optimizer.step();



# Testing loop.
def Test_Loop(test_dataset, Network):
    # Initialize test_loss, number of items to iterate over.
    test_loss = 0;
    test_dataset_size = len(test_dataset);

    # initialize number correct
    correct = 0;

    for i in range(test_dataset_size):
        # Get input, format it in a way that the network understands.
        input, target_index = test_dataset[i];
        input = input.flatten();

        # Compute the network’s prediction for this input.
        prediction = Network.forward(input);

        # aggregate loss
        test_loss += Loss_Function(prediction, target_index).item();

        if(prediction.argmax().item() == target_index):
            correct += 1;

    print("average test loss = " + str(test_loss/test_dataset_size));
    print("percent correct   = " + str(correct/test_dataset_size));
    print("\n", end = '');


# Main function.
def main():
    # Set up network hyperparameters.
    epochs = 10;
    learning_rate = .0015;
    batch_size = 64;

    # Set up the training dataloader, testing data set
    training_dataloader, testing_dataset = Load_Data(batch_size);

    # Set up our network.
    Network = Neural_Network(num_hidden_layers = 3, nodes_per_layer = 40);

    # Set up an optimizer.
    optimizer = torch.optim.Adam(Network.parameters(), lr = learning_rate);

    # Now, loop through the epochs.
    for t in range(epochs):
        print("Epoch #" + str(t));

        # Run Training loop.
        Training_Loop(training_dataloader, Network, optimizer);

        # Run testing loop.
        Test_Loop(testing_dataset, Network);

    # All done!
    print("done!");

    # Save the network.
    torch.save(Network.state_dict(), "./Network_State_Dict")

if __name__ == '__main__':
    main();
