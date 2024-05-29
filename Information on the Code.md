Welcome to my githup repository for my master thesis.

The documents hold the code for the networks I built.
Regarding the dataset and Hyperparameters used small changes have to be applied in the code when using it:

-Individual circuits have to be build as presented in the thesis. As no changes in the network are done, circuit changes like number and kind of rotations are to be changed individually.
Use the right .py data for this as building parallel circuits with the single qubit code will need way more adaptions than just using the fitting parallel blueprint.

-This holds aswell when changing between pooled and unpooled networks as the number of filters in the network architecture is different and has to be adapted.

-Switching between MNIST and breastmnist changes the number of classes which has therefore to be changed also in the network scheme so the outputs generated for crossentropy measurement fit the number of labels.
