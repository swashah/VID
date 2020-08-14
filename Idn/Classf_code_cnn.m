%% Create Simple Deep Learning Network for vein Pattern Classification Classification
%%
% This implementation creates and train a simple convolutional neural
% network for deep learning based classification of vein patterns. We feed the 
% processed vein patterns which are extracted from images taken from Intel
% RealSense D415. See preprocessing code for detailed implementation of
% vein extraction.
clc; clear all; close all;
%% Load and Explore the Image Data
% Load the vein pattern data as an |ImageDatastore| object.
% NOTE: Please place the vein pattern data (provided) in a folder and
% provide its full path below.

veinPatternDatasetPath = fullfile('Processed_Data');

veinData = imageDatastore(veinPatternDatasetPath, ...
        'IncludeSubfolders',true,'LabelSource','foldernames');
    
%% 
% |imageDatastore| function labels the images automatically based on folder
% names and stores the data as an |ImageDatastore| object. An
% |ImageDatastore| object lets you store large image data, including data
% that do not fit in memory, and efficiently read batches of images during
% training of a convolutional neural network.

%% 
% Display some of the images in the datastore. 
figure;
perm = randperm(1800,20);
for i = 1:20
    subplot(4,5,i);
    imshow(veinData.Files{perm(i)});
end

%%
% Check the number of veins patterns of each user. 
CountLabel = veinData.countEachLabel;

%%
% |CountLabel| is a table that contains the labels and the number of
% vein-patterns of each user (i.e., total of 11). 
 

%% 
% You must specify the size of the vein pattern (image) in the input layer of the
% network. Check the size of the first vein pattern in |digitData| .
img = readimage(veinData,1);
size(img)

%%
% Each vein image is 231-by-231-by-1 pixels.

%% Specify Training and Test Sets
% Divide the data into training and test sets, so that each category in the
% training set has 300 images and the test set has the remaining images
% from each label (Note that: We have varied training samples from 100-300 for analysis).
trainingNumFiles = 300;
rng(1) % For reproducibility
[trainveinData,testveinData] = splitEachLabel(veinData, ...
				trainingNumFiles,'randomize'); 

%%
% |splitEachLabel| splits the image files in |digitData| into two new datastores,
% |trainDigitData| and |testDigitData|.  

%% Define the Network Layers
% Define the convolutional neural network architecture. 
layers = [imageInputLayer([231 231 1])
          convolution2dLayer(4,20)
          reluLayer
          maxPooling2dLayer(4,'Stride',4)
          
          convolution2dLayer(5,4, 'Padding', 2)
          reluLayer
          maxPooling2dLayer(3,'Stride',2)
          
          fullyConnectedLayer(1568, 'Name', 'fc1')
          fullyConnectedLayer(784, 'Name', 'fc2')
          fullyConnectedLayer(392, 'Name', 'fc3')
          %fullyConnectedLayer(1960, 'Name', 'fc4')
          %fullyConnectedLayer(980, 'Name', 'fc5')
          %fullyConnectedLayer(490, 'Name', 'fc6')
          fullyConnectedLayer(5, 'Name', 'fc4')
          
          
          softmaxLayer
          classificationLayer()];  

%%
% *Image Input Layer* An imageInputLayer is where
% you specify the image size, which, in this case, is 231-by-231-by-1. These
% numbers correspond to the height, width, and the channel size. The vein
% data consists of gray scale images, hence the channel size (color
% channel) is 1. No need to shuffle the data as |trainNetwork|
% automatically does it at the beginning of the training.

%%
% *Convolutional Layer*
% In the convolutional layer, the first argument is |filterSize|, which is the
% height and width of the filters the training function uses while
% scanning along the images. In C1, the number 4 indicates that
% the filter size is [4,4].  The second argument is the number of
% filters (20), which is the number of neurons that connect to the same region
% of the output. This parameter determines the number of the feature maps.
% Likewise, see values in C2 as well.
% 

%% 
% *ReLU Layer*
% The convolutional layer is followed by a nonlinear activation function.
% 

%%
% *Max-Pooling Layer*
% The convolutional layer (with the activation function) is usually
% followed by a down-sampling operation to reduce the number of parameters
% and as another way of avoiding overfitting. One way of down-sampling is
% max-pooling. This layer returns the maximum
% values of rectangular regions of inputs, specified by the first argument,
% |poolSize|. In P1, the size of the rectangular region is [4,4]. The optional
% argument |Stride| determines the step size the training function takes as
% it scans along the image.  Likewise, see values in P2. This max-pooling layer takes place between the
% convolutional layers when there are multiple of them in the network.

%%
% *Fully Connected Layer*
% The convolutional (and down-sampling) layer  is followed by one 
% fully connected layer. As the name suggests, all neurons in a fully
% connected layer connect to the neurons in the layer previous to it. This
% layer combines all of the features (local information) learned by the
% previous layers across the image to identify the larger patterns. The
% last fully connected layer combines them to classify the images. That is
% why, the OutputSize parameter in the last fully connected layer is equal
% to the number of individuals in the target data (i.e.,max 35).

%%
% *Softmax Layer* 
% The fully connected layer uses the softmax
% activation function for classification. 

%%
% *Classification Layer* 
% The final layer is the classification layer. This layer uses
% the probabilities returned by the softmax activation function for each
% input to assign it to one of the mutually exclusive classes.
% 
%% Specify the Training Options
% After defining the layers (network structure), specify the training
% options. Set the options to default settings for the stochastic gradient
% descent with momentum. Set the maximum number of epochs at 100(an epoch
% is a full training cycle on the whole training data), and start the
% training with an initial learning rate of 0.0001.
options = trainingOptions('sgdm','MaxEpochs',100, ...
	'InitialLearnRate',0.0001);  
%gpuDevice(3)
%% Train the Network Using Training Data
% Train the network you defined in layers, using the training data and the
% training options you defined in the previous steps.
tic
convnet = trainNetwork(trainveinData,layers,options);
toc
%%
% |trainNetwork| displays the hardware it uses for training in the display
% window. It uses a GPU by default if there is one available (requires
% Parallel Computing Toolbox (TM) and a CUDA-enabled GPU with compute
% capability 3.0 and higher). If there is no available GPU, it uses a CPU.
% You can also specify the execution environment using the
% |'ExecutionEnvironment'| name-value pair argument in the call to
% |trainingOptions|.


%% Classify the Images in the Test Data and Compute Accuracy
% Run the trained network on the test set that was not used to train the
% network and predict the image labels (i.e., Vein-Patterns).
tic
[YTest, Scores] = classify(convnet,testveinData);
toc
TTest = testveinData.Labels;

%% 
% Calculate the accuracy. 
accuracy = sum(YTest == TTest)/numel(TTest)   

%%
% Accuracy is the ratio of the number of true labels in the test data
% matching the classifications from classify, to the number of images in
% the test data ...

% Confusion Matrix and ROC curve can be plotted using TTest and YTest ...
plotconfusion(TTest,YTest) ...

% Plot RoC ...

T = double(TTest); % TTest are labels of test data ...
Lb_Ts = zeros(5, numel(TTest)); % Matrix to hold test label in one-hot format--- 35 is evaluated group size, to be changed in accordingly...
for i = 1:5
    id = find(T==i);
    Lb_Ts(i,id)=1;
end
Scores = Scores'; % To match dimension with Lb_Ts for plotting ROC
plotroc(Lb_Ts, Scores);
% print('roc', '-dpng') % save the ROC in .png format and name it 'roc'