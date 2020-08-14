clc; clear all;
%% ................................................
% Load the training data into memory
Nu = 5; % Number of users
% NOTE: Please place the vein pattern data (provided) in a folder and
% provide its full path below.
veinPatternDatasetPath = fullfile('PD2');
veinData = imageDatastore(veinPatternDatasetPath, ...
        'IncludeSubfolders',true,'LabelSource','foldernames');
%[xTrainImages,tTrain] = digitTrainCellArrayData;
trainingNumFiles = 300;
 rng(1) % For reproducibility
[trainveinData,testveinData] = splitEachLabel(veinData, ...
				trainingNumFiles,'randomize');
for i = 1: numel(trainveinData.Files)
    x = imread(trainveinData.Files{i});
    x= imresize(x,[50 50]);
    xTrain{i} = x;   
    trainLabel(i) = trainveinData.Labels(i);
    clear x
end
 for j = 1: numel(testveinData.Files)
    x = imread(testveinData.Files{j});
    x= imresize(x,[50 50]);
    xTest{j} = x;    
    testLabel(j) = testveinData.Labels(j);
    clear x
 end
 % Genralized transformation to matrix form of Labels - i.e., one-hot encoding
    trainLabel = double(trainLabel);
    testLabel =  double(testLabel);
Lb_Tr = zeros(Nu,numel(trainveinData.Labels));
Lb_Ts = zeros(Nu,numel(testveinData.Labels));
for i = 1 : Nu
   id = find(trainLabel ==i) ;
   Lb_Tr(i,id)=1;
   id = find(testLabel ==i);
   Lb_Ts(i,id)=1;
end

%....trainveinData= input training data
%....testveinData= test vein Data
%% ..............................................
% Display some of the training images
clf
for i = 1:20
    subplot(4,5,i);
    imshow(xTrain{i}); % Each image of size 50x50
end
%% ..............................................
%Neural networks have weights randomly initialized before training. Therefore the results from training are different each time. To avoid this behavior, explicitly set the random number generator seed.
tic
rng('default')
%Set the size of the hidden layer for the autoencoder. For the autoencoder that you are going to train, it is a good idea to make this smaller than the input size.
hiddenSize1 =1200; % Hidden size set to (50*50)/2
autoenc1 = trainAutoencoder(xTrain,hiddenSize1, ...
    'MaxEpochs',400, ...
    'L2WeightRegularization',0.04, ...
    'SparsityRegularization',4, ...
    'SparsityProportion',0.15, ...
    'ScaleData', true,...
    'UseGpu', true,...
    'EncoderTransferFunction','satlin',...
    'DecoderTransferFunction','purelin');
view(autoenc1)
figure()
plotWeights(autoenc1);
feat1 = encode(autoenc1,xTrain);
%%
%Training the second autoencoder
hiddenSize2 = 200;
autoenc2 = trainAutoencoder(feat1,hiddenSize2, ...
    'MaxEpochs',100, ...
    'L2WeightRegularization',0.02, ...
    'SparsityRegularization',4, ...
    'SparsityProportion',0.1, ...
    'ScaleData', true,...
    'EncoderTransferFunction','satlin',...
    'DecoderTransferFunction','purelin');
view(autoenc2)
feat2 = encode(autoenc2,feat1);
%%
softnet = trainSoftmaxLayer(feat2,Lb_Tr,'MaxEpochs',400);
view(softnet)
%%
%As was explained, the encoders from the autoencoders have been used to extract features. You can stack the encoders from the autoencoders together with the softmax layer to form a deep network.
deepnet = stack(autoenc1,autoenc2,softnet);
toc
%You can view a diagram of the stacked network with the view function. The network is formed by the encoders from the autoencoders and the softmax layer.
view(deepnet)
%With the full deep network formed, you can compute the results on the test set. To use images with the stacked network, you have to reshape the test images into a matrix. You can do this by stacking the columns of an image to form a vector, and then forming a matrix from these vectors.
% Get the number of pixels in each image
imageWidth = 50;
imageHeight = 50;
inputSize = imageWidth*imageHeight;
%%
% Load the test images
%[xTestImages,tTest] = digitTestCellArrayData;
%%
% Turn the test images into vectors and put them in a matrix
xTest1 = zeros(inputSize,numel(xTest));
for i = 1:numel(xTest)
    xTest1(:,i) = xTest{i}(:);
end
%%
%You can visualize the results with a confusion matrix. The numbers in the bottom right-hand square of the matrix give the overall accuracy.
y = deepnet(xTest1);
plotconfusion(Lb_Ts,y);


%%
% Fine tunning the deep neural network
% The results for the deep neural network can be improved by performing the
% backpropagation on the whole multilayer network. This process is often
% referred to as fine tuning.
% You fine tune the network by retraining it on the training data in a
% supervised fashion. Befor you can do this, you have to reshape the
% training images into a matrix, as was done for the test images.

% Turn the training images into vectors and put them in a matrix
xTrain1 = zeros(inputSize, numel(xTrain));
for i = 1: numel(xTrain)
    xTrain1(:,i) = xTrain{i}(:); %(:) converts matric data to a vector (col);
end

% Perform fine tuning
deepnet2  = train(deepnet,xTrain1,Lb_Tr);

% You can now view the new results using confusion matrix
%%
tic
y = deepnet2(xTest1);
toc
%%
figure
plotconfusion(Lb_Ts,y);
print('BarPlot','-dpng')