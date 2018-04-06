function test_training_example_MATLAB

% Batch-size
batchSize = 32;
% Number of epcohs to train
epochs = 30;

%To read train/validation data
imds = imageDatastore('C:\Users\Public\Documents\MATLAB\forHackathon\cwt_images\train', 'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames', 'FileExtensions', '.jpg');
aug_imds = augmentedImageDatastore([28 28], imds);
imds_test = imageDatastore('C:\Users\Public\Documents\MATLAB\forHackathon\cwt_images\validation', 'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames', 'FileExtensions', '.jpg');
aug_imds_test = augmentedImageDatastore([28 28], imds_test);

%Simple training n/w
layers = [
    imageInputLayer([28 28 3])
    
    convolution2dLayer(3,8,'Padding',1)
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,16,'Padding',1)
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,32,'Padding',1)
    batchNormalizationLayer
    reluLayer
    
    fullyConnectedLayer(2)
    softmaxLayer
    classificationLayer];

validationFrequency = floor(length(imds.Files)/batchSize);

% Specify training options
opts = trainingOptions('sgdm', ...
    'InitialLearnRate', 0.0004, ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropFactor', 0.1, ...
    'LearnRateDropPeriod', 20, ...
    'L2Regularization', 0.0001, ...
    'MaxEpochs', epochs, ...
    'MiniBatchSize', batchSize, ...
    'ValidationData', aug_imds_test, ...
    'ValidationFrequency', validationFrequency, ...
    'ValidationPatience', Inf, ...
    'Plots','training-progress', ...
    'Verbose', true, 'ExecutionEnvironment', 'gpu');

%Training
[net, info] = trainNetwork(aug_imds, layers, opts)
