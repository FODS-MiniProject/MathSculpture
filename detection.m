
clear;
clc;
clear all;
% Set the data directory
dataDir = "/MATLAB Drive/Matlab_project/emotion_data";
doTraining = false;

% Check if the directory exists
if ~exist(dataDir, 'dir')
    error('Data directory "%s" not found.', dataDir);
end

% Emotion categories (must match subfolder names)
emotions = {'happy', 'sad', 'angry', 'surprise', 'neutral', 'fear', 'disgust'};
numEmotions = numel(emotions);
imageSize = [227 227 3];

% Load AlexNet
try
    net = alexnet;
catch
    error('AlexNet not found. Install Deep Learning Toolbox and AlexNet support.');
end

try
    imdsTrain = imageDatastore(fullfile(dataDir, 'train'), ...
        'IncludeSubfolders', true, ...
        'LabelSource', 'foldernames');
    imdsTrain.ReadFcn = @(filename) repmat(imresize(imread(filename), imageSize(1:2)), [1 1 3]);

    % 1. Class distribution
    figure;
    labelCount = countEachLabel(imdsTrain);
    bar(labelCount.Count);
    xticklabels(labelCount.Label);
    xlabel('Emotion');
    ylabel('Image Count');
    title('Class Distribution (Training Set)');

    % 2. Sample images
    figure;
    samplesPerClass = 5;
    for i = 1:length(emotions)
        idx = find(imdsTrain.Labels == emotions{i});
        for j = 1:min(samplesPerClass, numel(idx))
            subplot(length(emotions), samplesPerClass, (i-1)*samplesPerClass + j);
            imshow(readimage(imdsTrain, idx(j)));
            if j == 1
                ylabel(emotions{i});
            end
        end
    end
    sgtitle('Sample Images from Training Data');

numImagesToRead = min(1500, numel(imdsTrain.Files));  
imgSizes = zeros(numImagesToRead, 2);
channels = zeros(numImagesToRead, 1);

for i = 1:numImagesToRead
    img = imread(imdsTrain.Files{i});
    imgSizes(i,:) = size(img, 1:2);
    channels(i) = size(img, 3);
end

    figure;
    histogram(imgSizes(:,1));
    xlabel('Image Height');
    ylabel('Count');
    title('Image Height Distribution');

    figure;
    histogram(channels);
    xticks([1 3]);
    xticklabels({'Grayscale', 'RGB'});
    xlabel('Color Channels');
    ylabel('Count');
    title('Image Color Mode');

catch ME
    warning('EDA skipped: %s', ME.message);
end

%1.Prepare Data

if doTraining
    try
        imdsTrain = imageDatastore(fullfile(dataDir, 'train'), ...
            'IncludeSubfolders', true, ...
            'LabelSource', 'foldernames');

        imdsValidation = imageDatastore(fullfile(dataDir, 'test'), ...
            'IncludeSubfolders', true, ...
            'LabelSource', 'foldernames');

        % Read and resize images, convert grayscale to RGB if needed
        imdsTrain.ReadFcn = @(filename) repmat(imresize(imread(filename), imageSize(1:2)), [1 1 3]);
        imdsValidation.ReadFcn = @(filename) repmat(imresize(imread(filename), imageSize(1:2)), [1 1 3]);

        % Data augmentation
        augmenter = imageDataAugmenter( ...
            'RandRotation', [-20 20], ...
            'RandXTranslation', [-10 10], ...
            'RandYTranslation', [-10 10], ...
            'RandXScale', [0.9 1.1], ...
            'RandYScale', [0.9 1.1], ...
            'RandXReflection', true);

        datasourceTrain = augmentedImageDatastore(imageSize, imdsTrain, ...
            'DataAugmentation', augmenter);

        datasourceValidation = augmentedImageDatastore(imageSize, imdsValidation);
    catch ME
        error('Error preparing training data: %s', ME.message);
    end
end

%2.Modify Alexnet

if doTraining
    try
        layers = net.Layers;
        layers(1) = imageInputLayer(imageSize, 'Name', 'input', 'Normalization', 'none');
        layers(end-2:end) = [
            fullyConnectedLayer(numEmotions, 'Name', 'fc_emotion')
            softmaxLayer('Name', 'softmax')
            classificationLayer('Name', 'classOutput')];
    catch ME
        error('Error modifying network: %s', ME.message);
    end
end

% 3.Train the Network
if doTraining
    try
        options = trainingOptions('sgdm', ...
            'MiniBatchSize', 64, ...
            'MaxEpochs', 10, ...
            'InitialLearnRate', 0.001, ...
            'ValidationData', datasourceValidation, ...
            'ValidationFrequency', 30, ...
            'Verbose', false, ...
            'Plots', 'training-progress');

        net = trainNetwork(datasourceTrain, layers, options);
        save('emotion_detection_network.mat', 'net');
    catch ME
        error('Error training network: %s', ME.message);
    end
else
    try
        load('emotion_detection_network.mat', 'net');
    % 3.Train the Network
if doTraining
    try
        options = trainingOptions('sgdm', ...
            'MiniBatchSize', 64, ...
            'MaxEpochs', 10, ...
            'InitialLearnRate', 0.001, ...
            'ValidationData', datasourceValidation, ...
            'ValidationFrequency', 30, ...
            'Verbose', false, ...
            'Plots', 'training-progress');

        net = trainNetwork(datasourceTrain, layers, options);
        save('emotion_detection_network.mat', 'net');
    catch ME
        error('Error training network: %s', ME.message);
    end
else
    try
        load('emotion_detection_network.mat', 'net');
        % Load validation datastore for evaluation
    imdsValidation = imageDatastore(fullfile(dataDir, 'test'), ...
        'IncludeSubfolders', true, ...
        'LabelSource', 'foldernames');
    imdsValidation.ReadFcn = @(filename) repmat(imresize(imread(filename), imageSize(1:2)), [1 1 3]);
    datasourceValidation = augmentedImageDatastore(imageSize, imdsValidation);
catch ME
    error('Failed to load trained model: %s', ME.message);
end
end

disp('Evaluating model on validation set...');
predictedLabels = classify(net, datasourceValidation);
trueLabels = imdsValidation.Labels;

% Compute and display accuracy
accuracy = mean(predictedLabels == trueLabels);
fprintf('Validation Accuracy: %.2f%%\n', accuracy * 100);

% Plot confusion matrix
figure;
confusionchart(trueLabels, predictedLabels, ...
'Title', 'Confusion Matrix (Validation Set)', ...
'RowSummary', 'row-normalized', ...
'ColumnSummary', 'column-normalized');
%4.Real time Emotion Detection
try
    cam = webcam();
    cam.Resolution = '640x480';
catch ME
    error('Webcam error: %s', ME.message);
end
faceDetector = vision.CascadeObjectDetector;
figure;
h = gcf;
set(h, 'Visible', 'on');
try
    while ishandle(h)
        frame = snapshot(cam);
        faces = faceDetector.step(frame);
        if ~isempty(faces)
            for i = 1:size(faces, 1)
                face = faces(i, :);
                croppedFace = imcrop(frame, face);
                if size(croppedFace, 3) == 1
                    resizedFace = imresize(repmat(croppedFace, [1 1 3]), imageSize(1:2));
                else
                    resizedFace = imresize(croppedFace, imageSize(1:2));
                end
                label = classify(net, resizedFace);
                position = [face(1), face(2) - 15];
                if position(2) < 1
                    position(2) = face(2) + face(4) + 15;
                end
                % --- MODIFICATION HERE ---
                frame = insertObjectAnnotation(frame, 'rectangle', face, ...
                    char(label), 'Color', 'white', ... % Changed 'TextBoxColor' to 'Color'
                    'FontSize', 14, 'TextColor', 'black');
                % -------------------------
            end
        else
            frame = insertText(frame, [10 10], 'No faces detected', ...
                'FontSize', 14, 'TextColor', 'red');
        end
        imshow(frame);
        fps = 30;
delay = 1 / fps;
pause(delay);

    end
    clear cam;
catch ME
    clear cam;
    error('Real-time detection error: %s', ME.message);
end