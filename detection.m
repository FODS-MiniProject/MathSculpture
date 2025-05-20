clear;
clc;
close all;

% Set the data directory
dataDir = "/MATLAB Drive/Matlab_project/emotion_data";

% Toggle training or load pre-trained network
doTraining = true;

% Check if the directory exists
if ~exist(dataDir, 'dir')
    error('Data directory "%s" not found.', dataDir);
end

% Emotion categories (match subfolder names)
emotions = {'happy', 'sad', 'angry', 'surprise', 'neutral', 'fear', 'disgust'};
numEmotions = numel(emotions);

% Image size for AlexNet
imageSize = [227 227 3];

% Load AlexNet
try
    net = alexnet;
catch
    error('AlexNet not found. Install Deep Learning Toolbox and AlexNet support.');
end

% ======================================================================
% 1. Prepare Data
% ======================================================================
if doTraining
    try
        imdsTrain = imageDatastore(fullfile(dataDir, 'train'), ...
            'IncludeSubfolders', true, ...
            'LabelSource', 'foldernames');

        imdsValidation = imageDatastore(fullfile(dataDir, 'test'), ...
            'IncludeSubfolders', true, ...
            'LabelSource', 'foldernames');

        % Convert grayscale to RGB
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

% ======================================================================
% 2. Modify AlexNet
% ======================================================================
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

% ======================================================================
% 3. Train the Network
% ======================================================================
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
    catch ME
        error('Failed to load trained model: %s', ME.message);
    end
end

% ======================================================================
% 4. Real-Time Emotion Detection
% ======================================================================
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

                % Convert grayscale to RGB if needed
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

                frame = insertObjectAnnotation(frame, 'rectangle', face, ...
                    char(label), 'TextBoxColor', 'white', ...
                    'FontSize', 14, 'TextColor', 'black');
            end
        else
            frame = insertText(frame, [10 10], 'No faces detected', ...
                'FontSize', 14, 'TextColor', 'red');
        end

        imshow(frame);
        pause(0.1);
    end
    clear cam;
catch ME
    clear cam;
    error('Real-time detection error: %s', ME.message);
end
