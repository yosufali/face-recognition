
% Load the faces from the data-images directory
% this productes a variable faceDabatase with a 1x37 imageSet structure
faceDatabase = imageSet('data-images','recursive');

% determine the smallest amount of images in a category and use the
% partition method to trim the database, ensuring an equal number of images
% for each person
minFaceDatabaseCount = min([faceDatabase.Count]);
faceDatabase = partition(faceDatabase, minFaceDatabaseCount, 'randomize');

% Split the above datbase into training & test sets
% 80% will be used for training, and 20% for testing
[trainingSet, testingSet] = partition(faceDatabase,[0.8 0.2]);

% Call the function to create and train the FNN Classifier using HOG
% features
FNNClassifierWithHOGFeatures(trainingSet, testingSet);

%% Extract Histogram Of Oriented Gradient Features and train a Feedforward Neural Network with them
function FNNClassifierWithHOGFeatures(trainingSet, testingSet)

    % Extract HOG Features for training
    [trainingFeatures, trainingLabel, ~] = ExtractHisOfGradFeatures(trainingSet);
    
    % Create and train the FNN with 20 hidden neurons
    net = feedforwardnet(20, 'trainscg');
    net = configure(net,trainingFeatures,trainingLabel);
    net = train(net, trainingFeatures, trainingLabel);

    % Extract HOG Features for testing
    [testingfeatures, testingLabelsMatrix, setSize] = ExtractHisOfGradFeatures(testingSet);
    
    results = net(testingfeatures);
    
    % loop through the results to determine best matches, while retrieving
    % the labels
    for i = 1 : setSize
        [~, testingLabels(1,i)] = max(results(:,i));
        finalTestingLabels(i) = find(testingLabelsMatrix(:,i));
    end
    
    % Calculate the accuracy of the model
    accuracy = sum(testingLabels == finalTestingLabels) / setSize;
    
    % Uncomment line below to generate the .mat file for this classifier
    save FNNHOGClassifier net accuracy
end

%% Extract Histogram of Gradient Features
function [features, labels, setSize] = ExtractHisOfGradFeatures(dataSet)
    % Extract the Histogram of Gradient Features for the given data set

    dataSets = numel(dataSet);
    % Number of images, needed only for testing
    setSize = sum([dataSet.Count]);
    featureCount = 1;

    % Extract HOG features and return the labels for the given data set
    for i=1:dataSets
        for j = 1:dataSet(i).Count
            features(:,featureCount) = extractHOGFeatures(read(dataSet(i),j));
            labels(i, featureCount) = 1;
            featureCount = featureCount + 1;
        end
    end
end
