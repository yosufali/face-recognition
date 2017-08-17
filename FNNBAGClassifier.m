
% Load the faces from the data-images directory
% this productes a variable faceDabatase with a 1x37 imageSet structure
faceDatabase = imageSet('data-images','recursive');

% Split the above datbase into training & test sets
% 80% will be used for training, and 20% for testing
[trainingSet, testingSet] = partition(faceDatabase,[0.8 0.2]);

% Call the function to create and train an FNN Classifier using Bag of
% Features
FNNClassifierWithBagOfFeatures(trainingSet, testingSet);

%% Extract Bag Of Features and train a Feedforward Neural Network with them
function FNNClassifierWithBagOfFeatures( trainingSet, testingSet )

    % Generate the bag of visual words
    bag = bagOfFeatures(trainingSet);
    
    % Extract the bag of features for training use
    [trainingFeatures, trainingLabelsMatrix, ~] = ExtractBAGFeatures(trainingSet, bag);
    
    % Create and train the FNN with 20 hidden neurons
    net = feedforwardnet(20, 'trainscg');
    net = configure(net, trainingFeatures, trainingLabelsMatrix);
    net = train(net, trainingFeatures, trainingLabelsMatrix);
    
    % Extract Bag of features for testing use
    [testingFeatures, testinglabelsMatrix, setSize] = ExtractBAGFeatures(testingSet, bag);
    
    % Make predictions for images in the testing set
    results = net(testingFeatures);
    
    % loop through the results to determine best matches, while retrieving
    % the labels
    for i = 1 : setSize
        [~, testingLabels(1,i)] = max(results(:,i));
        finalTestingLabels(i) = find(testinglabelsMatrix(:,i));
    end

    % Calculate the accuracy of the model
    accuracy = sum(testingLabels == finalTestingLabels) / setSize;
    
    % Uncomment to generate a .mat file
    % save FNNBAGClassifier net bag accuracy
end

%% Extract Bag of Features
function [features, labelsMatrix, setSize] = ExtractBAGFeatures(dataSet, bag)
    % Extract the Bag of features for the given data set
    
    dataSets = numel(dataSet);         
    % Number of images, needed only for testing
    setSize = sum([dataSet.Count]);
    % Create the matrix for the labels
    labelsMatrix = zeros(dataSets, setSize);
    featureCount = 1;
    % Generate histogram of features vector
    features = encode(bag, dataSet).';
    % determine the labels for images in the given set
    for i=1:dataSets
        for j=1:dataSet(i).Count
            labelsMatrix(i, featureCount) = 1;
            featureCount = featureCount + 1;
        end
    end
end
