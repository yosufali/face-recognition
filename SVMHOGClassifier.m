
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

% Call the function to create and train the SVM Classifier using HOG
% features
SVMClassifierWithHOGFeatures(trainingSet, testingSet);

%% Extract Histogram Of Oriented Gradient Features and train an SVM Classifier with them
function SVMClassifierWithHOGFeatures( trainingSet, testingSet )
    
    % Extract HOG Features for training
    [trainingFeatures, trainingLabel, ~, ~] = ExtractHisOfGradFeatures(trainingSet);
    
    % Train a 37 class SVM classifier
    SVMHogModel = fitcecoc(trainingFeatures,trainingLabel);
    
    % Extract HOG Features for testing
    [testingfeatures, ~, testingLabels, setSize] = ExtractHisOfGradFeatures(testingSet);
    
    % Predict the labels for the test set
    testLabels = predict(SVMHogModel, testingfeatures);

    % Check accuracy by seeing how many predictions were correct
    noOfRightMatches = 0;
    for i=1:setSize
        if strcmp(testLabels{i}, testingLabels(i,:))
           noOfRightMatches = noOfRightMatches + 1;
        end
    end
    accuracy = noOfRightMatches/setSize;
    
    % Uncomment line below to generate the .mat file for this classifier
    % save SVMHOGClassifier SVMHogModel accuracy;

end

%% Extract Histogram of Gradient Features
function [features, trainingLabels, testingLabels, setSize] = ExtractHisOfGradFeatures(dataSet)
    % Extract the Histogram of Gradient Features for the given data set
    
    dataSets = numel(dataSet);
    % Number of images, needed only for testing
    setSize = sum([dataSet.Count]);
    featureCount = 1;

    % Extract HOG features for the given data set
    % Return labels for testing and training data sets seperately
    for i=1:dataSets
        label = dataSet(i).Description;
        for j = 1:dataSet(i).Count
            features(featureCount,:) = extractHOGFeatures(read(dataSet(i),j));
            trainingLabels{featureCount} = label;
            testingLabels(featureCount, :) = label;
            featureCount = featureCount + 1;
        end
    end
end
