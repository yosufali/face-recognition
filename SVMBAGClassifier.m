
% Load the faces from the data-images directory
% this productes a variable faceDabatase with a 1x37 imageSet structure
faceDatabase = imageSet('data-images','recursive');

% Split the above datbase into training & test sets
% 80% will be used for training, and 20% for testing
[trainingSet, testingSet] = partition(faceDatabase,[0.8 0.2]);

% Call the function to create and train an SVM Classifier using Bag of 
% Features
SVMClassifierWithBagOfFeatures(trainingSet, testingSet);

%% Extract Bag Of Features and train an SVM Multiclass Category Classifier
function SVMClassifierWithBagOfFeatures( trainingSet, testingSet )

    % Generate the bag of visual words
    bag = bagOfFeatures(trainingSet);
    
    % Train the SVM with this bag and the training images
    SVMBagModel = trainImageCategoryClassifier(trainingSet, bag);

    % Produce a confusion matrix for this classifier using the testing
    % images
    confMatrix = evaluate(SVMBagModel, testingSet);
    
    % find the average accuracy from the above evaluation
    accuracy = mean(diag(confMatrix));
    
    % Uncomment line below to generate the .mat file for this classifier
    % save SVMBAGClassifier SVMBagModel bag accuracy
end
