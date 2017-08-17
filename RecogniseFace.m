function [ P ] = RecogniseFace( I, featureType, classifierName )
    % This script is the main starting point for attempting to classify a
    % new image
    
    % A matric of Nx3 where the three is [ id, x, y]
    % This will get populated each time a face is found in the image I
    % provided
    P = [];
    
    % Read in the image and convert it to a usable format
    Image = imread(I);
    I = im2uint8(Image);
    
    % This uses the CascadeObjectDetector wchich in turn uses the
    % Viola-Jones algorithm
    FaceDetector = vision.CascadeObjectDetector();
    % Increase the merge threshold from the default of 4 to avoid false 
    % positive face detection 
    FaceDetector.MergeThreshold = 7;
    bbox = step(FaceDetector, I);
    N = size(bbox, 1);
    
    % Loops through the image I provided, using the bounding box created 
    % above to find faces
    for i=1:N
        faceNum = 1;
        % Extract the ith face
        a = bbox(i, 1);
        b = bbox(i, 2);
        c = a+bbox(i, 3);
        d = b+bbox(i, 4);
        F = I(b:d, a:c, :);
        % create a directory to store the found face once it has been
        % cropped
        mkdir found; 
        filename = strcat('found/' ,num2str(faceNum),'.jpg');
        % resize the face to match the dimensions of those used in training
        F = imresize(F, [200,200]);
        imwrite(F, filename);
        
        % Use the (a,c) and (b, d) coordinates of the bounding box to 
        % determine the central face region (x and y coordinates) 
        % of the person detected
        x = (a + c)/2;
        y = (b + d)/2;
        
        % Call the relevent classifier/model based on which classifier and 
        % feature type combination is chosen by the user
        % This chosen classifer returns what it beleives to be the matching
        % ID for the current face
        
        % SVM using Bag of Features
        if isequal(classifierName, 'SVM') && isequal(featureType, 'BAG')
            load SVMBAGClassifier.mat;
            [id, ~] = predict(SVMBagModel, F);
            
        % SVM using History of Gradients Features
        elseif isequal(classifierName, 'SVM') && isequal(featureType, 'HOG')
            load SVMHOGClassifier.mat;
            features = extractHOGFeatures(F);
            id = predict(SVMHogModel, features);
            id = str2num(id{1});
            
        % Feedforward Neural Network (MLP) using Bag of Features 
        elseif isequal(classifierName, 'FNN') && isequal(featureType, 'BAG')
            load FNNBAGClassifier.mat;
            features = encode(bag, F).';
            results = net(features);
            [~, id] = max(results(:,1));
            
        % Feedforeward Neural Network (MLP) using History of Gradients
        % Features 
        elseif isequal(classifierName, 'FNN') && isequal(featureType, 'HOG')
            load FNNHOGClassifier.mat
            features = extractHOGFeatures(F);
            featuresTranspose = features';
            results = net(featuresTranspose);
            [~, id] = max(results(:,1));
        % If you do not enter a correct combination of classifier and
        % feature type, a message will be displayed in the console and P
        % will be returned empty
        else
            disp('Please choose either "BAG" or "HOG" as your feature type, and "SVM" or "FNN" as you classifier');
            return
        end

        % Add the found coordinates for this current face and it's ID to 
        % the P matrix, storing the values as integers instead of doubles
        % floating points
        P = [P; int32(id), int32(x), int32(y)];     
    end
end
