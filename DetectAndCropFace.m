
% This script take a directory of images and does three things:
% 1) Detects any faces within that images
% 2) If if it finds a face, it crops it
% 3) If the face is not a 200x200 size image, its makes it so
% 4) It saves the face as an image in a new '/newImages' directory
% The images are numbered from 1 to N, where N is the total number of
% faces found in the original directory


% Choose the directory where the original images are stored,
contents = dir('Individual1/');
name = 1;
% Loop through the images withing the given directory
for i = 3:numel(contents)
    filename = contents(i).name;
    imagePath = strcat('Individual1/', filename);
    I = imread(imagePath);
    
    % This uses the CascadeObjectDetector wchich in turn uses the
    % Viola-Jones algorithm
    FaceDetector = vision.CascadeObjectDetector();
    FaceDetector.MergeThreshold = 6; % Increased from the default of 4 to avoid false positive detection

    bbox = step(FaceDetector, I);
    N = size(bbox, 1);
    
    % Loops through the image I provided, using the bounding box created 
    % above to find faces
    for i=1:N
        % Extract the ith face
        a = bbox(i, 1);
        b = bbox(i, 2);
        c = a+bbox(i, 3);
        d = b+bbox(i, 4);
        F = I(b:d, a:c, :);
        % create a directory to store the found faces once they have been
        % cropped
        mkdir newImages;
        filename = strcat('newImages/' ,num2str(name),'.jpg');
        % resize the face to match the dimensions of those used in training
        if size(F,1) ~= 200
            F = imresize(F, [200 200]);
        end
        imwrite(F, filename);
        name = name + 1;
    end
end

% The faces are then manually grouped into labelled directories of faces 
% that match thier own, within a new directory called 'data-images/', ready for
% classification
