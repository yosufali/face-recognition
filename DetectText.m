% I = imread('keypad.jpg');
I = imread('Individual1/IMG_3092.JPG');
% IG = rgb2gray(I);
% IGB=IG>210;
% 
% results = ocr(IGB, 'TextLayout', 'Block');
% 
% results.Words
% 
I = rgb2gray(im2uint8(I));
IGB = I > 210;
blobAnalyzer = vision.BlobAnalysis('MaximumCount', 500);
[area, centroids, roi] = step(blobAnalyzer, IGB);
for i = 2 : size(area,1)
    %roi = results.Words{i}
    wordBBox = roi(i,:)
    % Show the location of the word in the original image
    if wordBBox(3) > 300
        figure;
        hold
        Iname = insertObjectAnnotation(I, 'rectangle', wordBBox,i);
        imshow(Iname);
        wordBBoxInterest = wordBBox;
    end
end
box = I(wordBBoxInterest(2):wordBBoxInterest(2)+wordBBoxInterest(4),wordBBoxInterest(1):wordBBoxInterest(1)+wordBBoxInterest(3));
figure;
resultsNew = ocr(roi, 'TextLayout', 'Block');
imshow(I);
text = deblank( {resultsNew.Text(1:2)} );
img  = insertObjectAnnotation(I, 'rectangle', wordBBoxInterest, text);
figure;

% Turn this into a function afterwards, that takes in images as 
% a parameter and is called by DetectAndCrop, passing in the 
% image currently being processed
% it should then returns the digit that was found as text

