%clear
%% GET FILENAMES
Files=dir('Data/*.*');
Files = Files(3:length(Files));

%% PARAMETERS
numofBlobs = 20;
histBins = 20;
numofImages = 8*length(Files);

%% PREALOCATE ARRAYS OF FEATURES
images = zeros(numofImages,200,200);
targets = strings(numofImages,1);
pointScale = zeros(numofImages,numofBlobs);
pointOrientation = zeros(numofImages,numofBlobs);
pointOctave = zeros(numofImages,numofBlobs);
pointLayer = zeros(numofImages,numofBlobs);
pointLocation = zeros(numofImages,numofBlobs,2);
pointMetric = zeros(numofImages,numofBlobs);
pointLocs =  zeros(numofImages,numofBlobs,2);
stdLocs = zeros(numofImages,2);
imhistos = zeros(numofImages,histBins);
stdhisto = zeros(numofImages);

%% LOAD IMAGES AND CALCULATE FEATURES
for k=1:(numofImages/8)
    % LOAD IMAGE
    fileName = Files(k).folder + "/"+Files(k).name;
    img = mat2gray(imread(fileName));
    targets(8*k - 7:8*k) = [Files(k).name(1:2); Files(k).name(1:2) ;Files(k).name(1:2) ; Files(k).name(1:2) ;Files(k).name(1:2) ;...
        Files(k).name(1:2); Files(k).name(1:2); Files(k).name(1:2)];
    
    images(k,:,:) = img;

    [imhistos(8*k-7,:),stdhisto(8*k-7),pointLocs(8*k-7,:,:),pointScale(8*k-7,:),pointOrientation(8*k-7,:),...
    pointOctave(8*k-7,:),pointLayer(8*k-7,:),pointLocation(8*k-7,:,:),pointMetric(8*k-7,:),...
    stdLocs(8*k-7,:)] = getImageFeatures(img,numofBlobs,histBins);

    images(8*k-7+1,:,:) = imrotate(img,90);
    J1 = imrotate(img,90);
    
    [imhistos(8*k-7+1,:),stdhisto(8*k-7+1),pointLocs(8*k-7+1,:,:),pointScale(8*k-7+1,:),pointOrientation(8*k-7+1,:),...
    pointOctave(8*k-7+1,:),pointLayer(8*k-7+1,:),pointLocation(8*k-7+1,:,:),pointMetric(8*k-7+1,:),...
    stdLocs(8*k-7+1,:)] = getImageFeatures(imrotate(img,90),numofBlobs,histBins);

    images(8*k-7+2,:,:) = imrotate(img,180);
    J2 = imrotate(img,180);

    [imhistos(8*k-7+2,:),stdhisto(8*k-7+2),pointLocs(8*k-7+2,:,:),pointScale(8*k-7+2,:),pointOrientation(8*k-7+2,:),...
    pointOctave(8*k-7+2,:),pointLayer(8*k-7+2,:),pointLocation(8*k-7+2,:,:),pointMetric(8*k-7+2,:),...
    stdLocs(8*k-7+2,:)] = getImageFeatures(imrotate(img,180),numofBlobs,histBins);

    images(8*k-7+3,:,:) = imrotate(img,270);
    J3 = imrotate(img,270);

    [imhistos(8*k-7+3,:),stdhisto(8*k-7+3),pointLocs(8*k-7+3,:,:),pointScale(8*k-7+3,:),pointOrientation(8*k-7+3,:),...
    pointOctave(8*k-7+3,:),pointLayer(8*k-7+3,:),pointLocation(8*k-7+3,:,:),pointMetric(8*k-7+3,:),...
    stdLocs(8*k-7+3,:)] = getImageFeatures(imrotate(img,270),numofBlobs,histBins);

    images(8*k-7+4,:,:) = J3';

    [imhistos(8*k-7+4,:),stdhisto(8*k-7+4),pointLocs(8*k-7+4,:,:),pointScale(8*k-7+4,:),pointOrientation(8*k-7+4,:),...
    pointOctave(8*k-7+4,:),pointLayer(8*k-7+4,:),pointLocation(8*k-7+4,:,:),pointMetric(8*k-7+4,:),...
    stdLocs(8*k-7+4,:)] = getImageFeatures(J3',numofBlobs,histBins);

    images(8*k-7+5,:,:) = J2';

    [imhistos(8*k-7+5,:),stdhisto(8*k-7+5),pointLocs(8*k-7+5,:,:),pointScale(8*k-7+5,:),pointOrientation(8*k-7+5,:),...
    pointOctave(8*k-7+5,:),pointLayer(8*k-7+5,:),pointLocation(8*k-7+5,:,:),pointMetric(8*k-7+5,:),...
    stdLocs(8*k-7+5,:)] = getImageFeatures(J2',numofBlobs,histBins);

    images(8*k-7+6,:,:) = J1';

    [imhistos(8*k-7+6,:),stdhisto(8*k-7+6),pointLocs(8*k-7+6,:,:),pointScale(8*k-7+6,:),pointOrientation(8*k-7+6,:),...
    pointOctave(8*k-7+6,:),pointLayer(8*k-7+6,:),pointLocation(8*k-7+6,:,:),pointMetric(8*k-7+6,:),...
    stdLocs(8*k-7+6,:)] = getImageFeatures(J1',numofBlobs,histBins);

    images(8*k-7+7,:,:) = img';

    [imhistos(8*k-7+7,:),stdhisto(8*k-7+7),pointLocs(8*k-7+7,:,:),pointScale(8*k-7+7,:),pointOrientation(8*k-7+7,:),...
    pointOctave(8*k-7+7,:),pointLayer(8*k-7+7,:),pointLocation(8*k-7+7,:,:),pointMetric(8*k-7+7,:),...
    stdLocs(8*k-7+7,:)] = getImageFeatures(img',numofBlobs,histBins);

end

%% SHOW EXAMPLE IMAGE
I= squeeze(images(1,:,:));
figure(1)
points = detectSIFTFeatures(I);
points2go = selectUniform(points,numofBlobs,size(I));
imshow(I);
hold on;
plot(points2go)

%% MAKE INPUT OUTPUT DATA
%data = horzcat(pointScale,pointOrientation,pointOctave,pointLayer,pointLocation(:,:,1),pointLocation(:,:,2),pointMetric);
data = horzcat(pointScale,pointMetric,stdLocs(:,1),stdLocs(:,2),imhistos,stdhisto);
data = array2table(data);
data.Target = categorical(targets);

[tInd,vInd,teInd] = dividerand(numofImages/8,0.7,0.1,0.2);
for i=1:length(tInd)
    trainInd(8*i-7:8*i) = 8*tInd(i)-7:8*tInd(i);
end

testInd = zeros(size(teInd));
valInd = zeros(size(vInd));

for i=1:length(vInd)
    valInd(i) = 8*vInd(i);
end
for i=1:length(teInd)
    testInd(i) = 8*teInd(i);
end
xTrain = data(trainInd,:);
xVal = data(valInd,:);
xTest = data(testInd,:);


%% NEURAL NET

Mdl = fitcnet(data,"Target","Standardize",true,"OptimizeHyperparameters", {'Activations','LayerSizes','Lambda'}, ...
    "Verbose",0,"IterationLimit",200,'GradientTolerance',1e-5,  ...
    "HyperparameterOptimizationOptions", struct("AcquisitionFunctionName","expected-improvement", ...
    "MaxObjectiveEvaluations",100,'NumGridDivisions',10,'Holdout',0.2));


%  2.3299e-05    [172     86      4]
%  0.00014904    [298    219    276]
%Mdl = fitcnet(xTrain,"Target","LayerSizes", [298    219    276] ,"Activations",'relu','Standardize',true,"Verbose",1,...
%    'ValidationData',{xTest,[]},'IterationLimit',300,'ValidationPatience',10,'ValidationFrequency',1,Lambda=0.00014904 );

%% EVALUATE PERFORMANCE
testAccuracy = 1 - loss(Mdl,xTest,"Target", ...
    "LossFun","classiferror")
trainAccuracy = 1 - loss(Mdl,xTrain,"Target", ...
    "LossFun","classiferror")

figure(5)
confusionchart(xTest.Target,predict(Mdl,xTest))

function [imhistos,stdhisto,pointLocs,pointScale,pointOrientation,...
    pointOctave,pointLayer,pointLocation,pointMetric,stdLocs] = getImageFeatures(img,numofBlobs,histBins)
    % CHANGE CONTRAST AND EXTRACT HISTOGRAM
    img = adapthisteq(img);
    imhistos = imhist(img,histBins);
    stdhisto = std(imhist(img,histBins));

    % DETECT SIFT FEATURES
    pts = detectSIFTFeatures(img,ContrastThreshold=0.001);

    % KMEANS OF FEATURES
    [~ ,kmeanspts] = kmeans(pts.Location,numofBlobs);
    pointLocs = kmeanspts;

    % SELECT STRONGEST FEATURES AND EXTRACT THEM
    pts2go = pts.selectStrongest(numofBlobs);
    pointScale = pts2go.Scale;
    pointOrientation = pts2go.Orientation;
    pointOctave = pts2go.Octave;
    pointLayer = pts2go.Layer;
    pointLocation = pts2go.Location;
    pointMetric = pts2go.Metric;
    stdLocs = std(pts2go.Location,1,1);

end