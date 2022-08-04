%%指定輸入資料集路徑

%訓練資料集路徑
trainPath = " ";

%驗證資料集路徑
validationPath = " ";
%測試資料集路徑
testPath = " ";

%選擇使用的遷移學習
net = "alexnet";


%%匯入資料集
disp('Preparing Dataset')
%匯入訓練資料集
trainImgs = imageDatastore(trainPath,"IncludeSubfolders",true,"LabelSource","foldernames");

%匯入測試資料集
testImgs = imageDatastore(testPath,"IncludeSubfolders",true,"LabelSource","foldernames");
%測試資料集由訓練分開
%[trainImgs,testImgs] = splitEachLabel(trainImgs,0.7,'randomize');

%匯入驗證資料集
%imdsValidation = imageDatastore(validationPath,"IncludeSubfolders",true,"LabelSource","foldernames");
%驗證資料集由訓練分開
[trainImgs,validationImgs] = splitEachLabel(trainImgs,0.7,'randomize');


%%資料集擴增及調整
disp('Dataset Augmentation')
%擴增參數調整
pixelRange = [-30 30];
scaleRange = [0.8 1.2];
RotationRange= [-20 20];

%資料集擴增
augimdsTrain = augmentedImage(net,trainImgs,pixelRange,scaleRange,RotationRange);
augimdsValidation = augmentedImage(net,validationImgs,pixelRange,scaleRange,RotationRange);
augimdsTest = augmentedImage(net,testImgs,pixelRange,scaleRange,RotationRange);


%%訓練模型參數調整
options = trainingOptions("sgdm","InitialLearnRate", 0.0001, ...
    'MaxEpochs',20, ...
    'MiniBatchSize',64, ...
    'Shuffle','every-epoch', ...
    'ValidationData',augimdsValidation, ...
    'ValidationFrequency',50, ...
    'ExecutionEnvironment','gpu', ...
    'Plots','training-progress');

%訓練模型總類數量偵測
disp('Preparing Model')
numClasses = numel(categories(trainImgs.Labels));

%遷移學習
layers = netselect(net,numClasses);

%訓練模型
n_net= trainNetwork(augimdsTrain, layers, options);


%%資料集判斷+混淆矩陣
[ImgsPreds, scrs] = classify(n_net,augimdsTest);
ImgsActual = testImgs.Labels;
numCorrect = nnz(ImgsPreds == ImgsActual);
fracCorrect = numCorrect/numel(ImgsPreds);
plotconfusion(testImgs.Labels,ImgsPreds)
