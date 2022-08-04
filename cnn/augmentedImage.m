function augimds = augmentedImage(net,imgs,pixelrange,scalerange,rotationrange)
%%依順序輸入模型類型，像素移動數值，縮放大小數值，旋轉度數數值，輸出擴增後資料集

%網路模型選擇
   
     switch net
        case "alexnet"
            size = [227 227];
        case "googlenet"
            size = [224 224];
        case "vgg19"
            size = [224 224];
        case "squeezenet"
            size = [227 227];
        otherwise
            warning("Unexpected net type.")
    end

%擴增參數匯入

imageAugmenter = imageDataAugmenter( ...
    'RandXReflection',true, ...
    'RandYReflection',true, ...
    'RandRotation',rotationrange, ...
    'RandXTranslation',pixelrange, ...
    'RandYTranslation',pixelrange, ...
    'RandXScale',scalerange, ...
    'RandYScale',scalerange);

%執行擴增和調整資料集

augimds = augmentedImageDatastore(size,imgs,'ColorPreprocessing','gray2rgb','DataAugmentation',imageAugmenter);




end