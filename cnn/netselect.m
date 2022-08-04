function layers = netselect(net,numclass)
%%輸入模型類型以及類別數量，輸出已做遷移學習的模型

    switch net
        case "alexnet"
            net = alexnet;
            layers = net.Layers;
            layers(end-2) = fullyConnectedLayer(numclass); 
            layers(end) = classificationLayer;
        case "googlenet"
            net = googlenet;
            lgraph = layerGraph(net);
            newFCLayer = fullyConnectedLayer(numclass,'Name','new_fc');
            lgraph = replaceLayer(lgraph,'loss3-classifier',newFCLayer);
            newClassLayer = softmaxLayer('Name','new_softmax');
            lgraph = replaceLayer(lgraph,'prob',newClassLayer);
            newClassLayer = classificationLayer('Name','new_classoutput');
            lgraph = replaceLayer(lgraph,'output',newClassLayer);
            layers = lgraph;
        case "vgg19"
            net = vgg19;
            layers = net.Layers;
            layers(end-2) = fullyConnectedLayer(numclass); 
            layers(end) = classificationLayer;
        case "squeezenet"
            net = squeezenet;
            lgraph = layerGraph(net);
            newFCLayer = fullyConnectedLayer(numClasses,'Name','new_fc','WeightLearnRateFactor',10,'BiasLearnRateFactor',10);
            lgraph = replaceLayer(lgraph,'pool10',newFCLayer);
            newClassLayer = softmaxLayer('Name','new_softmax');
            lgraph = replaceLayer(lgraph,'prob',newClassLayer);
            newClassLayer = classificationLayer('Name','new_classoutput');
            lgraph = replaceLayer(lgraph,'ClassificationLayer_predictions',newClassLayer);
            layers = lgraph;
        otherwise
            warning("Unexpected net type.")
    end
end

