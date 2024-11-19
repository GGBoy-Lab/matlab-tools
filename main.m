% 加载图像数据
imds = imageDatastore('Dataset', 'IncludeSubfolders', true, 'LabelSource', 'foldernames');

% 将数据分为训练集、验证集和测试集
[imdsTrain, imdsRemaining] = splitEachLabel(imds, 0.60, 'randomized');
[imdsValidation, imdsTest] = splitEachLabel(imdsRemaining, 0.20, 'randomized');

% 打印数据集大小
disp(['训练集大小: ', num2str(numel(imdsTrain.Labels))]);
disp(['验证集大小: ', num2str(numel(imdsValidation.Labels))]);
disp(['测试集大小: ', num2str(numel(imdsTest.Labels))]);

% 加载预训练的AlexNet模型
net = alexnet;

% 定义迁移学习层
inputSize = net.Layers(1).InputSize;
layersTransfer = net.Layers(1:end-3);
numClasses = numel(categories(imdsTrain.Labels));

% 定义新层
layers = [
    layersTransfer
    fullyConnectedLayer(numClasses, 'WeightLearnRateFactor', 20, 'BiasLearnRateFactor', 20)
    softmaxLayer
    classificationLayer];

% 创建 imageDataAugmenter 对象，进行随机反射和平移
pixelRange = [-30 30];
imageAugmenter = imageDataAugmenter( ...
    'RandXReflection', true, ...
    'RandXTranslation', pixelRange, ...
    'RandYTranslation', pixelRange);
% 应用颜色抖动
augimdsTrain = augmentedImageDatastore(inputSize(1:2), imdsTrain, ...
    'DataAugmentation', imageAugmenter);
% 验证数据不进行数据增强
augimdsValidation = augmentedImageDatastore(inputSize(1:2), imdsValidation);

% 定义训练选项
options = trainingOptions('adam', ...
    'MiniBatchSize', 4, ...
    'MaxEpochs', 5, ...
    'InitialLearnRate', 1e-4, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', augimdsValidation, ...
    'ValidationFrequency', 3, ...
    'Verbose', false, ...
    'Plots', 'training-progress', ... % 显示训练进度图
    'ExecutionEnvironment', 'cpu');  % 确保在CPU上进行训练

% 训练网络
[netTransfer, info] = trainNetwork(augimdsTrain, layers, options);

% 打印训练精度
% for i = 1:length(info.Epoch)
%     disp(['第 ', num2str(info.Epoch(i)), ' 轮训练精度: ', num2str(info.TrainingAccuracy(i))]);
% end

% 对验证数据进行分类
[YPredValidation, scoresValidation] = classify(netTransfer, augimdsValidation, 'ExecutionEnvironment', 'cpu');
YValidation = imdsValidation.Labels;

% 计算验证集准确率
accuracyValidation = mean(YPredValidation == YValidation);
disp(['验证集准确率: ', num2str(accuracyValidation)]);

% 计算验证集的精确度、召回率和F1分数
[precisionValidation, recallValidation, f1ScoreValidation] = calculate_metrics(YValidation, YPredValidation);
disp(['验证集精确度: ', num2str(precisionValidation)]);
disp(['验证集召回率: ', num2str(recallValidation)]);
disp(['验证集F1分数: ', num2str(f1ScoreValidation)]);

% 计算验证集的混淆矩阵
CValidation = confusionmat(YValidation, YPredValidation);
disp('验证集混淆矩阵:');
disp(CValidation);

% 绘制验证集的混淆矩阵
figure;
confusionchart(CValidation, categories(imdsValidation.Labels));
title('验证集混淆矩阵');

% 测试数据不进行数据增强
augimdsTest = augmentedImageDatastore(inputSize(1:2), imdsTest);

% 对测试数据进行分类
[YPredTest, scoresTest] = classify(netTransfer, augimdsTest, 'ExecutionEnvironment', 'cpu');
YTest = imdsTest.Labels;

% 计算测试集准确率
accuracyTest = mean(YPredTest == YTest);
disp(['测试集准确率: ', num2str(accuracyTest)]);

% 计算测试集的精确度、召回率和F1分数
[precisionTest, recallTest, f1ScoreTest] = calculate_metrics(YTest, YPredTest);
disp(['测试集精确度: ', num2str(precisionTest)]);
disp(['测试集召回率: ', num2str(recallTest)]);
disp(['测试集F1分数: ', num2str(f1ScoreTest)]);

% 计算测试集的混淆矩阵
CTest = confusionmat(YTest, YPredTest);
disp('测试集混淆矩阵:');
disp(CTest);

% 绘制测试集的混淆矩阵
figure;
confusionchart(CTest, categories(imdsTest.Labels));
title('测试集混淆矩阵');

% 计算精确度、召回率和F1分数的函数
function [precision, recall, f1Score] = calculate_metrics(yTrue, yPred)
    % 如果标签不是分类变量，则转换为分类变量
    if ~iscategorical(yTrue)
        yTrue = categorical(yTrue);
    end
    if ~iscategorical(yPred)
        yPred = categorical(yPred);
    end

    % 获取唯一类别
    classes = categories(yTrue);
    numClasses = length(classes);

    % 初始化指标
    precision = zeros(1, numClasses);
    recall = zeros(1, numClasses);
    f1Score = zeros(1, numClasses);

    % 计算每个类别的TP、FP、TN、FN
    for i = 1:numClasses
        TP = sum(yTrue == classes(i) & yPred == classes(i));
        FP = sum(yTrue ~= classes(i) & yPred == classes(i));
        FN = sum(yTrue == classes(i) & yPred ~= classes(i));

        % 计算每个类别的精确度、召回率和F1分数
        if TP + FP > 0
            precision(i) = TP / (TP + FP);
        else
            precision(i) = 0;
        end

        if TP + FN > 0
            recall(i) = TP / (TP + FN);
        else
            recall(i) = 0;
        end

        if precision(i) + recall(i) > 0
            f1Score(i) = 2 * precision(i) * recall(i) / (precision(i) + recall(i));
        else
            f1Score(i) = 0;
        end
    end

    % 计算加权平均值
    classCounts = sum(yTrue == classes', 2);  % 确保正确的维度
    total = sum(classCounts);
    precision = sum(precision .* classCounts / total);
    recall = sum(recall .* classCounts / total);
    f1Score = sum(f1Score .* classCounts / total);
end
