
## 项目概述

本项目使用 MATLAB 实现了一个基于 AlexNet 的图像分类模型。通过迁移学习，我们对预训练的 AlexNet 模型进行了微调，以适应特定的数据集。项目包括数据加载、数据预处理、模型训练、模型评估和结果可视化等步骤。

## 目录结构

```
image_classification/
├── dataset/
│   ├── class1/
│   ├── class2/
│   └── ...
├── main.m
└── README.md
```


- `dataset/`: 存放图像数据集，每个子文件夹对应一个类别。
- `main.m`: 主脚本文件，包含完整的图像分类流程。
- `README.md`: 项目说明文档。

## 依赖项

- MATLAB R2021a 或更高版本
- Deep Learning Toolbox
- Image Processing Toolbox

## 运行步骤

1. **准备数据集**：
   - 将图像数据集放入 `dataset/` 文件夹中，每个类别的图像放在对应的子文件夹中。

2. **运行主脚本**：
   - 打开 MATLAB 并导航到项目目录。
   - 运行 `main.m` 脚本。

## 代码详解

### 数据加载

```matlab
% 加载图像数据
imds = imageDatastore('Dataset', 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
```


- 使用 `imageDatastore` 加载图像数据，并从文件夹名称中提取标签。

### 数据分割

```matlab
% 将数据分为训练集、验证集和测试集
[imdsTrain, imdsRemaining] = splitEachLabel(imds, 0.70, 'randomized');
[imdsValidation, imdsTest] = splitEachLabel(imdsRemaining, 0.15, 'randomized');
```


- 将数据集按比例分为训练集（70%）、验证集（15%）和测试集（15%）。

### 数据集大小

```matlab
% 打印数据集大小
disp(['训练集大小: ', num2str(numel(imdsTrain.Labels))]);
disp(['验证集大小: ', num2str(numel(imdsValidation.Labels))]);
disp(['测试集大小: ', num2str(numel(imdsTest.Labels))]);
```


- 打印每个数据集的大小。

### 加载预训练模型

```matlab
% 加载预训练的AlexNet模型
net = alexnet;
```


- 加载预训练的 AlexNet 模型。

### 定义迁移学习层

```matlab
% 定义迁移学习层
inputSize = net.Layers(1).InputSize;
layersTransfer = net.Layers(1:end-3];
numClasses = numel(categories(imdsTrain.Labels));

% 定义新层
layers = [
    layersTransfer
    fullyConnectedLayer(numClasses, 'WeightLearnRateFactor', 20, 'BiasLearnRateFactor', 20)
    softmaxLayer
    classificationLayer];
```


- 移除预训练模型的最后三层，并添加新的全连接层、softmax 层和分类层。

### 数据增强

```matlab
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
```


- 使用 `imageDataAugmenter` 进行数据增强，包括随机反射和平移。
- 训练数据进行数据增强，验证数据不进行数据增强。

### 定义训练选项

```matlab
% 定义训练选项
options = trainingOptions('adam', ...
    'MiniBatchSize', 4, ...
    'MaxEpochs', 5, ...
    'InitialLearnRate', 1e-4, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', augimdsValidation, ...
    'ValidationFrequency', 3, ...
    'Verbose', false, ...
    'ExecutionEnvironment', 'cpu');
```


- 定义训练选项，包括优化器、批量大小、最大轮数、初始学习率等。

### 训练模型

```matlab
% 训练网络
netTransfer = trainNetwork(augimdsTrain, layers, options);
```


- 使用 `trainNetwork` 训练模型。

### 模型评估

#### 验证集评估

```matlab
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
```


- 对验证集进行分类，并计算准确率、精确度、召回率和 F1 分数。
- 绘制验证集的混淆矩阵。

#### 测试集评估

```matlab
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
```


- 对测试集进行分类，并计算准确率、精确度、召回率和 F1 分数。
- 绘制测试集的混淆矩阵。

### 计算指标

```matlab
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
```


- 计算每个类别的 TP、FP、FN，并计算精确度、召回率和 F1 分数。
- 计算加权平均值，返回总体的精确度、召回率和 F1 分数。

## 结果分析

- **验证集** 和 **测试集** 的准确率、精确度、召回率和 F1 分数将显示在控制台中。
- 混淆矩阵将绘制出来，帮助分析模型在不同类别上的表现。

## 注意事项

- 确保数据集路径正确。
- 根据实际需求调整训练参数，如批量大小、最大轮数等。
- 可以根据需要调整数据增强策略，以提高模型性能。

