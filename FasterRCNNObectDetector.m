 % training data saved to workspace as 'datasetTruthCopy'

options = trainingOptions('sgdm', ...
      'MiniBatchSize', 1, ...
      'InitialLearnRate', 1e-3, ...
      'MaxEpochs', 5, ...
      'VerboseFrequency', 200, ...
      'CheckpointPath', tempdir);

rng(0);
shuffledIdx = randperm(height(datasetTruthCopy));
idx = floor(0.6 * height(datasetTruthCopy));
trainingData = datasetTruthCopy(shuffledIdx(1:idx),:);
testData = datasetTruthCopy(shuffledIdx(idx+1:end),:);
 [detector, info] = trainFasterRCNNObjectDetector(datasetTruthCopy, 'resnet50', options, ...
        'NegativeOverlapRange', [0 0.3], ...
        'PositiveOverlapRange', [0.6 1]);
numImages = height(testData);
    results = table('Size',[numImages 3],...
        'VariableTypes',{'cell','cell','cell'},...
        'VariableNames',{'Boxes','Scores','Labels'});
    
    % Run detector on each image in the test set and collect results.
    for i = 1:numImages
        
        % Read the image.
        I = imread(testData.imageFilename{i});
        
        % Run the detector.
        [bboxes, scores, labels] = detect(detector, I);
        
        % Collect the results.
        % Collect the results.
        results.Boxes{i} = bboxes;
        results.Scores{i} = scores;
        results.Labels{i} = labels;
    end
    
    % Extract expected bounding box locations from test data.
expectedResults = testData(:, 2:end);

% Evaluate the object detector using Average Precision metric.
[ap, recall, precision] = evaluateDetectionPrecision(results, expectedResults);


% Plot precision/recall curve
figure
plot(recall{4,:},precision{4,:})
xlabel('Recall')
ylabel('Precision')
grid on
title(sprintf('Average Precision = %.2f', ap))
