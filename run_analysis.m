clc;
clear all;
close all;
rng(1)

load data_all_env.mat

%Loading raw data and labels
data = all_env_data+rand(size(all_env_data))*1e-8; %a small value added if covariance matrix becomes singular
labels = all_env_labels;

%Running minimum Redundancy Maximum Relevance algorithm on raw data
[idx,scores] = fscmrmr(data,labels);
bar(idx,scores(idx))
xlabel('Feature No.'),ylabel('Predictor Score');

%Splitting training and testing data
P = randperm(length(labels));
training_end = floor(0.8*length(P));
training_data = data(P(1:training_end),:);
training_labels = labels(P(1:training_end),:);
testing_data = data(P(training_end+1:end),:);
testing_labels = labels(P(training_end+1:end),:);

%Fitting a Linear Discriminator Model on raw data and labels
model = fitcdiscr(training_data, training_labels);
predictions = predict(model,testing_data);
disp(sum(predictions == testing_labels)/length(testing_labels))

%Fitting an MLP on raw data and labels
[m,n] = size(training_data);

length_traces = n;

%training_data is split into 80% training and 20% validation data
number_training_traces = floor(.8*length(training_labels));
number_of_epochs = 50;

%trace conditioning for fitting into MATLAB's NN framework
[XTrain, YTrain] = trace_condition(training_data,length_traces,training_labels,1,number_training_traces,1);
[XValidation, YValidation] = trace_condition(training_data,length_traces,training_labels,number_training_traces+1,length(training_labels),1);

%Layers of MLP
layers = [
    imageInputLayer([length_traces 1 1])
    fullyConnectedLayer(100)
    batchNormalizationLayer  
    dropoutLayer(0.1)
    reluLayer
    fullyConnectedLayer(100)
    batchNormalizationLayer
    reluLayer
    fullyConnectedLayer(100)
    batchNormalizationLayer
    reluLayer 
    fullyConnectedLayer(7)
    softmaxLayer
    classificationLayer];

%training options
options = trainingOptions('sgdm', ...
    'Shuffle','every-epoch', ...
    'InitialLearnRate', 0.1, ...
    'MaxEpochs',number_of_epochs, ...
    'L2Regularization',0.0001, ...
    'ValidationData',{XValidation,YValidation}, ...
    'MiniBatchSize', 2048,...
    'ValidationFrequency',10, ...
    'ValidationPatience',Inf, ...
    'Verbose',false,...
    'Plots','training-progress');

%net contains the model
[net, info] = trainNetwork(XTrain,YTrain,layers,options);
%plot_accuracy_loss(info,number_of_epochs);

[r,c] = size(testing_data);
[XTest, YTest] = trace_condition(testing_data,length_traces,testing_labels,1,r,1);
YPred = classify(net,XTest);
test_accuracy = mean(YPred == YTest);
disp(test_accuracy)

%PCA algorithm
training_data_adjust = [];
training_data_mean = mean(training_data);
for i = 1:n
    training_data_adjust(:,i) = training_data(:,i) - training_data_mean(1,i);
end
c = cov(training_data);
[V,D] = eig(c);
D = diag(D);
[Dm,I] = sort(D,'descend');
Vm = V(:,I);
training_data = (Vm'*training_data_adjust')';

testing_data_adjust = [];
for i = 1:n
    testing_data_adjust(:,i) = testing_data(:,i) - training_data_mean(1,i);
end
testing_data = (Vm'*testing_data_adjust')';

%PCA-LDA
model = fitcdiscr(training_data, training_labels,'discrimType','pseudoLinear');
predictions = predict(model,testing_data);
disp(sum(predictions == testing_labels)/length(testing_labels))

%PCA-MLP
[m,n] = size(training_data);

length_traces = n;

number_training_traces = floor(.8*length(training_labels));
number_of_epochs = 50;

[XTrain, YTrain] = trace_condition(training_data,length_traces,training_labels,1,number_training_traces,1);
[XValidation, YValidation] = trace_condition(training_data,length_traces,training_labels,number_training_traces+1,length(training_labels),1);

layers = [
    imageInputLayer([length_traces 1 1])
    fullyConnectedLayer(100)
    batchNormalizationLayer  
    dropoutLayer(0.1)
    reluLayer
    fullyConnectedLayer(100)
    batchNormalizationLayer
    reluLayer
    fullyConnectedLayer(100)
    batchNormalizationLayer
    reluLayer 
    fullyConnectedLayer(7)
    softmaxLayer
    classificationLayer];

options = trainingOptions('sgdm', ...
    'Shuffle','every-epoch', ...
    'InitialLearnRate', 0.1, ...
    'MaxEpochs',number_of_epochs, ...
    'L2Regularization',0.0001, ...
    'ValidationData',{XValidation,YValidation}, ...
    'MiniBatchSize', 2048,...
    'ValidationFrequency',10, ...
    'ValidationPatience',Inf, ...
    'Verbose',false,...
    'Plots','training-progress');

[net, info] = trainNetwork(XTrain,YTrain,layers,options);

[r,c] = size(testing_data);
[XTest, YTest] = trace_condition(testing_data,length_traces,testing_labels,1,r,1);
YPred = classify(net,XTest);
test_accuracy = mean(YPred == YTest);
disp(test_accuracy)