clear all; close all; clc;
addpath(genpath('../my_data'));
%% Load in data
%supercell = {};
%supermatrix = [];
%for q=1:50
cell_in = {0 0 0 ; 0 0 0 ; 0 0 0; 0 0 0};
cell_out = {0 0 0 ; 0 0 0 ; 0 0 0; 0 0 0};

for phase=1:4
    for m=1:3 %1 = levelground, 2 = ramp, 3 = stair
        if m == 1
            mode = 'levelground';
        elseif m == 2
            mode = 'ramp';
        elseif m == 3
            mode = 'stair';
        end
        in_file = strcat(mode, '/full_ab21_input_', num2str(phase*2), '.mat');
        out_file = strcat(mode, '/full_ab21_output_', num2str(phase*2), '.mat');
        cell_in(phase,m) = struct2cell(load(in_file));
        cell_out(phase,m) = struct2cell(load(out_file));
    end
end

%% Data processing
clear phase m mode;
cell_mode_testing = {0 0 0 0};
cell_mode_training = {0 0 0 0};

for phase=1:4
    for m=1:3 %1 = levelground, 2 = ramp, 3 = stair
        in_table = cell_in{phase,m};
        out_table = cell_out{phase,m};
        if m == 1
            walk_mask = strcmp(out_table.labels_feat_last,'walk');
            in_table_walk = in_table(walk_mask,:);
            table_size = size(in_table_walk,1);
            classification = ones(table_size,1);
            in_table_class = table(classification);
            in_table_walk = [in_table_walk in_table_class];
            walk_training_vals = randsample(table_size,ceil(0.8.*table_size));
            training_mask = logical(zeros(table_size,1));
            for i = 1:length(walk_training_vals)
                index = walk_training_vals(i);
                training_mask(index) = 1;
            end
            in_table_walk_training = in_table_walk(training_mask,:);
            in_table_walk_testing = in_table_walk(~training_mask,:);
            clear training_mask classification;
        elseif m == 2
            ramp_asc_mask = strcmp(out_table.labels_feat_last,'rampascent');
            in_table_ramp_asc = in_table(ramp_asc_mask,:);
            table_size = size(in_table_ramp_asc,1);
            classification = 2.*ones(table_size,1);
            in_table_class = table(classification);
            in_table_ramp_asc = [in_table_ramp_asc in_table_class];
            ramp_asc_training_vals = randsample(table_size,ceil(0.8.*table_size));
            training_mask = logical(zeros(table_size,1));
            for i = 1:length(ramp_asc_training_vals)
                index = ramp_asc_training_vals(i);
                training_mask(index) = 1;
            end
            in_table_ramp_asc_training = in_table_ramp_asc(training_mask,:);
            in_table_ramp_asc_testing = in_table_ramp_asc(~training_mask,:);
            clear training_mask classification;
            %%%%%
            ramp_desc_mask = strcmp(out_table.labels_feat_last,'rampdescent');
            in_table_ramp_desc = in_table(ramp_desc_mask,:);
            table_size = size(in_table_ramp_desc,1);
            classification = 3.*ones(table_size,1);
            in_table_class = table(classification);
            in_table_ramp_desc = [in_table_ramp_desc in_table_class];
            ramp_desc_training_vals = randsample(table_size,ceil(0.8.*table_size));
            training_mask = logical(zeros(table_size,1));
            for i = 1:length(ramp_desc_training_vals)
                index = ramp_desc_training_vals(i);
                training_mask(index) = 1;
            end
            in_table_ramp_desc_training = in_table_ramp_desc(training_mask,:);
            in_table_ramp_desc_testing = in_table_ramp_desc(~training_mask,:);
            clear training_mask classification;
        elseif m == 3
            stair_asc_mask = strcmp(out_table.labels_feat_last,'stairascent');
            in_table_stair_asc = in_table(stair_asc_mask,:);
            table_size = size(in_table_stair_asc,1);
            classification = 4.*ones(table_size,1);
            in_table_class = table(classification);
            in_table_stair_asc = [in_table_stair_asc in_table_class];
            stair_asc_training_vals = randsample(table_size,ceil(0.8.*table_size));
            training_mask = logical(zeros(table_size,1));
            for i = 1:length(stair_asc_training_vals)
                index = stair_asc_training_vals(i);
                training_mask(index) = 1;
            end
            in_table_stair_asc_training = in_table_stair_asc(training_mask,:);
            in_table_stair_asc_testing = in_table_stair_asc(~training_mask,:);
            clear training_mask classification;
            %%%%%
            stair_desc_mask = strcmp(out_table.labels_feat_last,'stairdescent');
            in_table_stair_desc = in_table(stair_desc_mask,:);
            table_size = size(in_table_stair_desc,1);
            classification = 5.*ones(table_size,1);
            in_table_class = table(classification);
            in_table_stair_desc = [in_table_stair_desc in_table_class];
            stair_desc_training_vals = randsample(table_size,ceil(0.8.*table_size));
            training_mask = logical(zeros(table_size,1));
            for i = 1:length(stair_desc_training_vals)
                index = stair_desc_training_vals(i);
                training_mask(index) = 1;
            end
            in_table_stair_desc_training = in_table_stair_desc(training_mask,:);
            in_table_stair_desc_testing = in_table_stair_desc(~training_mask,:);
            clear training_mask classification;
        end
    end
    in_table_training = [in_table_walk_training;in_table_ramp_asc_training;in_table_ramp_desc_training;in_table_stair_asc_training;in_table_stair_desc_training];
    phase_class = 2.*phase.*ones(size(in_table_training,1),1);
    phase_table = table(phase_class);
    in_table_training = [in_table_training phase_table];
    cell_mode_training{phase} = in_table_training;
    in_table_testing = [in_table_walk_testing;in_table_ramp_asc_testing;in_table_ramp_desc_testing;in_table_stair_asc_testing;in_table_stair_desc_testing];
    phase_class = 2.*phase.*ones(size(in_table_testing,1),1);
    phase_table = table(phase_class);
    in_table_testing = [in_table_testing phase_table];
    cell_mode_testing{phase} = in_table_testing;
end

%training_set = [cell_mode_training{1};cell_mode_training{2};cell_mode_training{3};cell_mode_training{4}];
%testing_set = [cell_mode_testing{1};cell_mode_testing{2};cell_mode_testing{3};cell_mode_testing{4}];

training_set = cell_mode_training{2};
testing_set = cell_mode_testing{2};

training_matrix = table2array(training_set);
testing_matrix = table2array(testing_set);
training_matrix(:,end-1:end)=[];
testing_matrix(:,end-1:end)=[];
training_matrix(:,1)=[];
testing_matrix(:,1)=[];

nn_training_matrix = [training_matrix;testing_matrix];

classifiers = training_set.classification;
test_class = testing_set.classification;
nn_class = [classifiers;test_class];
nn_classifiers = zeros(length(nn_class),5);
for k = 1:length(nn_class)
    if nn_class(k)==1
        nn_classifiers(k,1)=1;
    elseif nn_class(k)==2
        nn_classifiers(k,2)=1;
    elseif nn_class(k)==3
        nn_classifiers(k,3)=1;
    elseif nn_class(k)==4
        nn_classifiers(k,4)=1;
    elseif nn_class(k)==5
        nn_classifiers(k,5)=1;
    end
end
cell_table = {};
cell_test = {};
for j = 1:length(classifiers)
    if classifiers(j)==1
        cell_table(j,1)={'walking'};
    elseif classifiers(j)==2
        cell_table(j,1)={'ramp_ascent'};
    elseif classifiers(j)==3
        cell_table(j,1)={'ramp_descent'};
    elseif classifiers(j)==4
        cell_table(j,1)={'stair_ascent'};
    elseif classifiers(j)==5
        cell_table(j,1)={'stair_descent'};
    end
end
for k = 1:length(test_class)
    if test_class(k)==1
        cell_test(k,1)={'walking'};
    elseif test_class(k)==2
        cell_test(k,1)={'ramp_ascent'};
    elseif test_class(k)==3
        cell_test(k,1)={'ramp_descent'};
    elseif test_class(k)==4
        cell_test(k,1)={'stair_ascent'};
    elseif test_class(k)==5
        cell_test(k,1)={'stair_descent'};
    end
end
class_table = table(cell_table);

%%
%ITERATION 1
end_set = training_set(:,137:end-2);
training_set(:,137:end) = [];
training_set(:,1) = [];
training_set = [end_set training_set];
acc_matrix=[];
acc_matrix_placeholder=[];
for i = 1:5:251
    if contains(training_set.Properties.VariableNames{i},'feat_a_1')
        model = fitcdiscr(training_set(:,i:i+9), class_table);
        predictions = predict(model,testing_set);
        conf_mat = confusionmat(cell_test,predictions);
        accuracy = (conf_mat(1,1)+conf_mat(2,2)+conf_mat(3,3)+conf_mat(4,4)+conf_mat(5,5))./sum(sum(conf_mat));
        acc_matrix= [acc_matrix accuracy];
        acc_matrix_placeholder=[acc_matrix_placeholder accuracy];
    elseif contains(training_set.Properties.VariableNames{i},'feat_a_6')
        acc_matrix_placeholder = [acc_matrix_placeholder 0];
    else
        model = fitcdiscr(training_set(:,i:i+4), class_table);
        predictions = predict(model,testing_set);
        conf_mat = confusionmat(cell_test,predictions);
        accuracy = (conf_mat(1,1)+conf_mat(2,2)+conf_mat(3,3)+conf_mat(4,4)+conf_mat(5,5))./sum(sum(conf_mat));
        acc_matrix= [acc_matrix accuracy];
        acc_matrix_placeholder=[acc_matrix_placeholder accuracy];
    end
	%confusionchart(cell_test,predictions,'Title','','RowSummary','row-normalized','ColumnSummary','column-normalized')
end
%figure(1)
%plot(acc_matrix)
[max1,sensor_index] = max(acc_matrix_placeholder);
if contains(training_set.Properties.VariableNames{sensor_index.*5-4},'feat_a_1')
    sensor1 = training_set(:,sensor_index.*5-4:sensor_index.*5+5);
    training_set(:,sensor_index.*5-4:sensor_index.*5+5)=[];
    flag1 = 10;
else
    sensor1 = training_set(:,sensor_index.*5-4:sensor_index.*5);
    training_set(:,sensor_index.*5-4:sensor_index.*5)=[];
    flag1 = 5;
end

%ITERATION 2
acc_matrix=[];
acc_matrix_placeholder=[];
for i = 1:5:251-flag1
    if contains(training_set.Properties.VariableNames{i},'feat_a_1')
        model = fitcdiscr([sensor1 training_set(:,i:i+9)], class_table);
        predictions = predict(model,testing_set);
        conf_mat = confusionmat(cell_test,predictions);
        accuracy = (conf_mat(1,1)+conf_mat(2,2)+conf_mat(3,3)+conf_mat(4,4)+conf_mat(5,5))./sum(sum(conf_mat));
        acc_matrix= [acc_matrix accuracy];
        acc_matrix_placeholder=[acc_matrix_placeholder accuracy];
    elseif contains(training_set.Properties.VariableNames{i},'feat_a_6')
        acc_matrix_placeholder = [acc_matrix_placeholder 0];
    else
        model = fitcdiscr([sensor1 training_set(:,i:i+4)], class_table);
        predictions = predict(model,testing_set);
        conf_mat = confusionmat(cell_test,predictions);
        accuracy = (conf_mat(1,1)+conf_mat(2,2)+conf_mat(3,3)+conf_mat(4,4)+conf_mat(5,5))./sum(sum(conf_mat));
        acc_matrix= [acc_matrix accuracy];
        acc_matrix_placeholder=[acc_matrix_placeholder accuracy];
    end
end
%figure(2)
%plot(acc_matrix)
[max2,sensor_index] = max(acc_matrix_placeholder);
if contains(training_set.Properties.VariableNames{sensor_index.*5-4},'feat_a_1')
    sensor2 = training_set(:,sensor_index.*5-4:sensor_index.*5+5);
    training_set(:,sensor_index.*5-4:sensor_index.*5+5)=[];
    flag2 = 10;
else
    sensor2 = training_set(:,sensor_index.*5-4:sensor_index.*5);
    training_set(:,sensor_index.*5-4:sensor_index.*5)=[];
    flag2 = 5;
end

%ITERATION 3
acc_matrix=[];
acc_matrix_placeholder=[];
for i = 1:5:251-(flag1+flag2)
    if contains(training_set.Properties.VariableNames{i},'feat_a_1')
        model = fitcdiscr([sensor1 sensor2 training_set(:,i:i+9)], class_table);
        predictions = predict(model,testing_set);
        conf_mat = confusionmat(cell_test,predictions);
        accuracy = (conf_mat(1,1)+conf_mat(2,2)+conf_mat(3,3)+conf_mat(4,4)+conf_mat(5,5))./sum(sum(conf_mat));
        acc_matrix= [acc_matrix accuracy];
        acc_matrix_placeholder=[acc_matrix_placeholder accuracy];
    elseif contains(training_set.Properties.VariableNames{i},'feat_a_6')
        acc_matrix_placeholder = [acc_matrix_placeholder 0];
    else
        model = fitcdiscr([sensor1 sensor2 training_set(:,i:i+4)], class_table);
        predictions = predict(model,testing_set);
        conf_mat = confusionmat(cell_test,predictions);
        accuracy = (conf_mat(1,1)+conf_mat(2,2)+conf_mat(3,3)+conf_mat(4,4)+conf_mat(5,5))./sum(sum(conf_mat));
        acc_matrix= [acc_matrix accuracy];
        acc_matrix_placeholder=[acc_matrix_placeholder accuracy];
    end
end
%figure(3)
%plot(acc_matrix)
[max3,sensor_index] = max(acc_matrix_placeholder);
if contains(training_set.Properties.VariableNames{sensor_index.*5-4},'feat_a_1')
    sensor3 = training_set(:,sensor_index.*5-4:sensor_index.*5+5);
    training_set(:,sensor_index.*5-4:sensor_index.*5+5)=[];
    flag3 = 10;
else
    sensor3 = training_set(:,sensor_index.*5-4:sensor_index.*5);
    training_set(:,sensor_index.*5-4:sensor_index.*5)=[];
    flag3 = 5;
end

%ITERATION 4
acc_matrix=[];
acc_matrix_placeholder=[];
for i = 1:5:251-(flag1+flag2+flag3)
    if contains(training_set.Properties.VariableNames{i},'feat_a_1')
        model = fitcdiscr([sensor1 sensor2 sensor3 training_set(:,i:i+9)], class_table);
        predictions = predict(model,testing_set);
        conf_mat = confusionmat(cell_test,predictions);
        accuracy = (conf_mat(1,1)+conf_mat(2,2)+conf_mat(3,3)+conf_mat(4,4)+conf_mat(5,5))./sum(sum(conf_mat));
        acc_matrix= [acc_matrix accuracy];
        acc_matrix_placeholder=[acc_matrix_placeholder accuracy];
    elseif contains(training_set.Properties.VariableNames{i},'feat_a_6')
        acc_matrix_placeholder = [acc_matrix_placeholder 0];
    else
        model = fitcdiscr([sensor1 sensor2 sensor3 training_set(:,i:i+4)], class_table);
        predictions = predict(model,testing_set);
        conf_mat = confusionmat(cell_test,predictions);
        accuracy = (conf_mat(1,1)+conf_mat(2,2)+conf_mat(3,3)+conf_mat(4,4)+conf_mat(5,5))./sum(sum(conf_mat));
        acc_matrix= [acc_matrix accuracy];
        acc_matrix_placeholder=[acc_matrix_placeholder accuracy];
    end
end
%figure(4)
%plot(acc_matrix)
[max4,sensor_index] = max(acc_matrix_placeholder);
if contains(training_set.Properties.VariableNames{sensor_index.*5-4},'feat_a_1')
    sensor4 = training_set(:,sensor_index.*5-4:sensor_index.*5+5);
    training_set(:,sensor_index.*5-4:sensor_index.*5+5)=[];
    flag4 = 10;
else
    sensor4 = training_set(:,sensor_index.*5-4:sensor_index.*5);
    training_set(:,sensor_index.*5-4:sensor_index.*5)=[];
    flag4 = 5;
end

%ITERATION 5
acc_matrix=[];
acc_matrix_placeholder=[];
for i = 1:5:251-(flag1+flag2+flag3+flag4)
    if contains(training_set.Properties.VariableNames{i},'feat_a_1')
        model = fitcdiscr([sensor1 sensor2 sensor3 sensor4 training_set(:,i:i+9)], class_table);
        predictions = predict(model,testing_set);
        conf_mat = confusionmat(cell_test,predictions);
        accuracy = (conf_mat(1,1)+conf_mat(2,2)+conf_mat(3,3)+conf_mat(4,4)+conf_mat(5,5))./sum(sum(conf_mat));
        acc_matrix= [acc_matrix accuracy];
        acc_matrix_placeholder=[acc_matrix_placeholder accuracy];
    elseif contains(training_set.Properties.VariableNames{i},'feat_a_6')
        acc_matrix_placeholder = [acc_matrix_placeholder 0];
    else
        model = fitcdiscr([sensor1 sensor2 sensor3 sensor4 training_set(:,i:i+4)], class_table);
        predictions = predict(model,testing_set);
        conf_mat = confusionmat(cell_test,predictions);
        accuracy = (conf_mat(1,1)+conf_mat(2,2)+conf_mat(3,3)+conf_mat(4,4)+conf_mat(5,5))./sum(sum(conf_mat));
        acc_matrix= [acc_matrix accuracy];
        acc_matrix_placeholder=[acc_matrix_placeholder accuracy];
    end
end
%figure(5)
%plot(acc_matrix)
[max5,sensor_index] = max(acc_matrix_placeholder);
if contains(training_set.Properties.VariableNames{sensor_index.*5-4},'feat_a_1')
    sensor5 = training_set(:,sensor_index.*5-4:sensor_index.*5+5);
else
    sensor5 = training_set(:,sensor_index.*5-4:sensor_index.*5);
end

%supercell(1,q) = cellstr(sensor1.Properties.VariableNames{1});
%supercell(2,q) = cellstr(sensor2.Properties.VariableNames{1});
%supercell(3,q) = cellstr(sensor3.Properties.VariableNames{1});
%supercell(4,q) = cellstr(sensor4.Properties.VariableNames{1});
%supercell(5,q) = cellstr(sensor5.Properties.VariableNames{1});

%supermatrix(1,q) = max1;
%supermatrix(2,q) = max2;
%supermatrix(3,q) = max3;
%supermatrix(4,q) = max4;
%supermatrix(5,q) = max5;

%q
%end

