clear;clc;
%% Convert data to matrix (this may need to be modified depending on how data is input)

load('data_all_timestamp.mat')

timestamp_data = all_env_data;
load('data_all_env.mat')
data = [timestamp_data, all_env_labels];
%writematrix(data, 'C:\Users\bread\Documents\MachineLearningHW\PROJECT\AB21_analysis\data.csv')

%%
% windowed_data.mat: 1st column is starting time of window,
% 2nd column and every 5th column afterwords is max value in window,
% 3rd column and every 5th column afterwords is min value in window,
% 4th column and every 5th column afterwords is mean value in window,
% 5th column and every 5th column afterwords is standard deviation in window,
% 6th column and every 5th column afterwords is final value in window,
% last column is classification

zero_vec = zeros(1,(size(data,2)-2)*5+2);
time_mat = zeros(length(data),(size(data,2)-2)*5+2);

for indx = 1:10:length(data)-9
    if data(indx+9,1)-data(indx,1) < 0.1 && data(indx+9,1)>data(indx,1)
        indx_vec = zero_vec;
        indx_data = data([indx:indx+9],:);
        indx_vec(1) = indx_data(1,1);
        indx_vec(end) = indx_data(1,end);
        indx_vec(2:5:length(zero_vec)-1) = max(indx_data(:,2:end-1)); %Find maximum value of features in window
        indx_vec(3:5:length(zero_vec)-1) = min(indx_data(:,2:end-1)); %Find minimum value of features in window
        indx_vec(4:5:length(zero_vec)-1) = mean(indx_data(:,2:end-1)); %Find mean value of features in window
        indx_vec(5:5:length(zero_vec)-1) = std(indx_data(:,2:end-1)); %Find standard deviation of features in window
        indx_vec(6:5:length(zero_vec)-1) = indx_data(end,2:end-1); %Find final (time-series) value of features in window
        time_mat(ceil(indx/10),:) = indx_vec; %Concatenate window into array
    end
    
end

windowed_data = time_mat(any(time_mat,2),:); %Trims out zero rows
save('windowed_data.mat');

