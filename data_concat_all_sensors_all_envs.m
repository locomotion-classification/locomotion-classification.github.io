clc;
clear all;
close all;

%'AB21' folder should be kept in the same directory
home = 'AB21\01_27_2019';

%'Treadmill' environment does not contain locomotion mode labels, just speeds, so have not
%considered that one.
envs = {'levelground','ramp','stair'};

%'id' (inverse dynamics) and 'ik' (inverse kinematics) contains some
%missing data, with 'NaN' values, so have not considered them
%'jp' (Joint Power), and 'fp' (Force Plate) seem to have very low
% information content, so have not considered those two as well.

%5 sensor data
%'gon' = Goniometer from hip, knee, and ankle, sampled at 1000 Hz
%'emg' = Electromyography from 11 muscles, sampled at 1000 Hz
%'gcLeft' = Gait cycle segmented by heel strike or toe off of left foot,
%sampled at 200 Hz
%'gcRight' = Gait cycle segmented by heel strike or toe off of right foot,
%sampled at 200 Hz
%'imu' = Inertial Measurement Unit data from trunk, thigh, shank, and foot
%segments. Sampled at 200 Hz
sensors = {'gon','emg','gcLeft','gcRight','imu'};

%7 locomotion modes have been considered
modes = {'idle','walk','stand','stairascent','stairdescent','rampascent','rampdescent'};

%Initialization of data and labels
all_env_data = [];
all_env_labels = [];

for i = 1:length(envs) %for all 3 environments
    home_env = strcat(home,'\',string(envs{i}),'\');
    all_data = [];
    all_labels = [];
    
    for m = 1:length(sensors) %for all 5 sensors
        
        %labels_dir contains the files with labels
        labels_dir = dir(strcat(home_env,'conditions\*.mat'));
        
        %sensor_dir contains the files with respective sensor data
        sensor_dir = dir(strcat(home_env,string(sensors{m}),'\*.mat'));
        data = [];
        true_labels = [];
        
        for j = 1:length(labels_dir) %for all data campaigns for a single environment and a single sensor
            
            labels = [];
            
            %labels_file contains the current data labels
            labels_file = strcat(home_env,'conditions\',labels_dir(j).name);
            
            %sensors_file contains the current sensor data
            sensors_file = strcat(home_env,string(sensors{m}),'\',sensor_dir(j).name);
            
            %objects to hold both data and labels
            X1 = load(labels_file);
            X2 = load(sensors_file);
            
            %Pre-processing: downsampling 'gon' and 'img' data (m<=2) by 5 as
            %they are sampled at 1000 Hz where 'gcLeft', 'gcRight' and
            %'imu' are sampled at 200 Hz
            if m <=2
                X2_data = X2.data(1:5:end,:);
            else
                X2_data = X2.data;
            end
            
            %Pre-processing
            %aligning timestamps from sensor data file and labels file
            %and getting rid of locomotion modes that we do not consider,
            %such as, 'walk-stand','stand-walk','turn1','turn2' etc.
            timestamp_labels = table2array(X1.labels(:,1));
            labels = table2array(X1.labels(:,2));
            [labels,label_names] = grp2idx(labels);

            classes = zeros(length(labels),1);            
            idx_all = [];
            for k = 1:length(modes)
                mode_idx = find(ismember(label_names,modes{k})==1);
                idx = find(ismember(labels,mode_idx)==1);
                idx_all = [idx_all; idx];
                classes(idx,1) = k;
            end
            
            data = [data; table2array(X2_data(idx_all,2:end))];
            true_labels = [true_labels; classes(idx_all,1)];
        end
        %all_data contains all sensor data for a specific environment
        %all_labels contains corresponding labels
        all_data = [all_data data];
        all_labels = true_labels;
    end
    all_env_data = [all_env_data;all_data];
    all_env_labels = [all_env_labels; all_labels];
end

save('data_all_env.mat','all_env_data','all_env_labels');