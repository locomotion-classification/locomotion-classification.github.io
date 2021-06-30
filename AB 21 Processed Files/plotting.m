clear all; close all; clc;

%% Data

PCA = 5:5:80;
LDA = [62.3,71.2,94,95.4,95.6,96,96.9,97.6,99.6,99.8,99.8,99.9,99.9,99.9,99.9,100];
SVM = [70,80.9,97.3,98.2,98.2,98.2,98.7,99.2,99.8,99.8,99.8,99.9,100,100,100,100];

LDA = (100-LDA);
SVM = (100-SVM);

figure(1)
plot(PCA,LDA,'LineWidth',2)
hold on
plot(PCA,SVM,'LineWidth',2)
legend('LDA','SVM')
xlim([5 80])
xlabel('PCA Components')
ylabel('Classification Error (%)')
title('Error Classification by Algorithm and PCA')
grid on
hold off

%% NN Data

NN = [];
NN_mean = [];
NN_stdev = [];
NN(:,1) = [79.1,79,80.7,55.6,69.6,54.3,80.7,71.6,35.3,84];
NN(:,2) = [92.1,89.9,65.2,90.1,100,61.5,98.8,89.4,100,90.6];
NN(:,3) = [97.8,100,100,99.8,99,100,100,100,99.8,100];
NN(:,4) = [99.5,100,100,99.5,86.4,100,100,61.2,99.5,99.8];
NN(:,5) = [100,100,100,100,99.8,99.8,99.8,100,99.8,99.8];
NN(:,6) = [18.8,100,100,100,100,100,100,100,100,99.8];

NN = (100-NN);

for i = 1:6
    NN_mean(i) = mean(NN(:,i));
    NN_stdev(i) = std(NN(:,i));
end

figure(2)
errorbar(1:6,NN_mean,NN_stdev,'LineWidth',2)
hold on
xlim([0.9 6.1])
ylim([0 50])
xlabel('Hidden Neural Nodes')
ylabel('Classification Error (%)')
title('Neural Network Error as a Function of Hidden Nodes')
grid on
hold off

%%
gracilis_trunk_ankle_mean = 100*[0.8547486034,0.9734636872,0.9934823091];
gracilis_trunk_ankle_stdev = 100*[0.005888785214,0.006152770826,0.001316772404];
gracilis_trunk_gluteus_mean = 100*[0.8547486034,0.9734636872,0.9944134078];
gracilis_trunk_gluteus_stdev = 100*[0.005888785214,0.006152770826,0];
gluteus_trunk_ankle_mean = 100*[0.8675815723,0.9781191806,0.9953445065];
gluteus_trunk_ankle_stdev = 100*[0.01271655416,0.006692693561,0.001316772404];
gluteus_trunk_knee_mean = 100*[0.8675815723,0.9781191806,1];
gluteus_trunk_knee_stdev = 100*[0.01271655416,0.006692693561,0];
gluteus_trunk_gracilis_mean = 100*[0.8675815723,0.9781191806,0.9981378026];
gluteus_trunk_gracilis_stdev = 100*[0.01271655416,0.006692693561,0];
gluteus_hip_ksag_mean = 100*[0.8675815723,0.9808016316,0.9976057462];
gluteus_hip_ksag_stdev = 100*[0.01271655416,0.005601531783,0.002072061975];
gluteus_hip_afront_mean = 100*[0.8675815723,0.9808016316,0.9970736898];
gluteus_hip_afront_stdev = 100*[0.01271655416,0.005601531783,0.001465169073];
gluteus_hip_asag_mean = 100*[0.8675815723,0.9808016316,0.9957435488];
gluteus_hip_asag_stdev = 100*[0.01271655416,0.005601531783,0.002986159128];

figure(3)
errorbar(1:3,100-gracilis_trunk_ankle_mean,gracilis_trunk_ankle_stdev)
hold on
errorbar(1:3,100-gracilis_trunk_gluteus_mean,gracilis_trunk_gluteus_stdev)
errorbar(1:3,100-gluteus_trunk_ankle_mean,gluteus_trunk_ankle_stdev)
%errorbar(1:3,100-gluteus_trunk_knee_mean,gluteus_trunk_knee_stdev)
%errorbar(1:3,100-gluteus_trunk_gracilis_mean,gluteus_trunk_gracilis_stdev)
errorbar(1:3,100-gluteus_hip_ksag_mean,gluteus_hip_ksag_stdev)
errorbar(1:3,100-gluteus_hip_afront_mean,gluteus_hip_afront_stdev)
errorbar(1:3,100-gluteus_hip_asag_mean,gluteus_hip_asag_stdev)
%xlim([1.99 3.01])
%ylim([0 3])
legend('Gracilis - Trunk Acceleration Z - Ankle Sagittal','Gracilis - Trunk Acceleration Z - Gluteus Medius',...
    'Gluteus Medius - Trunk Acceleration Z - Ankle Frontal','Gluteus Medius - Hip Sagittal - Knee Sagittal',...
    'Gluteus Medius - Hip Sagittal - Ankle Frontal','Gluteus Medius - Hip Sagittal - Ankle Sagittal')
xlabel('Features Selected')
ylabel('Classification Error (%)')
title('Classification Error as a Function of Features Selected')
grid on
hold off

%% Frequency
Y=[4,7,3,1,2,7,7,1,3,1,1,2,1,1,1,2,1,1,2,2];
Y=Y/50;
figure(4)
subplot(1,2,1)
bar(categorical({'Gluteus Medius','Gracilis'}),[46/50 4/50])
hold on
ylabel('Relative Frequency')
title('First Feature Selection')
subplot(1,2,2)
bar(categorical({'Gluteus-Sagittal Hip','Gluteus-Trunk Acc Z','Gracilus-Trunk Acc Z'}),[42/50 4/50 4/50])
ylim([0 1])
title('Second Feature Selection')
hold off
figure(5)
b=bar(categorical({'Glut-SagHip-VastusMed','Glut-SagHip-KneeSag','Glut-SagHip-FootAccZ','Glut-SagHip-Soleus',...
    'Glut-SagHip-ShankAccY','Glut-SagHip-AnkFront','Glut-SagHip-AnkSag','Glut-SagHip-VastusLat','Glut-SagHip-ShankGyroY',...
    'Glut-SagHip-FootGyroY','Glut-SagHip-RectFemoris','Glut-SagHip-ShankAccZ','Glut-SagHip-ShankGyroX',...
    'Glut-SagHip-ThighGyroY','Glut-SagHip-FootAccelY','Glut-TAZ-AnkFront','Glut-TAZ-KneeSag','Glut-TAZ-Gracilis',...
    'Grac-TAZ-AnkSag','Grac-TAZ-GlutMed'}),Y)
ylim([0 0.15])
ylabel('Relative Frequency')
title('Third Feature Selection')


