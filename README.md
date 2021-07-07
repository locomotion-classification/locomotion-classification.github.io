# Locomotion Mode Classification based on Wearable Sensor data from Lower Limb Biomechanics 

## 1 Introduction
Characterization of human movement and understanding of corresponding muscle coordination over various locmotion modes and terrain conditions is crucial for determining human movement strategies in environments more prevalent to community ambulation. Biomechanical data captured during locomotion, such as, Electromyography (EMG) signals, can be used to anticipate transitions from one mode to another. Moreover, combination of multiple sensor data has the possibility to assist in data-driven model development for locomotion intent recognition using Machine Learning (ML) to greatly accelerate such tasks. 

## 2 Problem Definition
The [public dataset](http://www.epic.gatech.edu/opensource-biomechanics-camargo-et-al/) for the project has been acquired by EPIC Lab at Georgia Tech. The dataset contains 3-dimensional biomechanical and wearable sensor data (EMG - 11 muscles, Goniometer (GON) - 3 body parts, Inertial Measurement Unit (IMU) - 4 body parts) along with the kinematic and kinetic profiles of joint biomechanics (as a function of gait phase) from right side of the body of 22 young and able-bodied adults for 5 locomotion modes (e.g. level-ground or treadmill walking, stair-ascent, stair-descent, ramp-ascent and ramp-descent), multiple terrain conidtions for each mode (walking speed, stair height, and ramp inclination) and multiple trials [1].

<p align="center">
  <img src="Project_Description.png" class="img-responsive" alt="Project">
</p>


In this project, we will develop subject-dependent classification models for 6 possible modes (standing and 5 locomotion modes) regardless of their terrain conditions based on biomechanics data captured from lower limb using EMG, GON, IMU, Inverse Dynamics (ID), Inverse Kinematics (IK), Joint Power (JP), Force Plate (FP) etc. from able-bodied participants for a single adult (e.g. AB21 from the dataset).

## 3 Literature Review
The study in [1] focused only on linear relationships between variables and terrain conditions. In this work, we will explore possible higher order relationships and develop ML models to classify locomotion modes. As the dataset has been released only recently(Feb. 2021), this will be one of the first studies on this dataset along this direction. Recently, gait phase estimation during multimodal locomotion from just IMU data [2], and subject-independent slope prediction from just IMU and biomechanical data [3] using Convoltuional Neural Network have been reported for exoskeleton assistance. However, in this study, we will combine data from multiple wearable sensors, thus providing a more comprehensive picture.

## 4 Data Collection

### 4.1 Finding Data

### 4.2 Cleaning and Merging Data

### 4.3 Data Pre-Processing

#### 4.3.1 Signal Conditioning

#### 4.3.2 Feature Engineering
##### 4.3.2.1 Minimum Relevance Maximum Redundancy (MRMR) Scoring

All of raw data and labels go into the algorithm which scores features based on their relevance to class labels.

<p align="center">
  <img src="images\MRMR_raw.svg" class="img-responsive" alt="Project"> 
</p>


As you can see, **feature no. 2 (Goniometer, ankle_frontal)** has been ranked the highest by the algorithm due to its importance as a predictor for class labels. The others just from raw data seems to be insignificant. This will later be verified by other algorithms later on.

## 5 Methods
In this project, we will use the aforementioned sesnor and biomechanical data as input features and corressponding locomotion modes as classification labels to develop supervised and unsupervised ML models such as k-Means Clustering, Principal Component Analysis (PCA), Linear Discriminant Analysis (LDA), Support Vector Machines (SVM), Random Forest (RF), Decision Trees (DT), and Neural Networks (NN). Although the data has been appropriately filtered for the specific sensors, we may have to further pre-process the raw data (e.g. normalization of Joint Moment and Powers data by the subject mass, rectification and low-pass filtering of EMG data) due to possibly unaccounted-for factors such as, skin conditions, electrode placements, and anatomical differences between the subjects. Also, for some sensors, instead of the actual data samples, use of the envelop of data samples might make more sense. We expect to have reasonable classification accuracy the locmotion modes using the aforementioned methods. Additionally, development of interpretable models will be of particular interest. 

### 5.1 Unsupervised Learning Component

### 5.2 Supervised Learning Component


## 6 Results

### 6.1 PCA

### 6.2 k-Means Clustering

### 6.4 Linear Discriminant Analysis (LDA)

Random permutation has been utilized to randomly split the raw data into training_data (80% of total) and testing_data (20% of total). This **LDA** achieves **68.41%** test accuracy on raw data, compared to 14.2% which would be for random class assignments, and thus contains significant results on just raw data.

After performing **PCA**, **LDA** achieves **68.41%** test accuracy with no dimensionality reduction, so, PCA without pruning does not improve accuracy for LDA.

### 6.3 MLP

<p align="center">
  <img src="images\MLP_arch.svg" class="img-responsive" alt="Project"  width="400" height="400"> 
</p>

After running for 50 epochs, this simple **MLP** with 3 fully connected layers achieves **99.06%** test accuracy on raw data (pretty significant).

<p align="center">
  <img src="images\MLP_raw_acc_loss.svg" class="img-responsive" alt="Project"> 
</p>

But **PCA-MLP** achieves **99.17% test accuracy** with no dimensionality reduction, which is higher than just the raw data. Running for a higher number of epochs will translate to higher accuracy, Thorough hyper-parameter tuning is yet to be done, but the result is already quite good.

<p align="center">
  <img src="images\MLP_PCA_acc_loss.svg" class="img-responsive" alt="Project">
</p>
 

### 6.4 CNN

<p align="center">
  <img src="images\CNN_arch2.svg" class="img-responsive" alt="Project" width="800" height="800"> 
</p>

<p align="center">
  <img src="images\CNN_acc.png" class="img-responsive" alt="Project"> <img src="images\CNN_loss.png" class="img-responsive" alt="Project">
</p>

## 7 Broader Impact
The broader impact of this project would be future development of robotic assistive devices and active prostheses for targeted rehabilitation methods beyond clinical settings, and improvement of biomimetic controllers that better adapt to terrain conditions (practical scenarios).

## 8 Contributions

* Ian Thomas Cullen: Data Cleaning, Data Pre-processing (Signal Conditioning using Windowed Moving Average), Feature Engineering (calculating max/min, mean, std from raw features), Video Editing and Presentation for Proposal

* Kennedy A Lee: Convolutional Neural Network (CNN) on raw data, Hyperparameter Tuning

* Imran Ali Shah: Principal Component Analysis (PCA), k-Means Clustering, Video Editing and Presentation for Proposal

* Anupam Golder: Data Merging (for different environments and different locomotion modes), Data Cleaning, Feature Scoring (Using Minimum Redundancy Maximum Relevance scoring), MLP on raw data, Github Page Editing

## 9 References
1. Camargo, J., Ramanathan, A., Flanagan, W., & Young, A. (2021). A comprehensive, open-source dataset of lower limb biomechanics in multiple conditions of stairs, ramps, and level-ground ambulation and transitions. Journal of Biomechanics, 119, 110320.
2. Kang, I., Molinaro, D. D., Duggal, S., Chen, Y., Kunapuli, P., & Young, A. J. (2021). Real-Time Gait Phase Estimation for Robotic Hip Exoskeleton Control During Multimodal Locomotion. IEEE Robotics and Automation Letters, 6(2), 3491-3497.
3. Lee, D., Kang, I., Molinaro, D. D., Yu, A., & Young, A. J. (2021). Real-Time User-Independent Slope Prediction Using Deep Learning for Modulation of Robotic Knee Exoskeleton Assistance. IEEE Robotics and Automation Letters, 6(2), 3995-4000.
