## Locomotion Mode Classification based on Wearable Sensor data from Lower Limb Biomechanics 

### Introduction
Characterization of human movement and understanding of corresponding muscle coordination over various locmotion modes and terrain conditions is crucial for determining human movement strategies in environments more prevalent to community ambulation. Biomechanical data captured during locomotion, such as, Electromyography (EMG) signals, can be used to anticipate transitions from one mode to another. Moreover, combination of multiple sensor data has the possibility to assist in data-driven model development for locomotion intent recognition using Machine Learning (ML) to greatly accelerate such tasks. 

### Problem Definition
The [public dataset](http://www.epic.gatech.edu/opensource-biomechanics-camargo-et-al/) for the project has been acquired by EPIC Lab at Georgia Tech. The dataset contains 3-dimensional biomechanical and wearable sensor data (EMG - 11 muscles, Goniometer (GON) - 3 body parts, Inertial Measurement Unit (IMU) - 4 body parts) along with the kinematic and kinetic profiles of joint biomechanics (as a function of gait phase) from right side of the body of 22 young and able-bodied adults for 5 locomotion modes (e.g. level-ground or treadmill walking, stair-ascent, stair-descent, ramp-ascent and ramp-descent), multiple terrain conidtions for each mode (walking speed, stair height, and ramp inclination) and multiple trials [1].

<img src="Project_Description.png" class="img-responsive" alt="Project">

In this project, we will develop subject-dependent classification models for 6 possible modes (standing and 5 locomotion modes) regardless of their terrain conditions based on biomechanics data captured from lower limb using EMG, GON, IMU, Inverse Dynamics (ID), Inverse Kinematics (IK), Joint Power (JP), Force Plate (FP) etc. from able-bodied participants for a single adult (e.g. AB21 from the dataset).

### Literature Review
The study in [1] focused only on linear relationships between variables and terrain conditions. In this work, we will explore possible higher order relationships and develop ML models to classify locomotion modes. As the dataset has been released only recently(Feb. 2021), this will be one of the first studies on this dataset along this direction. Recently, gait phase estimation during multimodal locomotion from just IMU data [2], and subject-independent slope prediction from just IMU and biomechanical data [3] using Convoltuional Neural Network have been reported for exoskeleton assistance. However, in this study, we will combine data from multiple wearable sensors, thus providing a more comprehensive picture.

### Data Collection

#### Finding Data

#### Cleaning and Merging Data

#### Data Pre-Processing

##### Signal Conditioning

##### Feature Engineering

### Methods
In this project, we will use the aforementioned sesnor and biomechanical data as input features and corressponding locomotion modes as classification labels to develop supervised and unsupervised ML models such as k-Means Clustering, Principal Component Analysis (PCA), Linear Discriminant Analysis (LDA), Support Vector Machines (SVM), Random Forest (RF), Decision Trees (DT), and Neural Networks (NN). Although the data has been appropriately filtered for the specific sensors, we may have to further pre-process the raw data (e.g. normalization of Joint Moment and Powers data by the subject mass, rectification and low-pass filtering of EMG data) due to possibly unaccounted-for factors such as, skin conditions, electrode placements, and anatomical differences between the subjects. Also, for some sensors, instead of the actual data samples, use of the envelop of data samples might make more sense. We expect to have reasonable classification accuracy the locmotion modes using the aforementioned methods. Additionally, development of interpretable models will be of particular interest. 

#### Unsupervised Learning Component

#### Supervised Learning Component


### Results

#### PCA

#### k-Means Clustering

#### MLP

#### CNN

### Broader Impact
The broader impact of this project would be future development of robotic assistive devices and active prostheses for targeted rehabilitation methods beyond clinical settings, and improvement of biomimetic controllers that better adapt to terrain conditions (practical scenarios).

#### Contributions
Ian Thomas Cullen: Data Cleaning, Data Pre-processing (Signal Conditioning using Windowed Moving Average), Feature Engineering (calculating max/min, mean, std from raw features) 
Kennedy A Lee: Convolutional Neural Network (CNN)
Imran Ali Shah: Principal Component Analysis (PCA), k-Means Clustering 
Anupam Golder: Data Merging (for different environments and different locomotion modes), Data Cleaning, Feature Scoring (Using Minimum Redundancy Maximum Relevance scoring), MLP on raw data

### References
1. Camargo, J., Ramanathan, A., Flanagan, W., & Young, A. (2021). A comprehensive, open-source dataset of lower limb biomechanics in multiple conditions of stairs, ramps, and level-ground ambulation and transitions. Journal of Biomechanics, 119, 110320.
2. Kang, I., Molinaro, D. D., Duggal, S., Chen, Y., Kunapuli, P., & Young, A. J. (2021). Real-Time Gait Phase Estimation for Robotic Hip Exoskeleton Control During Multimodal Locomotion. IEEE Robotics and Automation Letters, 6(2), 3491-3497.
3. Lee, D., Kang, I., Molinaro, D. D., Yu, A., & Young, A. J. (2021). Real-Time User-Independent Slope Prediction Using Deep Learning for Modulation of Robotic Knee Exoskeleton Assistance. IEEE Robotics and Automation Letters, 6(2), 3995-4000.
