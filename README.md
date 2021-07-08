# Locomotion Mode Classification based on Wearable Sensor data from Lower Limb Biomechanics 

## 1 Introduction
Characterization of human movement and understanding of corresponding muscle coordination over various locmotion modes and terrain conditions is crucial for determining human movement strategies in environments more prevalent to community ambulation. Biomechanical data captured during locomotion, such as, Electromyography (EMG) signals, can be used to anticipate transitions from one mode to another. Moreover, combination of multiple sensor data has the possibility to assist in data-driven model development for locomotion intent recognition using Machine Learning (ML) to greatly accelerate such tasks. 

## 2 Problem Definition
The dataset contains 3-dimensional biomechanical and wearable sensor data (EMG - 11 muscles, Goniometer (GON) - 3 body parts, Inertial Measurement Unit (IMU) - 4 body parts) along with the kinematic and kinetic profiles of joint biomechanics (as a function of gait phase) from right side of the body of 22 young and able-bodied adults for various locomotion modes (e.g. level-ground or treadmill walking, stair-ascent, stair-descent, ramp-ascent and ramp-descent), multiple terrain conidtions for each mode (walking speed, stair height, and ramp inclination) and multiple trials [1].

<p align="center">
  <img src="Project_Description.png" class="img-responsive" alt="Project">
</p>


In this project, we will develop subject-dependent classification models for 6 possible modes (standing and 5 locomotion modes) regardless of their terrain conditions based on biomechanics data captured from lower limb using EMG, GON, IMU, Inverse Dynamics (ID), Inverse Kinematics (IK), Joint Power (JP), Force Plate (FP) etc. from able-bodied participants for a single adult (e.g. AB21 from the dataset).

## 3 Literature Review
The study in [1] focused only on linear relationships between variables and terrain conditions. In this work, we will explore possible higher order relationships and develop ML models to classify locomotion modes. As the dataset has been released only recently(Feb. 2021), this will be one of the first studies on this dataset along this direction. Recently, gait phase estimation during multimodal locomotion from just IMU data [2], and subject-independent slope prediction from just IMU and biomechanical data [3] using Convoltuional Neural Network have been reported for exoskeleton assistance. However, in this study, we will combine data from multiple wearable sensors, thus providing a more comprehensive picture.

## 4 Data Collection

### 4.1 Finding Data
The [public dataset](http://www.epic.gatech.edu/opensource-biomechanics-camargo-et-al/) for the project has been acquired by EPIC Lab at Georgia Tech. In this project, we work on a randomly chosen subset of the data, i.e., we focus on sensor data collected from a single participant (e.g. AB21) and cosider all possible terrain conditions and locmotion modes and related sensors.

### 4.2 Cleaning and Merging Data

The dataset has an organized directory structure containing timestamps and corresponding locomotion labels. During data exploration, we have excluded Treadmill terrain condition from our considered dataset, as it does not contain corresponsing labels. It, howvwer, constains walking/running speeds, which can be a regression analysis task, based on the sensor data, and as such, was not considered in this study. Henceforth, we consider these three terrain conditions: **```Levelground```**, **```Ramp```**, and **```Stair```**.

Some of the biomechanics data had missing data values (NaN or Not-a-Number) or low information content, such as 'Inverse Kinematics' (ik), 'Inverse Dynamics' (id), 'Joint Power' (jp), and 'Force Plate' (fp). We have not considered them as well. In particular, we focused on **```5 sensors```**:

* **```gon```**: Goniometer data from hip, knee, and ankle, sampled at 1000 Hz
* **```emg```**: Electromyography data from 11 muscles, sampled at 1000 Hz
* **```gcLeft```**: Gait cycle segmented by heel strike or toe off of left foot, sampled at 200 Hz
* **```gcRight```**: Gait cycle segmented by heel strike or toe off of right foot, sampled at 200 Hz
* **```imu```**: Inertial Measurement Unit data from trunk, thigh, shank, and foot segments. Sampled at 200 Hz

Moreover, in this study, we consider only 7 locomotion modes, specifically, **```idle```**, **```walk```**, **```stand```**, **```stair-ascent```**, **```stair-descent```**, **```ramp-ascent```**, and **```ramp-descent```**. Consequently, the problem becomes that of a classification task with 7 output classes and 44 sensor data features.

Data merging process involved concatenating data from different locomotion modes, different sensors, and different acquisition campaigns. As you might notice, the sensor data are sampled at two different sampling frequencies (200 Hz and 1000 Hz). We chose to downsample the high frequency data to align the features.

### 4.3 Data Pre-Processing

#### 4.3.1 Signal Conditioning

The original dataset consists of several features representing individual sensor channels. Each feature is sampled at a rate of 200 Hz, resulting in a large number of samples. However, for noisy data, especially EMG data, this data can be difficult to classify without processing. One method of doing so is to collect samples in a moving window of time, and create features for each window of time. For this project, each window occurs in series after the previous one, as seen in the figure below, with gaps occurring whenever there are breaks in the time series data.

<p align="center">
  <img src="images\window_even.PNG" class="img-responsive" alt="Project"> 
</p>

For each time window, 5 sub-features were extracted from each feature: the maximum and minimum values in that window, the mean value of the window, the standard deviation of values in that window, and the final value in the window. These sub-features effectively multiply the number of dimensions of the dataset by 5. While this means that dimensionality reduction will be more intensive, the processed features will end up more informative than the raw data.

<p align="center">
  <img src="images\window_overlap.PNG" class="img-responsive" alt="Project"> 
</p>

#### 4.3.2 Feature Engineering

##### 4.3.2.1 Minimum Relevance Maximum Redundancy (MRMR) Scoring

All of raw data and labels go into the algorithm which scores features based on their relevance to class labels.

<p align="center">
  <img src="images\MRMR_raw2.svg" class="img-responsive" alt="Project"> 
</p>


As you can see, **```feature no. 2 (Goniometer, ankle-frontal)```** has been ranked the highest by the algorithm due to its importance as a predictor for class labels. The others just from raw data seems to be insignificant. This will later be verified by other algorithms later on.

## 5 Methods
In this project, we will use the aforementioned sesnor and biomechanical data as input features and corressponding locomotion modes as classification labels to develop supervised and unsupervised ML models such as k-Means Clustering, Principal Component Analysis (PCA), Linear Discriminant Analysis (LDA), Support Vector Machines (SVM), Random Forest (RF), Decision Trees (DT), and Neural Networks (NN). Although the data has been appropriately filtered for the specific sensors, we may have to further pre-process the raw data (e.g. normalization of Joint Moment and Powers data by the subject mass, rectification and low-pass filtering of EMG data) due to possibly unaccounted-for factors such as, skin conditions, electrode placements, and anatomical differences between the subjects. Also, for some sensors, instead of the actual data samples, use of the envelop of data samples might make more sense. We expect to have reasonable classification accuracy the locmotion modes using the aforementioned methods. Additionally, development of interpretable models will be of particular interest. 

### 5.1 Unsupervised Learning Component

While working on raw data, considering samples from each time point can be 
### 5.2 Supervised Learning Component


## 6 Results

### 6.1 PCA

Principal component analysis (PCA), which is a broadly used unsupervised algorithm, was performed on the processed data to project it to a lower dimensional space. In essence, PCA was used as a dimensionality reduction algorithm to identify the principal components of our processed data. After performing the PCA, we first analyzed the first two principal components of the analysis (n_components = 2); essentially, this is looking at reducing the dataset of 44 features to only 2, and analyzing the explained variance captured by the first 2 principal components. 

The visualized result as well as the total explained variance for this (PC 1 vs PC 2) is shown in the figure below. Here, we see that the total explained variance is 65.99%, which suggests that using only 2 of the principal components will capture approximately 66% of the variance. In addition, we see that PC1 and PC2 are unable to clearly separate each of the different locomotion labels (labels 1 through 7; 1 - idle, 2 - walk, 3 - stand, 4 - stair ascent, 5 - stair descent, 6 - ramp ascent, 7 - ramp descent). 

<p align="center">
  <img src="images\newplot.png" class="img-responsive" alt="Project"> 
</p>

Thus, we decided to look at additional components to see how much variance is captured with each of the components. Comparisons between the first 10 PCs are shown in the figure below. As seen, when comparing the 8th PC to each of the other PCs, there seems to be a pretty definite separation between each of the different labels. We also see that using 10 PCs describes approximately 99.70% of the variance in the data set (total explained variance is equal to 99.70%).

<p align="center">
  <img src="images\newplot (1).png" class="img-responsive" alt="Project"> 
</p>

To better understand how many of the principal components should be used in further analysis, we extracted the singular values that are obtained from the Singular Value Decomposition (SVD) when the PCA algorithm is performed. Similar to the Elbow Method utilized in finding the optimum number of clusters for a KMeans analysis, a sharp decline in the singular values would suggest that the entire data set can be appropriately captured by only a few components, which specifically are the eigenvectors associated with the highest singular values. A plot of the captured singular values is shown below. From this, we can see that there is a sharp decline in the singular values following 10 components, which suggests using the first 10 principal components would be sufficient for approximating the entire dataset; these results confirm the previous analysis of the different principal components. 

<p align="center">
  <img src="images\singular_value_decay.png" class="img-responsive" alt="Project" width = "500"> 
</p>

### 6.2 k-Means Clustering

KMeans clustering was employed to compare and contrast all datapoints in our cleaned dataset. We begin by performing the KMeans clustering on the entire dataset as a whole. We initialized the algorithm with 10 clusters to see how the algorithm would cluster all the data points and if there were any interesting takeaways. Results for the first cluster, a comparison of the 2nd and 8th cluster, and a plot displaying all of the labeled clusters are shown in the figures below. From the last plot specifically, we see that all of the clusters are overlapping, which indicates that KMeans may not provide any discernable results. 

<p align="center">
  <img src="images\kmeans1.png" class="img-responsive" alt="Project" width="350" />
  <img src="images\kmeans2.png" class="img-responsive" alt="Project" width="350" /> 
  <img src="images\kmeans3.png" class="img-responsive" alt="Project" width="350" />
</p>

KMeans clustering was also employed after a dimensionality reduction using PCA. Specifically, the data was reduced to the first two principal components (where 66% of the variance is captured, as mentioned previously). In this case again, the number of clusters was defined as 10, and the resulting visualization is shown in the figure below. In this case, we see that after dimensionality reduction, the data has been defined into distinct clusters with no overlap between each. 

<p align="center">
  <img src="images\kmeans_pca.png" class="img-responsive" alt="Project" width="500"> 
</p>

Finally, we perform an elbow method analysis of the data (following PCA/dimensionality reduction) to analyze what the optimal number of clusters would be. Here, the find_optimal_num_clusters function implemented in Homework 2 was applied, and the resulting figure is shown below. Using the elbow method, we see that the optimal number of clusters would be approximately 3 to 4. The number clusters chosen for the initial analysis (10) was greater than the value found using the elbow method.

<p align="center">
  <img src="images\elbow_method.png" class="img-responsive" alt="Project" width="500"> 
</p>

### 6.3 Linear Discriminant Analysis (LDA)

Random permutation has been utilized to randomly split the raw data into training_data (80% of total) and testing_data (20% of total). This **LDA** achieves **68.41%** test accuracy on raw data, compared to 14.2% which would be for random class assignments, and thus contains significant results on just raw data.

After performing **PCA**, **LDA** achieves **68.41%** test accuracy with no dimensionality reduction, so, PCA without pruning does not improve accuracy for LDA.

### 6.4 MLP

<p align="center">
  <img src="images\MLP_arch.svg" class="img-responsive" alt="Project"  width="400" height="400"> 
</p>

After running for 50 epochs, this simple **MLP** with 3 fully connected layers achieves **99.06%** test accuracy on raw data (pretty significant).

<p align="center">
  <img src="images\MLP_raw_acc_loss_2.svg" class="img-responsive" alt="Project"> 
</p>

But **PCA-MLP** achieves **99.17% test accuracy** with no dimensionality reduction, which is higher than just the raw data. Running for a higher number of epochs will translate to higher accuracy, Thorough hyper-parameter tuning is yet to be done, but the result is already quite good.

<p align="center">
  <img src="images\MLP_PCA_acc_loss_2.svg" class="img-responsive" alt="Project">
</p>
 

### 6.5 CNN

<p align="center">
  <img src="images\CNN_arch2.svg" class="img-responsive" alt="Project" width="600" height="600"> 
</p>

<p align="center">
  <img src="images\CNN_acc.png" class="img-responsive" alt="Project"> <img src="images\CNN_loss.png" class="img-responsive" alt="Project">
</p>

## 7 Broader Impact
The broader impact of this project would be future development of robotic assistive devices and active prostheses for targeted rehabilitation methods beyond clinical settings, and improvement of biomimetic controllers that better adapt to terrain conditions (practical scenarios).

## 8 Contributions

* ```Ian Thomas Cullen```: Data Cleaning, Data Pre-processing (Signal Conditioning using Windowed Moving Average), Feature Engineering (calculating max/min, mean, std from raw features), Video Editing and Presentation for Proposal

* ```Kennedy A Lee```: Convolutional Neural Network (CNN) on raw data, Hyperparameter Tuning

* ```Imran Ali Shah```: Principal Component Analysis (PCA), k-Means Clustering, Video Editing and Presentation for Proposal

* ```Anupam Golder```: Data Merging (for different environments and different locomotion modes), Data Cleaning, Feature Scoring (Using Minimum Redundancy Maximum Relevance scoring), MLP on raw data, Github Page Editing

## 9 References
1. Camargo, J., Ramanathan, A., Flanagan, W., & Young, A. (2021). A comprehensive, open-source dataset of lower limb biomechanics in multiple conditions of stairs, ramps, and level-ground ambulation and transitions. Journal of Biomechanics, 119, 110320.
2. Kang, I., Molinaro, D. D., Duggal, S., Chen, Y., Kunapuli, P., & Young, A. J. (2021). Real-Time Gait Phase Estimation for Robotic Hip Exoskeleton Control During Multimodal Locomotion. IEEE Robotics and Automation Letters, 6(2), 3491-3497.
3. Lee, D., Kang, I., Molinaro, D. D., Yu, A., & Young, A. J. (2021). Real-Time User-Independent Slope Prediction Using Deep Learning for Modulation of Robotic Knee Exoskeleton Assistance. IEEE Robotics and Automation Letters, 6(2), 3995-4000.
