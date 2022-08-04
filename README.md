![MLinfo](https://user-images.githubusercontent.com/90532657/174424871-785ce7d2-3e6d-46a6-b7aa-ce24235c9a5f.PNG)


# Psychiatric classification
Machine learning project for CS4641 at Georgia Institute of Technology
Collab: Saif Syed Ali, Ibraheem Abutair, Matthew Johnson, Micah Grimes


## Introduction
The objective of this project is to use machine learning to gain insight onto potential predictors or biomarkers present in the brain of patients diagnosed with psychiatric disorders. 

Ideally the inclusion criteria is as follows: 
1. subject must have been diagnosed with a psychiatric disorder by a medical professional
2. Must be having a current psychiatric episodes OR has had an episode in the past 6 months. 

We believe that machine learning is especially useful in this field because it allows for the standardization, comparison, and subsequent analysis of datasets that come from different metrics/studies. Succesful completion of this project would allow us to provide physicians and researchers with possible predictive measures for the development of psychiatric disorders. This would allow more conclusive 


<img width="436" alt="image" src="https://user-images.githubusercontent.com/8241982/182700187-c72d0a1c-597b-4943-a6c9-0809cd671b3b.png">

Pictured above is a generic electrode map, with the top representing the front of the head. The electrodes are named after their respective brain region: for example, the “O1” and “O2” electrodes monitor brain activity in the occipital lobe. Likewise, the “FP1” and “FP2” electrodes monitor brain activity in the frontal cortex. 



## Methodology

### Data types
Our first task was to go about trying to decide what measurements would be most useful to building a psychiatric classifier. Naturally neuro imaging and EEG data came to mind as they are both non-invasive and risk free. However, in our search we found that securing MRI and fMRI images was extremely difficult with many studies being blocked behind a paywalls, not providing enough scope to be a psychiatric classifier, or providing a sparse amount of data points, most of these issues likely being a result of the high expenses that come with operating and owning neuroimaging technology. Due to these limitations, we setteled for using EEG datasets, which is much more inexpensive to perform and use. 


### Dataset
The dataset we selected was an EEG dataset with 2 sets of labels from kaggle. While this data was much more plentiful than other sets we came across, it unfortunately had very little documentation, so data cleaning and analysis was a bit harder than usual.

#### Dimensionality and Description
1. Rows = 945 patients
2. Columns = 1149, for a total of 1148 recorded features (One column being an empty divider between the recordings)
3. The first 114 columns were the frequencies of each sensor with respect to a particular frequency range, categorically known as brain waves
  - 19 electrodes, 6 brain wave classes = 114 readings 
4. The 115th column is empty serving as the aformentioned divider
5. The rest of the dataset from column 116 to 1149 were coherence measurments, which capture the 'communication' between regions by measuring the frequency in which the two regions share wave formations. This is more formally known as phase-amplitude coupling. 
6. The first set of labels consisted of strings that were the specific disorder each patient had. This set consisted of 12 unique values. 
7. The second set of labels consisted of the more general groupings of each disorder, called the main disorders. This set consisted of 8 unique values.

### Data cleaning, Feature extraction, Engineering, and Reduction
1. Initially, PCA was run on the entire dataset in order to reduce dimensionality. This was a must, as the initial dataset contains over 1000 features. After running PCA, we found that aiming for even 70% recovered variance led to 40+ components, which still wasn't ideal. Thus, we decided to divide the data into subsets using feature selection. See results for more information.
2. After PCA analysis we decided to do feature extraction, columns 116 - 1000 were all coherence values between each electrode. Since we were not particularly interested in these values and more so the raw frequencies of each wave, we dropped these columns. This instantly made our dataset much easier to work with. 
Next, for the supervised learning portion, we focused on major depressive disorder (MDD), which allowed us to drop all rows pertaining to other disorders. Lastly, we focused on the frontal parietal lobe (FP1 and FP2 electrodes), as this a critical brain region involved with higher-order thinking. 
All of these feature selections allowed us to create a workable dataset, which we run several learning algorithms on. 

Feature engineering involved normalizing the wave frequencies in order to yield more accurate/consistent models.

Data preprocessing consisted largely of isolating subdata sets and particular features to work with that would increase metrics for our models, this was because the main bulk of the work, transferring time series data to frequency domain had already been done by the authors of the dataset using FFT. Additional preprocessing done by the dataset authors included removing EOG (eye movemnt) artifacts from the frontal electrodes, allowing for a high confidence of signal to be neural in nature. 

### Machine Learning

For unsupervised learning (US), we used the many tools of US that allowed us to quantifiably measure and understand how certain data is distributed in a given data space. We did a kernel density estimation first on selected features that we believed would have significance, as well as clustering analysis both soft and hard to find out if certain neural oscillatory behavior followed any kind of behavior.

For supervised learning, we used Support Vector Machines, Neural Networks (Perceptron), Decision Trees, and Regression (specifically, stochastic gradient descent). For the SVMs, LinearSVC, SVC w/ a Linear Kernel, SVC with an RBF Kernal, and SVC with polynomial kernel were all tried. The polynomial SVC yielded the best results. In depth explanation of these methods are covered in the Results section.

## Results

## Unsupervised results

### feature selection, engineering, and reduction

Results of Principal Component Analysis:

![PCA on fulldata](https://raw.githubusercontent.com/Grimes-Gif/ML_Psychiatric/main/Project%20images/PCA%20on%20fulldata.png)

The large number of principle componenets required suggests that the variance over the entire dataset was scattered about and not that easy to grasp, motivating the creation of subdata sets further in order to have cleaner results.

Due to the incredibly large number of features (1000+) in the dataset, it is important to conduct informed data analysis to perform targeted machine learning. This was accomplished by specifically examining the FP1 and FP2 (frontal parietal cortex) electrodes of depressed patients and healthy controls. Figure 1 shows the frequency correlations of FP1/FP2 between depressed (blue) and control (red) for Alpha, Beta, Delta, and Theta waves. This was done to determine which waves had the largest differences between the two groups.

### Nonparametric methods

While the Alpha, Beta, and Theta waves seem to have considerable overlap, the Delta waves tell a different story: the control subjects seem to have lower frequencies of delta waves, while the depressed subjects seem to have higher frequencies. This is illustrated by the apparent skew of blue dots towards the upper right corner and the skew of red dots towards the bottom left corner in the Delta Wave figure. 

<img width="468" alt="image" src="https://user-images.githubusercontent.com/8241982/182700042-a8986ec1-be96-42f1-ae72-168fdfb956fd.png">


This hypothesis is furthered by the results of running a Kernel Density Estimation on the same electrode data.  As seen below, the frequency of Delta waves is skewed to the right when compared to the control, corroborating the evidence of the scatter plots. 

 <img width="468" alt="image" src="https://user-images.githubusercontent.com/8241982/182700078-b8dc9830-2346-4898-bbdb-e797006a6436.png">
 
This apparent difference immediately tells us that the delta waves are worth investigating further. Thus, we took our machine learning approach to build a predictive model that analyzes the delta waves in the FP1 and FP2 electrodes.

### Clustering analysis
 
 We started clustering analysis by first performing the algorithms on all 114 measurements with our reduced dataset where each graph is the distortion against the number of clusters generated by the following algorithm:
 

```ruby
def KMeans_analysis(K, data):
  distortion = []
  predictions = []
  x_axis = np.arange(1,K+1)
  for i in range(1,K+1):
    km = KMeans(n_clusters=i)
    km.fit(data)
    distortion.append(km.inertia_)
    predictions.append(km.predict(data))
  return (x_axis, distortion, predictions)
```
 
Note: The graphs were all performed on PCA reduced datasets of varying variance, with the top graph being 85 percent, and the bottom being 95

 ![Elbow plot of kmeans clustering](Project%20images/Kmeans%20clustering%20on%20full%20data.png)
 
It becomes evident quickly that the elbow plots generated by kmeans are very smooth, not seeming to level off at any particular point on the x axis, indicating that kmeans is having trouble clustering the data.

Its possible that the shape of the data is not circular but more eliptical, so we attempted to perform a GMM analysis instead. To measure the effectiveness of GMM, we calculated some external measurements, mainly fowlkes score and the normalized mutual information, and graphed the scores against the number of componenets per mixture mode.

![NMI and FMS against num componenets](Project%20images/GMM%20prediction%20on%20fulldata.png)

The results of the GMM suggested that the soft clustering provided little similarity with the ground truth clustering (FMS) and that the predictions and true labels were virtually independent, never breaking past even .1.

**The results of the first bout of unsupervised learning demonstrated that the dataset was incredibly noisy and needed further cleaning**

As mentioned in the methods, subdata sets were generated, comparing each psychiatric disease against the healthy control for a total of 12 different subdata sets.

```ruby
def disorder_vs_control_electrode(data, label, disorders, control):
    newData = []
    for value in disorders:
        #print(value)
        a = data.loc[data[label].isin([value, control])]
        newData.append(a)
        #print(newData)
    return newData
```
 
 


## Supervised results

In the medical field, Support Vector Machines (SVMs) are a popular method of supervised machine learning. This is due to the fact SVMs specialize in binary classification, resulting in a robust medical tool to use for predictive diagnosis. 

The results of applying an SVM to the above dataset is shown below. We first used a linear kernel, but the results yielded a poor ~52% accuracy rate on the test data. This makes sense, given the abysmal boundaries of classification yielded by SVC w/ linear kernel and LinearSVC. Applying an RBF Kernel yielded even worse results, as the decision boundaries were nearly non-existent. However, when a polynomial kernel was used, the accuracy of the model increased by an enormous 15%, averaging at around 71% accuracy. The bottom right figure illustrates why this was the case; the decision boundaries are much more nuanced and capture the pattern of the control and depression data.   

<img width="468" alt="image" src="https://user-images.githubusercontent.com/8241982/182703513-d5c0f790-beb4-4f6a-a2d6-2c1831687262.png">


Other Supervised ML Algorithms used were:

Stochastic Gradient Descent (SGD): 71.4% Accuracy

Multilayer Neural Network (MPL): 58% Acucuracy

Decision Tree: 62% Accuracy








## Discussion

### Feature Selection/Engineering/Reduction
An important aspect of the study that needed investigation was the correlation of EEG signal between opposing electrodes. For example, we needed to verify that the FP1 electrode (right side front parietal) correlated with the FP2 eletrode (left side front parietal). This was verified by the red and blue scatter plot displayed in the Results section. 

The specific features that were chosen were the frontal electrodes, as the frontal brain regions are the most important regions of interest when researching depressive disorders. Additionally, we decided to drop all coherence features due to the distortive impact it had on various learning algorithms, including clustering analysis and GMM. 

Furthermore, for supervised learning, the Alpha, Theta, and Beta wave features were all dropped from the frontal electrodes. We decided to use this approach in order to maximize saliency of results: as seen in the scatter plots in the Results section, the delta waves had the least overlap for depression vs control in FP1 and FP2 electrodes. This illustrated that the delta waves were worth investigating further. 

We used PCA for our dimensionality reduction because we already had to standardize our features beforehand for other reasons. Thus, it made the most sense to implement PCA, since the dataset was already normalized/standardized. Additionally, we used PCA is order to remove correlated features, such as overlapping waveforms. Moreover, the ease-of-use of PCA along with the ease of visualising the results made the algorithm a particularly attractive option for dimensionality reduction.


### Supervised Learning Methods
The poor results given by the MLP are likely due to the lack of training data – there were only around 400 training points for the network to learn from. If more training data (several thousands) was present, the network would likely have performed better. 
The decision tree did not perform well due to the number of features in the dataset, as the algorithm usually breaks down once a large number of features is reached. 

Interestingly, the SGD algorithm, on average, performed the best out of all algorithms. This is likely due to the nature of SGD, and how it is less likely to get stuck in local minima of the loss function (shown below).

![image](https://user-images.githubusercontent.com/8241982/182924234-4b12ac12-c3b8-440e-aed0-c89fde28ba1d.png)

(from https://towardsdatascience.com/why-visualize-gradient-descent-optimization-algorithms-a393806eee2)

This is due to the fact that SGD is online and updates frequently, relative to other learning algorithms. For this reason, the SGD performs particularly well on large datasets such as the EEG dataset used in this study. The equation used in SGD is shown below.

<img width="965" alt="Stochastic" src="https://user-images.githubusercontent.com/8241982/182955637-63d004e6-068b-4729-b18e-c463fefb13c2.png">


----------------------------------------------------------------------------------------------------------------------------------------------------------------------






