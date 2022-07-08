![MLinfo](https://user-images.githubusercontent.com/90532657/174424871-785ce7d2-3e6d-46a6-b7aa-ce24235c9a5f.PNG)

# Biomarker-search
Machine learning project for CS4641 at Georgia Institute of Technology

Collab: Saif Syed Ali, Ibraheem Abutair, Matthew Johnson, Micah Grimes

Introduction
The objective of this project is to use machine learning to gain insight on physical biomarkers present in the brain of patients diagnosed with major depressive disorder (MDD). Our inclusion criteria is as follows: subject must have been diagnosed with MDD by a medical professional and must be having a current episode of major depression OR has had an episode of major depression in the past 6 months. 
The current research in this field is plentiful, but very few have used machine learning on imaging data in order to generate patterns. We believe that machine learning is especially useful in this field because it allows for the standardization, comparison, and subsequent analysis of datasets that come from different metrics/studies. Succesful completion of this project would allow us to provide physicians and researchers with possible predictive measures for the development of MDD. This would allow physicians to mark certain individuals as "high-risk" for MDD and can provide preventative care to the patient.

Methods
The dataset type used will be numerical data taken from the Archives of General Psychiatry. The nature of the sets is processed MRI data from individuals with MDD. The entire data sheet contains MRI results from 225 studies, which will be more than enough to run machine learning algorithms on. Each table from the dataset has 17 features. The targets will be discrete, as we hope to use these targets to determine specific biomarkers of MDD.
Since the data was already standardized by the researchers (quality MRI studies will use a standardized template on which to "map" the brain of the subject onto), standardization should not be an issue. We will attempt to perform dimensionality reduction via a covariance filter. 
We currently plan on mainly using K-means clustering (unsupervised) to analyze the data and potential patterns that may emerge. We will also employ linear regression (supervised) in order to draw relationships between variables and draw predictions based on said relationships.

Results
We expect to find certain features in specific brain regions that are consistent across most study subjects. A metric of success would be a high level of correlation between one or more physical brain anomolies and a corresponding psychiatric diagnosis. For example, a measure of success could be a high degree of correlation between grey matter volume in the caudate nucleus and recurrance of MDD. 

Discussion
Using machine learning, we hope to be able to advance the understanding of the physical biomarkers associated with major depressive disorder. Using a database of images, we are able to distinguish unique similarities between patients. 
MRI imaging makes it possible to use imaging as a data type, however there are some issues with MRI imaging only holding greyscale information as well as some images maybe cloudy. We would also need to corrected positioning/perspective differences in images.
Not much research has been done in this field using machine learning. We hope to advance this method in that field and find very useful results. 
Information on biomarkers for MDD can be very beneficial to doctors and researchers, but most of all to the patients suffering from the condition.
The only risk associated with this work would be a waste of time and resources, which is well worth the risk for the oppurtunity this offers.
The run times for the algorithms will vary based on the algorithm created but should not be longer than a day for each runtime and a couple of weeks until ready for deployment.


MIDTERM REPORT:

Introduction
The objective of this project is to use machine learning to gain insight on physical biomarkers present in the brain of patients diagnosed with major depressive disorder
(MDD) and other similar psychiartic disorders. Our inclusion criteria is as follows: subject must have been diagnosed with MDD and/or other psychiatric disorders by a 
medical professional and must be having a current mental episode OR in the past 6 months. 

The current research in this field is plentiful, but very few have used machine learning on imaging data in order to generate patterns. We believe that machine learning
is especially useful in this field because it allows for the standardization, comparison, and subsequent analysis of datasets that come from different metrics/studies. 
Succesful completion of this project would allow us to provide physicians and researchers with possible predictive measures for the development of MDD and similar disorders.
This would allow physicians to mark certain individuals as "high-risk" for physiomental health issues and can provide preventative care to the patient.

Methods
The dataset type used will be numerical data taken from the Department of Psychiatry of Boramae Medical Center. The nature of the sets is processed EEGs signals from 
individuals with psychiatric disorders. The entire data sheet contains EEG results from 945 subjects, which will be more than enough to run machine learning algorithms on. 
Each table from the dataset has 400 features. The targets will be discrete, as we hope to use these targets to determine specific biomarkers of psychiatric disorders.
Since the data was already standardized by the researchers (quality MRI studies will use a standardized template on which to "map" the brain of the subject onto), 
standardization should not be an issue. We will attempt to perform dimensionality reduction via a covariance filter. 
We currently plan on mainly using K-means clustering (unsupervised) to analyze the data and potential patterns that may emerge. We will also employ linear regression
(supervised) in order to draw relationships between variables and draw predictions based on said relationships.

Results - Updated

We have employed a "trial and error" approach to our selection of datasets. While we started out with a focus on fMRI data, we quickly found that fMRI/MRI 
datasets are hard to come by. Most datasets do not deal directly with major depressive disorder (MDD), and the ones that do are unusable for a variety of 
reasons, ranging from lack of data to a lack of normalization between brain scans/brain regions. Additionlly, we found it exceedingly difficult
and out of scope to attempt to write a machine learning algorithm that utilizes computer vision to analyze brain scans. We explored a variety of more 'psychiatric'
data sets that involved rating scales for MDD symptoms alongside demographic data. While in some data sets, we were able to find clusters/corellations, 
it was all known info that could already be discerned with statistical analysis. Similar data sets we examined that did not already possess published
findings yieled no clusters/corellations. After concluding we could not work with psyciatric sets, we decided to shift our focus to numeral MRI datasets. 
With this dataset, we ran into a different set of issues. Namely, normalization appeared to be an issue again - comparing MRI data from different studies and 
different brain regions would run the risk of faulty or inconclusive results, due to inconsistent metrics of data collection, ranging from MRI machine tonnage 
to individual differences in brain structure.

We finally ended up looking at an EEG dataset. EEG data is usually easier to work with than MRI data, as EEG data is less visual and more numerical. 
Additionally, the brain waves recorded from EEG. Early results from PCA are promising, but we plan to experiment with a variety of methods. 

EEG data report: Due to the extremely large dimensionality of the data set, measuring nearly 1000 features, finding a good way to visualize our results has become a point of tension. PCA while 
useful, still generates around 40 to 60 principle componenets on the range of 90 to 95 percent of the variance. Clustering has been performed but analysis is still underway.

