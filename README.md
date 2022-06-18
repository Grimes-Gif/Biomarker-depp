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


