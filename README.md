# Rider-Driver Supply and Demand Gap Forecast
The [Di-Tech Challenge](http://research.xiaojukeji.com/competition/main.action?competitionId=DiTech2016&&locale=en) is organized by DiDi Chuxing, China's largest ride-hailing company. It challenges contestants to use real data to generate predictive rider-driver supply and demand gap model, to direct drivers to where riders will need to be picked up.

When I learned of the challenge announcement, I just completed all of the required courses in Udacity Machine Learning Engineer Nanodegree. It's exciting to choose this competition as my Capstone project, practice and consolidate what I've learned throughout the Nanodegree program by tackling a real world problem, and what's more, in a highly competitive ongoing contest!

The development life cycle of this machine learning project can be summarized using below steps:

1.	Define the problem as a supervised regression learning problem
2.	Define MAPE as the metrics to evaluate the models
3.	Explore the data type, basic statistics and distribution of features and labels, perform univariate and bivariate visualizations to gain insight into the data, and guide feature engineering and cross validation strategy.
4.	Identify KNN, Random Forest, GBM Neural network as potential models/algorithms, Find out state of art benchmark that these models aims to reach/beat.
5.	Perform feature engineering, feature transformation, outlier and missing value handling.
6.	Implement models by leveraging Sklearn, XGBoost, and Tensorflow learning libraries.
7.	Fine tune the models via iterative feature selection/engineering, model selection, hyper parameter tuning. Cross validation is used to ensure that the models generalize well into unseen data.  

The best score in Di-Tech challenge leaderboard is about 0.39. Our final model's score is 0.42. A score of 0.42 is not really state of the art for this competition, but still it is quite respectable. This score gap of 0.03 might be further narrowed by experimenting some improvement ideas listed in improvement section of this document.

For details, please refer to the **[project report](https://github.com/LevinJ/Supply-demand-forecasting/blob/master/Rider-Driver%20Supply%20and%20Demand%20Gap%20Forecast.pdf)** Rider-Driver Supply and Demand Gap Forecast.pdf, which resides under the root directory of this github project.



Required library to run the scripts
--------------
* Sklearn
* XGBOOST
* Tensorflow

Instructions for running the scripts
--------------
1.  Download the source codes from [here](https://github.com/LevinJ/Supply-demand-forecasting)
2.	Get data files( data_preprocessed.zip and data_raw.zip) from [Dropbox shared link](https://www.dropbox.com/sh/33cfeiidegucins/AACdvKFkiyCcbqByBTl3wG8wa?dl=0)
3.  Extract data_preprocessed.zip under root directory of the project. After extraction, all temporary preprocessed dump files will be under data_preprocessed folder
4.  Extract data_raw.zip under root directory of the project. After extraction, all raw files will be under data_raw folder
5.	In the console/terminal, set implement as current directory
5. 	Run scripts  
	Run python didineuralmodel.py  to train/validate neural network model   
	Run python knnmodel.py  to train/validate KNN model  
	Run python randomforestmodel.py  to train/validate random forest model  
	Run python xgboostmodel.py  to train/validation GBM model  
	Run python forwardfeatureselection  to try out greedy forward feature selection  
	Run python tunemodels.py to try out grid search for models  

