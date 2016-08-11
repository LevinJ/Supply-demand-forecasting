# Rider-Driver Supply and Demand Gap Forecast
The Di-Tech Challenge(1) is organized by DiDi Chuxing, China’s largest ride-hailing company. It challenges contestants to use real data to generate predictive rider-driver supply and demand gap model, to direct drivers to where riders will need to be picked up.

Supply-demand forecasting helps to predict the gap of drivers and riders at a certain time period in a specific geographic area, and it is critical to enabling Didi to maximize utilization of drivers and ensure that riders can always get a car whenever and wherever they may need a ride.

This project report describes my solution for this competition.

The best score in Di-Tech challenge leaderboard is about 0.39. Our final model’s score is 0.42. 

A score of 0.42 is not really state of the art for this competition, but still it is quite respectable. This score gap of 0.02 might be further narrowed by experimenting some improvement ideas listed in improvement section of this document.

The target variable to be predicted Gapi,j+1 is a continuous variable, so we can attempt to solve this problem by regression algorithm. In this project, below algorithms are experimented:
1.	KNN(Sklearn)
2.	Random forest(Sklearn)
3.	Gradient Boosting machine (XGBoost)
4.	Neural network (Tensorflow)
Below major techniques are used in this project:
1.	Cross validation
As the problem is posed as a competition, the label for test dataset is not known to contestants.  Having a solid cross validation strategy will allow us to quickly experiment new ideas without being limited by official test dataset prediction submission, which occurs only once a day
Also the cross validation will reduce the chance of overfitting public leaderboard in the process of fine tuning models.
2.	Feature engineering
Original Input data is not directly usable and has to be transformed into a tabular format. Some new features will also be devised to improve model prediction capability.
3.	Feature selection
For some algorithms, greedy forward feature selection is used to select input features out of all the engineered features; while for other algorithms, all engineered features are used as input features.
4.	Grid search
Grid search is used to find optimal hyper parameter for models

For details, please refer to the project report Rider-Driver Supply and Demand Gap Forecast.pdf, which resides under the root directory of this github project.



Required libray to run the scripts
--------------
* Sklearn
* XGBOOST
* Sklearn
* Tensorflow

Instructions for running the scripts
--------------
1.  Download the source codes from github
2.	Get data files( data_preprocessed.zip and data_raw.zip) from the author of this project report
3.  Extract data_preprocessed.zip under root directory of the project. After extraction, all temporary preproces dump files will be under data_preprocessed folder
4.  Extract data_raw.zip under root direcoty of the project. After extraction, all raw files will be under data_preprocessed folder
5. 	Run scripts
	*Run implement/didineuralmodel.py file to train/validate neural network model.
	*Run implement/xgboostmodel.py file to train/validation GBM model
	*Run implement/knnmodel.py file to train/validate KNN model
	*Run implement/randomforestmodel.py file to train/validate random forest model
	*Run implement/forwardfeatureselection file to try out greedy forward feature selection
	*Run implement/tunemodels.py file to try out grid search for models

