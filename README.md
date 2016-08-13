# Rider-Driver Supply and Demand Gap Forecast
The Di-Tech Challenge(1) is organized by DiDi Chuxing, Chinaâ€™s largest ride-hailing company. It challenges contestants to use real data to generate predictive rider-driver supply and demand gap model, to direct drivers to where riders will need to be picked up.

Supply-demand forecasting helps to predict the gap of drivers and riders at a certain time period in a specific geographic area, and it is critical to enabling Didi to maximize utilization of drivers and ensure that riders can always get a car whenever and wherever they may need a ride.

When I learned of the challenge announcement, I just completed all the required courses in Udacity Machine Learning Nanodegree. It¡¯s exciting to choose this competition as my Capstone project, practice and consolidate what I¡¯ve learned throughout the Nanodegree program by tackling a real world problem, and what¡¯s more, in a highly competitive ongoing contest!

This project report describes my solution for this competition.

The best score in Di-Tech challenge leaderboard is about 0.39. Our final modelâ€™s score is 0.42. A score of 0.42 is not really state of the art for this competition, but still it is quite respectable. This score gap of 0.02 might be further narrowed by experimenting some improvement ideas listed in improvement section of this document.

For details, please refer to the project report Rider-Driver Supply and Demand Gap Forecast.pdf, which resides under the root directory of this github project.



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

