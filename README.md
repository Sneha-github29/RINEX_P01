# Weather Data Classification-Determining Humidity
# importing the dataframe
import pandas as pd
df = pd.read_csv('daily_weather Dataset.csv')
df.head()
# to display last 5 rows
df.tail()
# to find the size of a dataframe
df.shape
#  to display all column labels of a dataframe
df.columns
#returns the number of missing values in the dataframe
df.isnull().sum()
# best workflow will be remove things from the dataframe without changing the original dataframe
new_df = df
del new_df['Unnamed: 11']
del new_df['number']
new_df.columns
import math
# Data Preprocessing - Handling Missing data by calculating the mean

mean_press_9am = math.floor(new_df['air_pressure_9am'].mean())
new_df['air_pressure_9am'] = new_df['air_pressure_9am'].fillna(mean_press_9am)

mean_temp_9am = math.floor(new_df['air_temp_9am'].mean())
new_df['air_temp_9am'] = new_df['air_temp_9am'].fillna(mean_temp_9am)

mean_avg_dir = math.floor(new_df['avg_wind_direction_9am'].mean())
new_df['avg_wind_direction_9am'] = new_df['avg_wind_direction_9am'].fillna(mean_avg_dir)

mean_avg_spd = math.floor(new_df['avg_wind_speed_9am'].mean())
new_df['avg_wind_speed_9am'] = new_df['avg_wind_speed_9am'].fillna(mean_avg_spd)

mean_max_dir = math.floor(new_df['max_wind_direction_9am'].mean())
new_df['max_wind_direction_9am'] = new_df['max_wind_direction_9am'].fillna(mean_max_dir)

mean_max_spd = math.floor(new_df['max_wind_speed_9am'].mean())
new_df['max_wind_speed_9am'] = new_df['max_wind_speed_9am'].fillna(mean_max_spd)

mean_rain_acc = math.floor(new_df['rain_accumulation_9am'].mean())
new_df['rain_accumulation_9am'] = new_df['rain_accumulation_9am'].fillna(mean_rain_acc)

mean_rain_dur = math.floor(new_df['rain_duration_9am'].mean())
new_df['rain_duration_9am'] = new_df['rain_duration_9am'].fillna(mean_rain_dur)

new_df.head()
new_df.isnull().sum()
# Extracting independent variable

X = new_df[['air_pressure_9am', 'air_temp_9am', 'avg_wind_direction_9am',
       'avg_wind_speed_9am', 'max_wind_direction_9am', 'max_wind_speed_9am',
       'rain_accumulation_9am', 'rain_duration_9am', 'relative_humidity_9am']]
X.head()
# Extracting dependent variable

new_df['humidity_3pm'] = (new_df['relative_humidity_3pm'] > 25)*1
y = new_df[['humidity_3pm']]
y.head()
X.columns
y.columns
X.shape
y.shape
# Splitting the Dataset into the Training set and Test set

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=10)
# Arranging objects into classes

from sklearn.tree import DecisionTreeClassifier
humidity_classifier = DecisionTreeClassifier(max_leaf_nodes=10, random_state=1)
humidity_classifier.fit(X_train, y_train)
type(humidity_classifier)
y_predicted = humidity_classifier.predict(X_test)
y_predicted[:10]
y_test['humidity_3pm'][:10]
# To calculate the accuracy score of the model

from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_predicted)*100
