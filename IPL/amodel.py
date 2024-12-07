import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import math

import warnings
warnings.simplefilter("ignore", UserWarning)

import numpy as np
import pandas as pd
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

data = pd.read_csv("data1.csv")

input_features = ['batting_team', 'bowling_team', 'venue']
encoder = OneHotEncoder(sparse_output=False,handle_unknown='ignore')
encoded_categorical = encoder.fit_transform(data[input_features])

xfeats = ['over', 'ball']
scalerx = StandardScaler()
scaled_x = scalerx.fit_transform(data[xfeats])

yfeats = ['runs', 'wickets']
scalery = StandardScaler()
scaled_y = scalery.fit_transform(data[yfeats])

X = np.hstack((encoded_categorical, scaled_x))
X = X.reshape(X.shape[0], 1, X.shape[1])
y = np.vstack((scaled_y))

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

'''model = keras.Sequential([
        keras.layers.Input(shape=(X_train.shape[1], X_train.shape[2])),
        keras.layers.LSTM(64, return_sequences=True),
        keras.layers.LSTM(32),
        keras.layers.Dense(16, activation='relu'),
        keras.layers.Dense(2)
    ])
    
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

history = model.fit(X_train, y_train, epochs=2, batch_size=16, validation_data=(X_val, y_val))
model.evaluate(X_val, y_val,batch_size=16)
model.save("lstm1.keras")'''

model = keras.models.load_model("lstm1.keras")
team = {'Chennai Super Kings': 1, 'Delhi Capitals': 2, 'Gujarat Titans': 3, 'Kolkata Knight Riders': 5, 'Lucknow Super Giants': 6, 'Mumbai Indians': 7,'Punjab Kings': 9, 'Rajasthan Royals': 10, 'Royal Challengers Bengaluru': 12, 'Sunrisers Hyderabad': 13}
venue = {'Arun Jaitley Stadium': 1, 'Barsapara Cricket Stadium': 3,'Dr DY Patil Sports Academy': 7,'Eden Gardens': 10, 'Ekana Cricket Stadium': 11, 'Feroz Shah Kotla': 12, 'M Chinnaswamy Stadium': 18, 'MA Chidambaram Stadium': 19, 'Maharaja Yadavindra Singh International Cricket Stadium': 20,'Narendra Modi Stadium': 22, 'Punjab Cricket Association IS Bindra Stadium': 27, 'Punjab Cricket Association Stadium': 28, 'Rajiv Gandhi International Stadium': 29, 'Sardar Patel Stadium': 30, 'Saurashtra Cricket Association Stadium': 31, 'Sawai Mansingh Stadium': 32, 'Sheikh Zayed Stadium': 35, 'Subrata Roy Sahara Stadium': 37, 'Vidarbha Cricket Association Stadium': 39, 'Wankhede Stadium': 40}

for key,value in team.items():
    print("Team = ",key)
    print("Team Code = ",value,"\n")
print("")
team1 = int(input("Choose the team 1 Code: "))
if team1 not in team.values():
    print("Error. Invalid Team Code.")
    quit()
team2 = int(input("Choose the team 2 Code: "))
if team2 not in team.values():
    print("Error. Invalid Team Code.")
    quit()
print("")
for key,value in venue.items():
    print("Stadium = ",key)
    print("Stadium Code = ",value,"\n")
print("")
venue1 = int(input("Choose the stadium Code: "))
if venue1 not in venue.values():
    print("Error. Inalid Stadium Code.")
    quit()
print("")
Input1 = [[team1,team2,venue1]]
Input2 = [[team2,team1,venue1]]

scaled_input1 = encoder.transform(Input1)
scaled_input2 = encoder.transform(Input2)

Extrainput = [[19,6]]
scaled_extrainput = scalerx.transform(Extrainput)

X_give1 = np.hstack((scaled_input1, scaled_extrainput))
X_give1 = X_give1.reshape(X_give1.shape[0], 1, X_give1.shape[1])

X_give2 = np.hstack((scaled_input2, scaled_extrainput))
X_give2 = X_give2.reshape(X_give2.shape[0], 1, X_give2.shape[1])

pred1 = model.predict(X_give1,verbose = 0)
pred1 = scalery.inverse_transform(pred1)

pred2 = model.predict(X_give2,verbose = 0)
pred2 = scalery.inverse_transform(pred2)

for key1, value in team.items():
    if team1 == value:
        team1_score = round(pred1[0, 0])
        team1_wickets = round(pred1[0, 1])
        team1_run_rate = team1_score / 20
        print(f"Score when {key1} is batting first = {team1_score}/{team1_wickets}, Run Rate: {team1_run_rate:.2f}")
        break

for key2, value2 in team.items():
    if team2 == value2:
        team2_score=round(pred2[0,0])
        team2_wickets=round(pred2[0,1])
        team2_run_rate = team2_score / 20
        if team2_score<team1_score:
            print(f"Score when {key2} is batting second = {team2_score}/{team2_wickets}, Run Rate: {team2_run_rate:.2f}")
            print(f"{key1} won by {team1_score - team2_score} runs")
        else:
            team2_scored=math.ceil(round(team1_score/team2_run_rate,1)*team2_run_rate)
            ballsfaced=int((team1_score/team2_run_rate)*6)
            oversface = int(ballsfaced// 6) 
            remaining_balls = ballsfaced % 6
            if team1_score==team2_scored and team2_run_rate>team1_run_rate:
                team2_scored+=1
            print(f"Score when {key2} is batting second = {team2_scored}/{team2_wickets}, Run Rate: {team2_run_rate:.2f}")
            print(f"{key2} chased down the score in {oversface}.{remaining_balls} overs")