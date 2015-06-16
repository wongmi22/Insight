
# coding: utf-8

# In[130]:

import MySQLdb
import MySQLdb.cursors
import pandas as pd
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import sys


def get_predictions(month, date, players):
    month = str(month)
    date = str(date)
    conn = MySQLdb.connect(
            user="root",
            passwd="",
            db="Player_Team_Data",
            cursorclass=MySQLdb.cursors.DictCursor)

    cmd_date= 'SELECT Player_Name, Opponent_Team FROM NBA_player_data WHERE Month IN (\'' + month + '\') AND Day IN (\'' + date + '\') AND Year IN (\'2015\');'
    df_players = pd.read_sql(cmd_date, con=conn) 
    player_list = list(df_players.Player_Name)

    predict_dict = {}
    averages_dict ={}
    team_dict={}
    for player in players:
        if player in player_list:
            cmd_Rk = 'SELECT Rk, Opponent_Team, Home_Away, DateDiff FROM NBA_player_data WHERE Player_Name IN (\'' + player + '\') AND Month IN (\'' + month + '\') AND Day IN (\'' + date + '\') AND Year IN (\'2015\');'
            df_Rk = pd.read_sql(cmd_Rk, con=conn)  
            Rk=str(int(df_Rk.Rk))
            Team=str(df_Rk.Opponent_Team.ix[0])
            Court=int(df_Rk.Home_Away)
            if Court == 1:
                Court = 'Home'
            elif Court == 2:
                Court = 'Away'
            DaysOff=str(int(df_Rk.DateDiff))

            cmd_target_2015 = 'SELECT PTS,3P,TRB,AST,STL,BLK,TOV FROM NBA_player_data WHERE Player_Name IN (\'' + player + '\') AND Year IN (\'2015\') AND Rk < '+Rk+' ;'
            cmd_target_2014 = 'SELECT PTS,3P,TRB,AST,STL,BLK,TOV FROM NBA_player_data WHERE Player_Name IN (\'' + player + '\') AND Year IN (\'2014\') AND Rk >= '+Rk+' ;'
            cmd_train_2015 = 'SELECT Rk,Home_Away,DateDiff,TeamID,Win,OPPG,OTPR,O3Ppercent,ORPG,OBPG,OSPG,DEF,O3PM,OFGpercent,OTPG,OAPG,TPG,SPG,TRBR,OBLKpercent FROM NBA_player_data WHERE Player_Name IN (\'' + player + '\') AND Year IN (\'2015\') AND Rk < '+Rk+';'
            cmd_train_2014 = 'SELECT Rk,Home_Away,DateDiff,TeamID,Win,OPPG,OTPR,O3Ppercent,ORPG,OBPG,OSPG,DEF,O3PM,OFGpercent,OTPG,OAPG,TPG,SPG,TRBR,OBLKpercent FROM NBA_player_data WHERE Player_Name IN (\'' + player + '\') AND Year IN (\'2014\') AND Rk >= '+Rk+';'
            cmd_operate = 'SELECT Rk,Home_Away,DateDiff,TeamID,Win,OPPG,OTPR,O3Ppercent,ORPG,OBPG,OSPG,DEF,O3PM,OFGpercent,OTPG,OAPG,TPG,SPG,TRBR,OBLKpercent FROM NBA_player_data WHERE Player_Name IN (\'' + player + '\') AND Year IN (\'2015\') AND Rk = '+Rk+';'
            cmd_truth = 'SELECT PTS,3P,TRB,AST,STL,BLK,TOV FROM NBA_player_data WHERE Player_Name IN (\'' + player + '\') AND Year IN (\'2015\') AND Rk = '+Rk+' ;'

            df_target_2015 = pd.read_sql(cmd_target_2015, con=conn) 
            df_target_2014 = pd.read_sql(cmd_target_2014, con=conn) 
            df_train_2015 = pd.read_sql(cmd_train_2015, con=conn) 
            df_train_2014 = pd.read_sql(cmd_train_2014, con=conn) 
            df_operate = pd.read_sql(cmd_operate, con=conn) 
            df_truth = pd.read_sql(cmd_truth, con=conn) 


            df_target=pd.concat([df_target_2014, df_target_2015],ignore_index=True)
            df_train=pd.concat([df_train_2014, df_train_2015],ignore_index=True)
            df_target = df_target.applymap(lambda x:float(x))
            df_train = df_train.applymap(lambda x:float(x))
            df_target_2015 = df_target_2015.applymap(lambda x: float(x))

            df_inquire = df_operate.applymap(lambda x:float(x))
            df_train_plus_inquire=pd.concat([df_train, df_inquire])
            df_raw = df_train_plus_inquire.reindex()
            df_raw_scaled = df_raw.copy()
            df_raw_scaled = df_raw_scaled.applymap(lambda x: np.log(x))

            df_raw_transform = df_raw.copy()
            df_raw_scaled = df_raw_scaled.apply(lambda x:preprocessing.StandardScaler().fit(x).transform(x))
            df_raw_transform = df_raw_transform.apply(lambda x:preprocessing.StandardScaler().fit(x))
            df_evaluate = df_raw_scaled.tail(1)
            df_train_scaled = df_raw_scaled.iloc[:-1]

            rf = RandomForestRegressor(n_estimators=100)
            
            rf.fit(df_train_scaled, df_target)
            predictions = rf.predict(df_evaluate)
            if int(Rk) < 20:
                average_stats=df_target.mean()   
            elif int(Rk) >= 20:
                average_stats=df_target_2015.mean()
            predict_dict[player] = predictions.round()
            averages_dict[player] = average_stats.round()
            team_dict[player]=[Team, Court, DaysOff]
        else:
            print 'Sorry', player, 'is not playing on this day \n'
    conn.close()
    return predict_dict, averages_dict,team_dict


def get_fanduel(predict,avg):
    fanduel_pre = predict[0]+predict[2]*1.2+predict[3]*1.5+predict[4]*2+predict[5]*2-predict[6]
    fanduel_avg = avg[0]+avg[2]*1.2+avg[3]*1.5+avg[4]*2+avg[5]*2-avg[6]
    fanduel_outlook = fanduel_pre - fanduel_avg
    return fanduel_pre, fanduel_avg, fanduel_outlook


# In[134]:

def get_boxscore_diff(predict, avg):
    diff = predict - avg
    return diff


def main():
  
    args = sys.argv[1:]
    if not args:
        print "usage: [--month month] [--date day]"
        sys.exit(1)

    month = str(args[0]) 
    date = str(args[1])
    metric = args[2]
    players = args[3:]
    if metric == 'absolute':
        Fmetric = 'Fmetric'
    elif metric == 'relative': 
        Fmetric = 'FMetric_Outlook'
    prediction, avg_stats, teams = get_predictions(month,date,players)

    fanduel_dict = {}
    boxscore_diff = {}

    for name in prediction:
        fanduel_pre, fanduel_avg, fanduel_outlook=get_fanduel(prediction[name][0],avg_stats[name])
        fanduel_dict[name] = [fanduel_pre, fanduel_avg, fanduel_outlook]
        diff=get_boxscore_diff(prediction[name][0],avg_stats[name])
        boxscore_diff[name] = diff

    big_dict={}

    for name in prediction:
        big_dict[name]=[prediction[name],np.asarray(avg_stats[name]),np.asarray(boxscore_diff[name]),np.asarray(fanduel_dict[name]),teams[name],fanduel_dict[name][0],fanduel_dict[name][2]]
    
    df_big=pd.DataFrame(big_dict,index = ['Prediction','Average','Outlook','Fanduel','Conditions','FMetric','FMetric_Outlook'])
    df_big=df_big.T
    df_big_sorted = df_big.sort([Fmetric],ascending=False)

    for name in df_big_sorted.index:
        print 'Name: ', name, ' vs.', df_big_sorted.Conditions[name][0], ' ', df_big_sorted.Conditions[name][1], 'Game with', df_big_sorted.Conditions[name][2], 'days rest \n' 
        print 'Predict: PTS: ', df_big_sorted.Prediction[name][0][0], ' 3P: ', df_big_sorted.Prediction[name][0][1], ' TRB: ', df_big_sorted.Prediction[name][0][2], ' AST: ', df_big_sorted.Prediction[name][0][3], ' STL: ', df_big_sorted.Prediction[name][0][4], ' BLK: ', df_big_sorted.Prediction[name][0][5], ' TOV: ', df_big_sorted.Prediction[name][0][6]
        print 'Average: PTS: ', df_big_sorted.Average[name][0], ' 3P: ', df_big_sorted.Average[name][1], ' TRB: ', df_big_sorted.Average[name][2], ' AST: ', df_big_sorted.Average[name][3], ' STL: ', df_big_sorted.Average[name][4], ' BLK: ', df_big_sorted.Average[name][5], ' TOV: ', df_big_sorted.Average[name][6]
        print 'Outlook: PTS: ', df_big_sorted.Outlook[name][0], ' 3P: ', df_big_sorted.Outlook[name][1], ' TRB: ', df_big_sorted.Outlook[name][2], ' AST: ', df_big_sorted.Outlook[name][3], ' STL: ', df_big_sorted.Outlook[name][4], ' BLK: ', df_big_sorted.Outlook[name][5], ' TOV: ', df_big_sorted.Outlook[name][6]
        print 'Fanduel (Predict): ', df_big_sorted.Fanduel[name][0]
        print 'Fanduel (Average): ', df_big_sorted.Fanduel[name][1]
        print 'Fanduel (Outlook): ', df_big_sorted.Fanduel[name][2], '\n'
  # +++your code here+++
  # Call your functions
  
if __name__ == "__main__":
  main()


# In[ ]:



