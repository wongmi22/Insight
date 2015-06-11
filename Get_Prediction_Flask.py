import MySQLdb
import MySQLdb.cursors
import pandas as pd
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import sys

def calculate_predictions(opponent,player,game_diff,home):
    if home == 'home':
        home = '0'
    elif home == 'away':
        home = '1'
    else:
        print 'Please enter \'home\' or \'away\''
    
    player = player.lower()
    opponent = opponent.lower()
    
    conn = MySQLdb.connect(
        user="root",
        passwd="",
        db="Player_Team_Data",
        cursorclass=MySQLdb.cursors.DictCursor)

    cmd_target = 'SELECT PTS,3P,TRB,AST,STL,BLK,TOV FROM NBA_player_data WHERE Player_Name IN (\'' + player + '\');'
    cmd_train = 'SELECT Rk,Home_Away,DateDiff,TeamID,Win,OPPG,OTPR,O3Ppercent,ORPG,OBPG,OSPG,DEF,O3PM,OFGpercent,OTPG,OAPG,OFTpercent,TPG,SPG,TRBR,OBLKpercent FROM NBA_player_data WHERE Player_Name IN (\'' + player + '\');'
    cmd_truth = 'SELECT FGpercent,PTS,3P,AST,TRB,STL,BLK,TOV FROM NBA_player_data WHERE Player_Name IN (\'' + player + '\') AND Rk >50 AND Rk <60 ;'
    cmd_test = 'SELECT Rk,Home_Away,DateDiff,TeamID,Win,OPPG,OTPR,O3Ppercent,ORPG,OBPG,OSPG,DEF,O3PM,OFGpercent,OTPG,OAPG,OFTpercent,TPG,SPG,TRBR,OBLKpercent FROM NBA_player_data WHERE Player_Name IN (\'' + player + '\') AND Rk >50 AND Rk <60;'
    cmd_operate = 'SELECT TeamID,Win,OPPG,OTPR,O3Ppercent,ORPG,OBPG,OSPG,DEF,O3PM,OFGpercent,OTPG,OAPG,OFTpercent,TPG,SPG,TRBR,OBLKpercent FROM NBA_player_data WHERE Opponent_Team IN (\'' + opponent + '\') AND Month = 4 LIMIT 1;'

    df_target = pd.read_sql(cmd_target, con=conn)    
    df_train = pd.read_sql(cmd_train,con=conn)
    df_truth= pd.read_sql(cmd_truth, con=conn)    
    df_test = pd.read_sql(cmd_test,con=conn)
    df_operate = pd.read_sql(cmd_operate,con=conn)
    
    df_target = df_target.applymap(lambda x:float(x))
    df_train = df_train.applymap(lambda x:float(x))
    df_truth = df_truth.applymap(lambda x:float(x))
    df_test = df_test.applymap(lambda x:float(x))
    conn.close()
    
    df_operate['Rk']= 83
    df_operate['Home_Away']= home
    df_operate['DateDiff'] = game_diff
    df_inquire = df_operate.applymap(lambda x:float(x))
    df_train_plus_inquire=pd.concat([df_train, df_inquire])
    df_raw = df_train_plus_inquire.reindex()
    df_raw_scaled = df_raw.copy()
    df_raw_pure = df_raw.copy()
    df_raw_transform = df_raw.copy()
    df_raw_scaled = df_raw_scaled.apply(lambda x:preprocessing.StandardScaler().fit(x).transform(x))
    df_raw_transform = df_raw_transform.apply(lambda x:preprocessing.StandardScaler().fit(x))
    df_evaluate = df_raw_scaled.tail(1)
    df_train_scaled = df_raw_scaled.iloc[:-1]
    rf = RandomForestClassifier(n_estimators=1000)
    average_stats=df_target.mean()
    rf.fit(df_train_scaled, df_target)
    predictions = rf.predict(df_evaluate)
    return predictions,average_stats.round()
    
def get_fanduel(predict,avg):
    fanduel_pre = predict[0]+predict[2]*1.2+predict[3]*1.5+predict[4]*2+predict[5]*2-predict[6]
    fanduel_avg = avg[0]+avg[2]*1.2+avg[3]*1.5+avg[4]*2+avg[5]*2-avg[6]
    fanduel_outlook = fanduel_pre - fanduel_avg
    return fanduel_pre, fanduel_avg, fanduel_outlook

def get_boxscore_diff(predict, avg):
    diff = predict[0] - avg
    return diff
    
def get_prediction(opponent,player, game_diff, home):
    prediction,avg_stats=calculate_predictions(opponent,player,game_diff,home)  
    predict_fanduel, average_fanduel, outlook_fanduel=get_fanduel(prediction[0],avg_stats)
    outlook = get_boxscore_diff(prediction,avg_stats)
    return prediction,avg_stats,outlook,predict_fanduel, average_fanduel, outlook_fanduel 
  

