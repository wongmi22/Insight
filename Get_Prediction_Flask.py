import MySQLdb
import MySQLdb.cursors
import pandas as pd
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import sys
from sklearn.linear_model import LinearRegression as LinR

def calculate_predictions(opponent,player,game_diff,home):
    if home == 'home':
        home = '1'
    elif home == 'away':
        home = '2'
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
    df_raw_scaled = df_raw_scaled.applymap(lambda x: np.log(x))
    df_raw_pure = df_raw.copy()
    df_raw_transform = df_raw.copy()
    df_raw_scaled = df_raw_scaled.apply(lambda x:preprocessing.StandardScaler().fit(x).transform(x))
    df_raw_transform = df_raw_transform.apply(lambda x:preprocessing.StandardScaler().fit(x))
    df_evaluate = df_raw_scaled.tail(1)
    df_train_scaled = df_raw_scaled.iloc[:-1]
    rf = RandomForestClassifier(n_estimators=1000)
    rf.fit(df_train_scaled, df_target)
    prediction_rf = rf.predict(df_evaluate)
    
    PTS = LinR()
    PTS.fit(df_train_scaled, df_target.PTS)
    pPTS = PTS.predict(df_evaluate)
    REB = LinR()
    REB.fit(df_train_scaled, df_target.TRB)
    pREB = REB.predict(df_evaluate)
    AST = LinR()
    AST.fit(df_train_scaled, df_target.AST)
    pAST = AST.predict(df_evaluate)
    TP = LinR()
    TP.fit(df_train_scaled, df_target['3P'])
    pTP = TP.predict(df_evaluate)
    STL = LinR()
    STL.fit(df_train_scaled, df_target.STL)
    pSTL = STL.predict(df_evaluate)
    BLK = LinR()
    BLK.fit(df_train_scaled, df_target.BLK)
    pBLK = BLK.predict(df_evaluate)
    TOV = LinR()
    TOV.fit(df_train_scaled, df_target.TOV)
    pTOV = TOV.predict(df_evaluate)
    
    pre_dict={'PTS':pPTS,'3P':pTP,'REB':pREB,'AST':pAST,'STL':pSTL,'BLK':pBLK,'TOV':pTOV}
    prediction=pd.DataFrame(pre_dict)
    
    prediction=prediction.applymap(lambda x:round(x))
    prediction[prediction < 0] = 0
    prediction_lm =prediction.applymap(lambda x:float(x))
    
    average_stats=df_target.mean()
    
    return prediction_rf, np.asarray(prediction_lm),average_stats.round()
    
def get_fanduel(predict,avg):
    
    fanduel_pre = predict[0]+predict[2]*1.2+predict[3]*1.5+predict[4]*2+predict[5]*2-predict[6]
    fanduel_avg = avg[0]+avg[2]*1.2+avg[3]*1.5+avg[4]*2+avg[5]*2-avg[6]
    fanduel_outlook = fanduel_pre - fanduel_avg
    return fanduel_pre, fanduel_avg, fanduel_outlook

def get_boxscore_diff(predict, avg):
    diff = predict[0] - avg
    return diff

def get_outlook_lm(predict,avg):
    
    diff = predict - avg;
    return diff
    
def get_prediction(opponent,player, game_diff, home):
    prediction_rf, prediction_lm,avg_stats=calculate_predictions(opponent,player,game_diff,home)  
    prediction_lm = np.asarray([prediction_lm[0][3],prediction_lm[0][0],prediction_lm[0][4],prediction_lm[0][1],prediction_lm[0][5],prediction_lm[0][2],prediction_lm[0][6]])
    predict_fanduel, average_fanduel, outlook_fanduel=get_fanduel(prediction_lm,avg_stats)
    outlook_rf = get_boxscore_diff(prediction_rf,avg_stats)
    outlook_lm = get_outlook_lm(prediction_lm,avg_stats)
    return prediction_rf, prediction_lm, avg_stats, outlook_rf, outlook_lm, predict_fanduel, average_fanduel, outlook_fanduel 
  

