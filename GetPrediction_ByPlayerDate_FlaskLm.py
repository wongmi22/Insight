
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
import re


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
    absent_dict={}
    dummy_dict={}
    for player in players:
        if player in player_list:
            cmd_Rk = 'SELECT Rk, Opp,Opponent_Team, Home_Away, DateDiff FROM NBA_player_data WHERE Player_Name IN (\'' + player + '\') AND Month IN (\'' + month + '\') AND Day IN (\'' + date + '\') AND Year IN (\'2015\');'
            df_Rk = pd.read_sql(cmd_Rk, con=conn)  
            Rk=str(int(df_Rk.Rk))
            Team=str(df_Rk.Opponent_Team.ix[0])
            Opponent=str(df_Rk.Opp.ix[0])
            Court=int(df_Rk.Home_Away)
            if Court == 1:
                Court = 'Home'
            elif Court == 2:
                Court = 'Away'
            DaysOff=str(int(df_Rk.DateDiff))

            cmd_target_2015 = 'SELECT PTS,3P,TRB,AST,STL,BLK,TOV FROM NBA_player_data_MP WHERE Player_Name IN (\'' + player + '\') AND Year IN (\'2015\') AND Rk < '+Rk+' ;'
            cmd_target_2014 = 'SELECT PTS,3P,TRB,AST,STL,BLK,TOV FROM NBA_player_data_MP WHERE Player_Name IN (\'' + player + '\') AND Year IN (\'2014\') AND Rk >= '+Rk+' ;'
            cmd_train_2015 = 'SELECT MP,Rk,Home_Away,DateDiff,TeamID,Win,OPPG,OTPR,O3Ppercent,ORPG,OBPG,OSPG,DEF,O3PM,OFGpercent,OTPG,OAPG,TPG,SPG,TRBR,OBLKpercent FROM NBA_player_data_MP WHERE Player_Name IN (\'' + player + '\') AND Year IN (\'2015\') AND Rk < '+Rk+';'
            cmd_train_2014 = 'SELECT MP,Rk,Home_Away,DateDiff,TeamID,Win,OPPG,OTPR,O3Ppercent,ORPG,OBPG,OSPG,DEF,O3PM,OFGpercent,OTPG,OAPG,TPG,SPG,TRBR,OBLKpercent FROM NBA_player_data_MP WHERE Player_Name IN (\'' + player + '\') AND Year IN (\'2014\') AND Rk >= '+Rk+';'
            cmd_operate = 'SELECT Rk,Home_Away,DateDiff,TeamID,Win,OPPG,OTPR,O3Ppercent,ORPG,OBPG,OSPG,DEF,O3PM,OFGpercent,OTPG,OAPG,TPG,SPG,TRBR,OBLKpercent FROM NBA_player_data_MP WHERE Player_Name IN (\'' + player + '\') AND Year IN (\'2015\') AND Rk = '+Rk+';'
            cmd_truth = 'SELECT PTS,3P,TRB,AST,STL,BLK,TOV FROM NBA_player_data_MP WHERE Player_Name IN (\'' + player + '\') AND Year IN (\'2015\') AND Rk = '+Rk+' ;'
            cmd_min_2015 = 'SELECT MP FROM NBA_player_data_MP WHERE Player_Name IN (\'' + player + '\') AND Year IN (\'2015\') AND Rk < '+Rk+';'   
            cmd_min_2014 = 'SELECT MP FROM NBA_player_data_MP WHERE Player_Name IN (\'' + player + '\') AND Year IN (\'2015\') AND Rk >= '+Rk+';'   

            df_min_2015 = pd.read_sql(cmd_min_2015, con=conn) 
            df_min_2014 = pd.read_sql(cmd_min_2014, con=conn) 
            df_target_2015 = pd.read_sql(cmd_target_2015, con=conn) 
            df_target_2014 = pd.read_sql(cmd_target_2014, con=conn) 
            df_train_2015 = pd.read_sql(cmd_train_2015, con=conn) 
            df_train_2014 = pd.read_sql(cmd_train_2014, con=conn) 
            df_operate = pd.read_sql(cmd_operate, con=conn) 
            df_truth = pd.read_sql(cmd_truth, con=conn) 
            df_truth = df_truth.applymap(lambda x: float(x))
            
            df_min = pd.concat([df_min_2014, df_min_2015],ignore_index=True).applymap(lambda x: float(x))
            mean_MP = df_min.mean().values[0]
            
            df_operate['MP'] = mean_MP
            
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

            predictions = np.asarray([pPTS,pTP,pREB,pAST,pSTL,pBLK,pTOV])


            if int(Rk) < 20:
                average_stats=df_target.mean()   
            elif int(Rk) >= 20:
                average_stats=df_target_2015.mean()
            predict_dict[player] = predictions.round()
            averages_dict[player] = average_stats.round()
            team_dict[player]=[Team, Opponent,Court, DaysOff]
        else:
            print 'Sorry', player, 'is not playing on this day \n'
            match = re.search('([\w\.\-]+) ([\w\.\-]+)', player)
            absent_dict[player] = '../static/Player_Photos/'+match.group(1)+'_'+match.group(2) +'.png' 
            dummy_dict[player] = 2
    conn.close()
    return predict_dict, averages_dict,team_dict,absent_dict,dummy_dict


def get_fanduel(predict,avg):
    fanduel_pre = predict[0]+predict[2]*1.2+predict[3]*1.5+predict[4]*2+predict[5]*2-predict[6]
    fanduel_avg = avg[0]+avg[2]*1.2+avg[3]*1.5+avg[4]*2+avg[5]*2-avg[6]
    fanduel_outlook = fanduel_pre - fanduel_avg
    return fanduel_pre, fanduel_avg, fanduel_outlook


# In[134]:

def get_boxscore_diff(predict, avg):
    diff = predict - avg
    return diff


def get_results_team(month,date,metric,players):
  
    
    if metric == 'absolute':
        Fmetric = 'Fmetric'
    elif metric == 'relative': 
        Fmetric = 'FMetric_Outlook'
    prediction, avg_stats, teams, absent, dummy = get_predictions(month,date,players)

    fanduel_dict = {}
    boxscore_diff = {}
    picture_dict ={}
    picture_team_dict={}

    for name in prediction:
        fanduel_pre, fanduel_avg, fanduel_outlook=get_fanduel(prediction[name][0],avg_stats[name])
        fanduel_dict[name] = [fanduel_pre, fanduel_avg, fanduel_outlook]
        diff=get_boxscore_diff(prediction[name][0],avg_stats[name])
        boxscore_diff[name] = diff
        match = re.search('([\w\.\-]+) ([\w\.\-]+)', name)
        match.group(1), match.group(2)
        picture_dict[name] = '../static/Player_Photos/'+match.group(1)+'_'+match.group(2) +'.png' 
        picture_team_dict[name] = '../static/Team_Photos/'+ teams[name][1]+'.png' 

    print picture_team_dict

    big_dict={}
    abs_dict={}
    for name in prediction:
        big_dict[name]=[prediction[name],np.asarray(avg_stats[name]),np.asarray(boxscore_diff[name]),np.asarray(fanduel_dict[name]),teams[name],fanduel_dict[name][0],fanduel_dict[name][2],picture_dict[name],picture_team_dict[name]]
    for name in absent:
        abs_dict[name]=[absent[name],dummy[name]]

    df_big=pd.DataFrame(big_dict, index = ['Prediction','Average','Outlook','Fanduel','Conditions','Fmetric','FMetric_Outlook','Photo','Team_Photo'])
    df_big=df_big.T
    df_big_sorted = df_big.sort([Fmetric],ascending=False)
    df_absent = pd.DataFrame(abs_dict, index =['Photo', 'Dummy'])
    df_absent = df_absent.T
    return df_big_sorted, df_absent
  # +++your code here+++
  # Call your functions
  



# In[ ]:



