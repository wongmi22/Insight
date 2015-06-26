
# coding: utf-8

# In[95]:
from __future__ import division
import MySQLdb
import MySQLdb.cursors
import pandas as pd
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression as LinR
import numpy as np
import sys
import re
import matplotlib as plt
from sklearn.decomposition import PCA
import scipy as sp



def get_fanduel(predict,avg,tru):
    fanduel_pre = predict[0]+predict[2]*1.2+predict[3]*1.5+predict[4]*2+predict[5]*2-predict[6]
    fanduel_avg = avg[0]+avg[2]*1.2+avg[3]*1.5+avg[4]*2+avg[5]*2-avg[6]
    fanduel_tru = tru[0]+tru[2]*1.2+tru[3]*1.5+tru[4]*2+tru[5]*2-tru[6]
    return fanduel_pre, fanduel_avg, fanduel_tru



def get_trend(fan_pre,fan_avg,fan_tru):
    
    if (fan_pre > fan_avg) and (fan_tru > fan_avg):
        val = 1
    elif (fan_pre < fan_avg) and (fan_tru < fan_avg):
        val = 1
    elif (fan_pre < fan_avg) and (fan_tru > fan_avg):
        val = 0
    elif (fan_pre > fan_avg) and (fan_tru < fan_avg):    
        val = 0
    else:
        val = 2
    return val

def main():

    conn = MySQLdb.connect(
                user="root",
                passwd="",
                db="Player_Team_Data",
                cursorclass=MySQLdb.cursors.DictCursor)


    # In[97]:
    # read in AllPlayerNames .csv from basketball-reference
    df_all_players = pd.read_csv('~/Insight/Players100.csv')
        # Remove rows that were separated by random 'Player' entries
    df_all_players = df_all_players[df_all_players.Name != 'Player']
        
    name_list=list(df_all_players.values)
    new_name_list = []
        
    for name in name_list:
            # convert entries to strings
        name = str(name)
        new_name_list.append(name)
            
    unique_name_list=list(set(new_name_list))
    unique_name_list.sort()
    name_list_final = []
    for name in unique_name_list:
        match = re.search('([\w\.\-\']+) ([\w\.\-]+)', name)
        prename = match.group(1).lower()[1:] + ' ' + match.group(2).lower()
        name_list_final.append(prename)

    name_list_fin = [x for x in name_list_final if x != 'george hill']
    accuracy_list=[]
    for player in name_list_fin:
        print player
        cmd_Rk= 'SELECT Rk FROM NBA_player_data WHERE Player_Name IN (\'' + player + '\')  AND Year IN (\'2015\');'
        df_Rk = pd.read_sql(cmd_Rk, con=conn) 

        f_pre = []
        f_avg = []
        f_tru = []
        for Rki in df_Rk.Rk:
            Rk = str(Rki)
            print Rk
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
            df_truth = df_truth.applymap(lambda x: float(x))

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

            pca = PCA()
            pca.fit(df_raw_scaled)
            pca.n_components=7
            train_reduced = pca.fit_transform(df_raw_scaled)
            df_train_reduced=pd.DataFrame(train_reduced)
            df_evaluate = df_train_reduced.tail(1)
            df_train_scaled = df_train_reduced.iloc[:-1]


            # rf = RandomForestRegressor(n_estimators=100)
            # rf.fit(df_train_scaled, df_target)
            # predictions = rf.predict(df_evaluate).round()[0]
            
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
                #average_stats=df_target.mean().round()   
            elif int(Rk) >= 20:
                #average_stats=df_target_2015.mean().round()
                average_stats=df_target_2015.mean()
            average_stats=np.asarray(average_stats)
            true_stats = np.asarray(df_truth)[0]
            fanduel_pre, fanduel_avg, fanduel_tru = get_fanduel(predictions,average_stats,true_stats)
            f_pre.append(fanduel_pre)
            f_avg.append(fanduel_avg)
            f_tru.append(fanduel_tru)
            
        score =[]
        for i,f in enumerate(f_pre):
            s=get_trend(f,f_avg[i],f_tru[i])
            score.append(s)
        a = [x for x in score if x != 2]

        pts =[]
        for i,f in enumerate(predictions):
            s=get_trend(f,f_avg[i],f_tru[i])
            score.append(s)

        score =[]
        for i,f in enumerate(f_pre):
            s=get_trend(f,f_avg[i],f_tru[i])
            score.append(s)
        accuracy = a.count(1)/len(a)
    accuracy_list.append(accuracy)
    accuracy_np = np.asarray(accuracy_list).mean()
    
    print accuracy_np

if __name__ == "__main__":
  main()



