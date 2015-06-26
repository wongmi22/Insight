from __future__ import division
from instagram import client, subscriptions
from instagram.client import InstagramAPI
import urlparse
import urllib
from pandas import Series,DataFrame
import pandas as pd
import re
import numpy as np
from datetime import datetime
from pytz import timezone
import pytz
from sklearn.ensemble import RandomForestClassifier
import time
import sys
from sqlalchemy import create_engine
from sqlalchemy.types import String 
import os
import MySQLdb
import MySQLdb.cursors
import pandas as pd
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import sys
from scipy import stats
import scipy as sp
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

def get_data(filenames_dict,first_names,last_names, filenames):
    # return a dict with the player names that are in the downloaded player files
    # the key = player name, the value = csv filename

    final_dict ={}
    for key in filenames_dict:
        if len(key) == 7:
            last = key[:3]
            first = key [3:-2]
            for i,fn in enumerate(first_names):
                ln = last_names[i][:3]
                if first == fn[1:3] and last == ln[:3]:
                    name = fn[1:] + ' ' + last_names[i]
                    f = filenames[filenames_dict[key]]
                    final_dict[name]=f
        elif len(key) == 8:
            last = key[:4]
            first = key [4:-2]
            for i,fn in enumerate(first_names):
                ln = last_names[i][:4]
                if first == fn[1:3] and last == ln[:4]:
                    name = fn[1:] + ' ' + last_names[i]
                    f = filenames[filenames_dict[key]]
                    final_dict[name]=f
        elif len(key) ==9:
            last = key[:5]
            first = key [5:-2]
            for i,fn in enumerate(first_names):
                #fn = fn.translate(None,'.')
                ln = last_names[i][:5]
                if first == fn[1:3] and last == ln[:5]:
                    name = fn[1:] + ' ' + last_names[i]
                    f = filenames[filenames_dict[key]]
                    final_dict[name]=f
    return final_dict
    
def get_fanduel(predict,avg,truth):

    fanduel_pre = predict[0]+predict[2]*1.2+predict[3]*1.5+predict[4]*2+predict[5]*2-predict[6]
    fanduel_avg = avg[0]+avg[2]*1.2+avg[3]*1.5+avg[4]*2+avg[5]*2-avg[6]
    fanduel_tru = truth[0]+ truth[2]*1.2+truth[3]*1.5+truth[4]*2+truth[5]*2-truth[6]
    return fanduel_pre, fanduel_avg, fanduel_tru
    
def get_validation(path):    

# read in AllPlayerNames .csv from basketball-reference
    df_all_players = pd.read_csv('~/Insight/AllPlayerNames.csv')
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
    
    first_names =[]
    last_names =[]
    #separate first and last names of each player
    for name in unique_name_list:
        match = re.search('([\w\.\-\']+) ([\w\.\-]+)', name)
        first_names.append(match.group(1).lower())
        last_names.append(match.group(2).lower())
    # capture all csv files in the Players directory to list
    filenames = os.listdir(path)
    # first file is a .DS_store
    del filenames[0]
    # grab the unique identifier and store its location
    filenames_dict={}
    for index, name in enumerate(filenames):
        match = re.search('players_(\w)_(\w+)_gamelog', name)
        filenames_dict[match.group(2)]=index
    
    # get final_dict of key=names, values = csv filename
    final_dict = get_data(filenames_dict,first_names,last_names,filenames)
    acc =[]
    perc_diff_pre =[]
    perc_diff_avg =[]
    # for all players
    playas=['Russell Westbrook','James Harden','LeBron James','Anthony Davis','DeMarcus Cousins','Stephen Curry', 'LaMarcus Aldridge', 'Blake Griffin', 'Kyrie Irving', 'Klay Thompson', 'Rudy Gay', 'Damian Lillard', 'Nikola Vucevic', 'Gordon Hayward', 'Chris Paul','Monta Ellis', 'Pau Gasol', 'Victor Oladipo', 'Kyle Lowry', 'John Wall', 'Marc Gasol', 'Dirk Nowitzki', 'Brook Lopez', 'Eric Bledsoe', 'Andrew Wiggins', 'Paul Millsap', 'Tyreke Evans', 'Kevin Love', 'Goran Dragic', 'Zach Randolph', 'Derrick Favors']
    for p in playas:
        player=p.lower()
        print player
        conn = MySQLdb.connect(
            user="root",
            passwd="",
            db="Player_Team_Data",
            cursorclass=MySQLdb.cursors.DictCursor)

        cmd_target = 'SELECT PTS,3P,TRB,AST,STL,BLK,TOV FROM NBA_player_data WHERE Player_Name IN (\'' + player + '\') AND RK <60;'
        cmd_train = 'SELECT Rk,Home_Away,DateDiff,TeamID,Win,OPPG,OTPR,O3Ppercent,ORPG,OBPG,OSPG,DEF,O3PM,OFGpercent,OTPG,OAPG,OFTpercent,TPG,SPG,TRBR,OBLKpercent FROM NBA_player_data WHERE Player_Name IN (\'' + player + '\') AND RK <60;'
        cmd_truth = 'SELECT PTS,3P,TRB,AST,STL,BLK,TOV FROM NBA_player_data WHERE Player_Name IN (\'' + player + '\') AND Rk >60 ;'
        #cmd_test = 'SELECT Rk,Home_Away,DateDiff,TeamID,Win,OPPG,OTPR,O3Ppercent,ORPG,OBPG,OSPG,DEF,O3PM,OFGpercent,OTPG,OAPG,OFTpercent,TPG,SPG,TRBR,OBLKpercent FROM NBA_player_data WHERE Player_Name IN (\'' + player + '\') AND Rk >60;'
        cmd_operate = 'SELECT Rk,Home_Away,DateDiff,TeamID,Win,OPPG,OTPR,O3Ppercent,ORPG,OBPG,OSPG,DEF,O3PM,OFGpercent,OTPG,OAPG,OFTpercent,TPG,SPG,TRBR,OBLKpercent FROM NBA_player_data WHERE Player_Name IN (\'' + player + '\') AND Rk >60;'
        #cmd_train = 'SELECT TeamID,Win,OPPG,OTPR,O3Ppercent,ORPG,OBPG,OSPG,DEF,O3PM,OFGpercent,OTPG,OAPG,TPG,SPG,TRBR,OBLKpercent FROM NBA_player_data WHERE Player_Name IN (\'' + player + '\') AND RK <60;'
        #cmd_operate = 'SELECT TeamID,Win,OPPG,OTPR,O3Ppercent,ORPG,OBPG,OSPG,DEF,O3PM,OFGpercent,OTPG,OAPG,TPG,SPG,TRBR,OBLKpercent FROM NBA_player_data WHERE Player_Name IN (\'' + player + '\') AND Rk >60;'
        df_target = pd.read_sql(cmd_target, con=conn)    
        df_train = pd.read_sql(cmd_train,con=conn)
        df_truth= pd.read_sql(cmd_truth, con=conn)    
        #df_test = pd.read_sql(cmd_test,con=conn)
        df_operate = pd.read_sql(cmd_operate,con=conn)
        
    
        df_target = df_target.applymap(lambda x:float(x))
        df_train = df_train.applymap(lambda x:float(x))
        df_truth = df_truth.applymap(lambda x:float(x))
        #df_test = df_test.applymap(lambda x:float(x))
        df_operate = df_operate.applymap(lambda x:float(x))
        conn.close()
        df_train_plus_inquire=pd.concat([df_train, df_operate])
        df_raw = df_train_plus_inquire.reindex()
        #df_operate.notnull()[0]==True
        cn=len(df_operate)
        t=2
        #print cn
        #if key != 'carmelo anthony' and key != 'wesley matthews' and key != 'chris bosh' and key!= 'kevin durant':
        if t==2 :
            df_raw_scaled = df_raw.copy()
            df_raw_pure = df_raw.copy()
            df_raw_transform = df_raw.copy()
            df_raw_scaled = df_raw_scaled.apply(lambda x:preprocessing.StandardScaler().fit(x).transform(x))
            df_raw_transform = df_raw_transform.apply(lambda x:preprocessing.StandardScaler().fit(x))
            df_evaluate = df_raw_scaled.tail(cn)
            df_train_scaled = df_raw_scaled.iloc[:-cn]
            rf = RandomForestClassifier(n_estimators=1000)
            #rf = DecisionTreeClassifier()
            #rf= KNeighborsClassifier() 
            rf.fit(df_train_scaled, df_target)
            average_stats=df_target.mean()
            rf.fit(df_train_scaled, df_target)
            predictions = rf.predict(df_evaluate)
            fanduel_pre, fanduel_avg, fanduel_tru = get_fanduel(predictions[0],average_stats,df_truth.iloc[0,:])
            fan_pre=[]
            fan_avg=[]
            fan_tru=[]
            for i in range(cn):
                fanduel_pre, fanduel_avg, fanduel_tru = get_fanduel(predictions[i],average_stats,df_truth.iloc[i,:])
                fan_pre.append(fanduel_pre)
                fan_avg.append(fanduel_avg)
                fan_tru.append(fanduel_tru)
            val = []
            for i in range(cn):
                if (fan_pre[i] > fan_avg[i]) and (fan_tru [i] > fan_avg[i]):
                    val.append(1)
                elif (fan_pre[i] < fan_avg[i]) and (fan_tru [i] < fan_avg[i]):
                    val.append(1)
                elif (fan_pre[i] < fan_avg[i]) and (fan_tru [i] > fan_avg[i]):
                    val.append(0)
                elif (fan_pre[i] > fan_avg[i]) and (fan_tru [i] < fan_avg[i]):    
                    val.append(0)
            accuracy=val.count(1)/len(val)
            per_diff_pre=[]
            per_diff_avg=[]
            for i in range(cn):
                p_pre = 2*abs(fan_pre[i]-fan_tru[i])/(fan_pre[i]+fan_tru[i])
                p_avg = 2*abs(fan_avg[i]-fan_tru[i])/(fan_avg[i]+fan_tru[i])   
                per_diff_pre.append(p_pre)
                per_diff_avg.append(p_avg)
            percent_diff_pre=np.asarray(per_diff_pre)
            pdp=percent_diff_pre.mean()
            percent_diff_avg=np.asarray(per_diff_avg)
            pda=percent_diff_avg.mean()
        acc.append(accuracy)
        perc_diff_pre.append(pdp)
        perc_diff_avg.append(pda)
   #  final_acc = np.asarray(acc).mean()
#     final_pdp = np.asarray(perc_diff_pre).mean()
#     final_pda = np.asarray(perc_diff_avg).mean()
    final_acc = np.asarray(acc).mean()
    final_pdp = np.asarray(perc_diff_pre).mean()
    final_pda = np.asarray(perc_diff_avg).mean()
    return final_acc, final_pdp, final_pda
        
def main():
  # This basic command line argument parsing code is provided.
  # Add code to call your functions below.

  # Make a list of command line arguments, omitting the [0] element
  # which is the script itself.
  args = sys.argv[1:]
  final_acc, final_pdp, final_pda = get_validation(args[0])  
  print 'Accuracy: ', final_acc
  print 'Prediction Percent Change: ', final_pdp
  print 'Average Percent Change: ', final_pda
  # Call your functions
  
if __name__ == "__main__":
  main()
        