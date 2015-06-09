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

def get_realtime(date_string):
    new_string = ''
    for letter in date_string:
        if letter == '-':
            l =' '
        else:
            l = letter
        new_string = new_string + l
    t =time.strptime(new_string, "%Y %m %d") 
    return t
    
def get_gameday_diff_series(seri):
    difference = [3];
    for index, elem in enumerate(seri):
        if (index + 1) < len(seri):
            t1 = get_realtime(elem)
            t2 = get_realtime(seri[index+1])
        
        
            if t2.tm_year == t1.tm_year:
                diff = abs(t2.tm_yday - t1.tm_yday)
            else:
        
                diff = t2.tm_yday + (365 - t1.tm_yday)  
                if diff > 4:
                    diff=4
           
            difference.append(diff)
        
    return difference

def get_gameday_month(seri):
    month = [];
    for elem in seri:
        
        t1 = get_realtime(elem)
        m = t1.tm_mon
        month.append(m)
        
    return month

def get_home_assign_series(seri):
    home_away =[]
    for elem in (seri):
        if elem == '@' :
            h = 1
        else:
            h=0
        home_away.append(h)
    return home_away

def get_data(filenames_dict,first_names,last_names, filenames):

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
                ln = last_names[i][:5]
                if first == fn[1:3] and last == ln[:5]:
                    name = fn[1:] + ' ' + last_names[i]
                    f = filenames[filenames_dict[key]]
                    final_dict[name]=f
    return final_dict

def clean_data_and_write_sql(path):
  
    df = pd.read_csv('~/Insight/AllPlayerNames.csv')
    df = df[df.Name != 'Player']
    df_list=list(df.values)
    new_list = []
    
    for name in df_list:
        name = str(name)
        new_list.append(name)
        
    newlist=list(set(new_list))
    newlist.sort()
    first_names =[]
    last_names =[]
    for name in newlist:
        match = re.search('([\w\.\-\']+) ([\w\.\-]+)', name)
        first_names.append(match.group(1).lower())
        last_names.append(match.group(2).lower())
    filenames = os.listdir(path)
    del filenames[0]
    filenames_dict={}
    for index, name in enumerate(filenames):
        match = re.search('players_(\w)_(\w+)_gamelog', name)
        filenames_dict[match.group(2)]=index
    
    final_dict = get_data(filenames_dict,first_names,last_names,filenames)
    
    for key in final_dict:
  
        df_player=pd.read_csv('~/Insight/Players/'+final_dict[key])
        df_player1=df_player[df_player.GS =='0']
        df_player2=df_player[df_player.GS =='1']
        df_player=pd.concat([df_player1,df_player2])
        df_player.Rk=df_player.Rk.convert_objects(convert_numeric=True)
        df_player = df_player.sort(columns=['Rk'])
        df_player = df_player.rename(columns={'Unnamed: 5': 'Home','Unnamed: 7': 'Win'})
        df_player=df_player.reset_index(drop=True)
        df_player['Player_Name']= key
        home = get_home_assign_series(df_player.Home)
        df_player['Home_Away']=home
        differences = get_gameday_diff_series(df_player.Date)
        df_player['DateDiff']=differences
        month = get_gameday_month(df_player.Date)
        df_player['Month']=month
        df_player_clean = df_player.drop(df_player.columns[[3,5,7,10,11,14,16,17,19,20,26,28,29,30]], axis=1)
        df_teams_new = pd.read_csv('~/Insight/TeamStats.csv')
        complete_csv=pd.merge(df_player_clean, df_teams_new, on=['Opp','Month'], how='inner')
        complete_csv = complete_csv.rename(columns={'FG%': 'FGpercent','3P%':'3Ppercent','FT%':'FTpercent','O3P%':'O3Ppercent','OFG%':'OFGpercent','OFT%':'OFTpercent','OBLK%':'OBLKpercent'})
        engine = create_engine("mysql+pymysql://root@localhost/nba_gamestats") 
        complete_csv.to_sql('Historical_Test3', engine,if_exists='append')
  
    
def main():
    if len(sys.argv) != 2:
        print 'Gimme a .csv'
        sys.exit(1)

    clean_data_and_write_sql(sys.argv[1])
    print 'done'


if __name__ == '__main__':
    main()