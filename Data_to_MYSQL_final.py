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
    # strip date parameters from the basketball-reference string
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
    # get an array of differences by year-date given the df.Date of Player Data Frame
    # first game has a 3 day break
    difference = [3];
    for index, elem in enumerate(seri):
        if (index + 1) < len(seri):
            t1 = get_realtime(elem)
            t2 = get_realtime(seri[index+1])
            if t2.tm_year == t1.tm_year:
                diff = abs(t2.tm_yday - t1.tm_yday)
                if diff > 4:
                    diff=4
            else:
                diff = t2.tm_yday + (365 - t1.tm_yday)  
                if diff > 4:
                    diff=4
            difference.append(diff)
        
    return difference

def get_gameday_month(seri):
    # get the month for each game date for merging purposes with team data
    month = [];
    day = [];
    for elem in seri:
        
        t1 = get_realtime(elem)
        m = t1.tm_mon
        d = t1.tm_mday
        month.append(m)
        day.append(d)
        
    return day,month

def get_home_assign_series(seri):
    # return a binarized array of home versus away games
    home_away =[]
    for elem in (seri):
        if elem == '@' :
            h = 2
        else:
            h=1
        home_away.append(h)
    return home_away

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
                ln = last_names[i][:5]
                if first == fn[1:3] and last == ln[:5]:
                    name = fn[1:] + ' ' + last_names[i]
                    f = filenames[filenames_dict[key]]
                    final_dict[name]=f
    return final_dict

def clean_data_and_write_sql(path):
    
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
    
    # for all players
    for key in final_dict:
        print key
        # read in the .csv player file
        df_player=pd.read_csv('~/Insight/Players/2015/'+final_dict[key])
        # select for games actually played
        df_player1=df_player[df_player.GS =='0']
        df_player2=df_player[df_player.GS =='1']
        df_player=pd.concat([df_player1,df_player2])
        df_player.Rk=df_player.Rk.convert_objects(convert_numeric=True)
        df_player = df_player.sort(columns=['Rk'])
        # rename empty columns
        df_player = df_player.rename(columns={'Unnamed: 5': 'Home','Unnamed: 7': 'Win'})
        df_player = df_player.reset_index(drop=True)
        # add player name columns
        df_player['Player_Name']= key
        home = get_home_assign_series(df_player.Home)
        # binary value for home vs away
        df_player['Home_Away']=home
        differences = get_gameday_diff_series(df_player.Date)
        # differences in game-day
        df_player['DateDiff']=differences
        day, month = get_gameday_month(df_player.Date)
        # get the month to merge team data into
        df_player['Month']=month
        df_player['Day']=day
        # drop player columns that are not for training or for target (add/remove 30 for 2015/2014)
        df_player_clean = df_player.drop(df_player.columns[[5,7,10,11,14,16,17,19,20,28,29,30]], axis=1)
        # read in team.csv
        df_teams_new = pd.read_csv('~/Insight/TeamStats2015.csv')
        # merge with player DataFrame
        complete_pandas=pd.merge(df_player_clean, df_teams_new, on=['Opp','Month'], how='inner')
        # SQL doesn't like % for the column names
        complete_pandas = complete_pandas.rename(columns={'FG%': 'FGpercent','3P%':'3Ppercent','FT%':'FTpercent','O3P%':'O3Ppercent','OFG%':'OFGpercent','OFT%':'OFTpercent','OBLK%':'OBLKpercent'})
        engine = create_engine("mysql+pymysql://root@localhost/Player_Team_Data") 
        complete_pandas.to_sql('NBA_player_data', engine,if_exists='append')
  
    
def main():
    if len(sys.argv) != 2:
        print 'Gimme a .csv'
        sys.exit(1)

    clean_data_and_write_sql(sys.argv[1])
    print 'done'


if __name__ == '__main__':
    main()