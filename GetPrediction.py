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

    cmd_target = 'SELECT FGpercent,PTS,3P,TRB,AST,STL,BLK,TOV FROM NBA_player_data WHERE Player_Name IN (\'' + player + '\');'
    cmd_train = 'SELECT Rk,Home_Away,DateDiff,TeamID,Win,OPPG,OTPR,O3Ppercent,ORPG,OBPG,OSPG,DEF,O3PM,OFGpercent,OTPG,OAPG,OFTpercent,TPG,SPG,TRBR,OBLKpercent FROM NBA_player_data WHERE Player_Name IN (\'' + player + '\');'
    cmd_truth = 'SELECT FGpercent,PTS,3P,AST,TRB,STL,BLK,TOV FROM NBA_player_data WHERE Player_Name IN (\'' + player + '\') AND Rk >50 AND Rk <60 ;'
    cmd_test = 'SELECT Rk,Home_Away,DateDiff,TeamID,Win,OPPG,OTPR,O3Ppercent,ORPG,OBPG,OSPG,DEF,O3PM,OFGpercent,OTPG,OAPG,OFTpercent,TPG,SPG,TRBR,OBLKpercent FROM NBA_player_data WHERE Player_Name IN (\'' + player + '\') AND Rk >50 AND Rk <60;'
    cmd_operate = 'SELECT TeamID,Win,OPPG,OTPR,O3Ppercent,ORPG,OBPG,OSPG,DEF,O3PM,OFGpercent,OTPG,OAPG,OFTpercent,TPG,SPG,TRBR,OBLKpercent FROM NBA_player_data WHERE Opponent_City IN (\'' + opponent + '\') AND Month = 4 LIMIT 1;'

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
    rf.fit(df_train_scaled, df_target)
    predictions = rf.predict(df_evaluate)
    return predictions
    
def get_fanduel(boxscore):
    fanduel = boxscore[1]+boxscore[3]*1.2+boxscore[4]*1.5+boxscore[5]*2+boxscore[6]*2-boxscore[7]
    return fanduel

def main():
  # This basic command line argument parsing code is provided.
  # Add code to call your functions below.

  # Make a list of command line arguments, omitting the [0] element
  # which is the script itself.
    args = sys.argv[1:]
    if not args:
        print "usage: [--firstname player] [--lastname player] [--home_away home] [--opponent opp] [--gamediff gamediff]";
        sys.exit(1)

  # todir and tozip are either set from command line
  # or left as the empty string.
  # The args array is left just containing the dirs.
    player = args[0] + ' ' + args[1]
    opponent = args[2]
    game_diff = args[3]
    home = args [4]
    prediction=calculate_predictions(opponent,player,game_diff,home)
    print 'FG%: ', prediction[0][0]
    print 'PTS: ', prediction[0][1]
    print '3P: ', prediction[0][2]
    print 'TRB: ', prediction[0][3]
    print 'AST: ', prediction[0][4]
    print 'STL: ', prediction[0][5]  
    print 'BLK: ', prediction[0][6]
    print 'TOV: ', prediction[0][7]
    fanduel=get_fanduel(prediction[0])
    print fanduel

  # +++your code here+++
  # Call your functions
  
if __name__ == "__main__":
  main()

