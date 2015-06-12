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
    print player, opponent, home, game_diff
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
    df_raw_scaled = df_raw_scaled.applymap(lambda x: np.log(x))
    
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
    
    pre_dict={'PTS':pPTS,'3P':pTP,'REB':pREB,'AST':pAST,'STL':pSTL,'BLK':pBLK,'TOV':pTOV}
    prediction=pd.DataFrame(pre_dict)
    
    prediction=prediction.applymap(lambda x:round(x))
    prediction[prediction < 0] = 0
    predictions=prediction.applymap(lambda x:float(x))
    print predictions
    #rf = RandomForestClassifier(n_estimators=1000)
    average_stats=df_target.mean()
    #rf.fit(df_train_scaled, df_target)
    #predictions = rf.predict(df_evaluate)
    return predictions,average_stats.round()
    
def get_fanduel(predict,avg):
    fanduel_pre = predict[0]+predict[2]*1.2+predict[3]*1.5+predict[4]*2+predict[5]*2-predict[6]
    fanduel_avg = avg[0]+avg[2]*1.2+avg[3]*1.5+avg[4]*2+avg[5]*2-avg[6]
    fanduel_outlook = fanduel_pre - fanduel_avg
    return fanduel_pre, fanduel_avg, fanduel_outlook

def get_boxscore_diff(predict, avg):
    diff = predict[0] - avg
    return diff
    
#def get_prediction(opponent,player, game_diff, home):
    #prediction,avg_stats=calculate_predictions(opponent,player,game_diff,home)  
    #predict_fanduel, average_fanduel, outlook_fanduel=get_fanduel(prediction[0],avg_stats)
    #outlook = get_boxscore_diff(prediction,avg_stats)
    #return prediction,avg_stats,outlook,predict_fanduel, average_fanduel, outlook_fanduel 
  
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
    prediction,avg_stats=calculate_predictions(opponent,player,game_diff,home)
    # print 'Predict: PTS: ', prediction[0][0], ' 3P: ', prediction[0][1], ' TRB: ', prediction[0][2], ' AST: ', prediction[0][3], ' STL: ', prediction[0][4], ' BLK: ', prediction[0][5], ' TOV: ', prediction[0][6]
#     print 'Average: PTS: ', avg_stats[0], ' 3P: ', avg_stats[1], ' TRB: ', avg_stats[2], ' AST: ', avg_stats[3], ' STL: ', avg_stats[4], ' BLK: ', avg_stats[5], ' TOV: ', avg_stats[6]
#     diff_predict = get_boxscore_diff(prediction,avg_stats)
#     print 'Outlook: PTS:  ', diff_predict[0], ' 3P: ', diff_predict[1], ' TRB: ', diff_predict[2], ' AST: ', diff_predict[3], ' STL: ', diff_predict[4], ' BLK: ', diff_predict[5], ' TOV: ', diff_predict[6]
#     
#     predict_fanduel, average_fanduel, outlook_fanduel=get_fanduel(prediction[0],avg_stats)
#    
#     print 'Fanduel (Predict): ', predict_fanduel
#     print 'Fanduel (Average): ', average_fanduel
#     print 'Fanduel (Outlook): ', outlook_fanduel
  # +++your code here+++
  # Call your functions
  
if __name__ == "__main__":
  main()


