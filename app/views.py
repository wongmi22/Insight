
from flask import render_template, request
from app import app
import pymysql as mdb
from a_Model import ModelIt
from GetPrediction_ByDate_FlaskLm import get_results
from GetPrediction_ByTeam_FlaskLm import get_results_team
import pandas as pd
import re

def get_date(date):
  match = re.search('(\w\w)/(\w\w)/(\w\w\w\w)', date)

  day_i = match.group(1)
  if day_i[0] == 0:
    day_i = day_i[1]
  month_i = match.group(2)
  if month_i[0] == 0:
    month_i = month_i[1]
  year = match.group(3)

  return day_i, month_i, year


@app.route('/')
def origin():
  return render_template("home.html")


@app.route('/home')
def cities_home():
  return render_template("home.html")

@app.route('/input')
def cities_input():
  mode =request.args.get('mode')
  number =int(request.args.get('number'))
  player_number = []

  # Produce variable names for the player form ID's
  for i in range(number):
    playa = 'Player_'+ str(i)
    player_number.append(playa)

  # If user chooses to predict all players
  if mode == 'all':
    month = request.args.get('month')
    date = request.args.get('date')
    metric = request.args.get('metric')
  # check if user properly entered date via datepicker calendar.  Default date is January 5th, 2015
    if len(date) > 0:

      day_i,month_i,year = get_date(date)

    else: 
      
      day_i = 5
      month_i = 1
      date = '01/05/2015'
    choice = metric == 'relative'
    df_big_sorted = get_results(month_i,day_i,metric)

    return render_template("output_project.html", df_big_sorted = df_big_sorted, d=date, m=choice)
  # If user chooses to predict a team of players
  else:
    return render_template("input_projects2.html",playas=player_number,n=number)

  
@app.route('/output_team')
def teams_output():
 
  #Predictions by date: For a specified team
  month = request.args.get('month')
  date = request.args.get('date')
  metric = request.args.get('metric')
  num = int(request.args.get('playernum'))
  player_list = []

  # Aggregate entered player names as a list
  for i in range(num):
    playa = 'Player_'+ str(i)
    player_list.append(str(request.args.get(playa)))

  # check if user properly entered date via datepicker calendar.  Default date is January 5th, 2015
  if len(date) > 0:
    
    day_i,month_i,year = get_date(date)

  else: 
    day_i = 5
    month_i = 1
    date = '01/05/2015'
  choice = metric == 'relative'

  # return data frame from GetPrediction_ByTeam_FlaskLm.py
  df_big_sorted,df_absent,df_big_sorted_acc = get_results_team(month_i,day_i,metric,player_list)
  
  #boolean to determine if players either played or didn't play on that date
  show = len(df_big_sorted)>0
  show2 = len(df_absent>0)
  return render_template("output_project_team.html", df_big_sorted=df_big_sorted, ab=df_absent,d=date, m=choice, show=show, show2=show2)
  
@app.route('/output')
def cities_output():
  #Predictions by date: All Players
  month = request.args.get('month')
  date = request.args.get('date')
  metric = request.args.get('metric')
  

  # check if user properly entered date via datepicker calendar.  Default date is January 5th, 2015
 
  if len(date) > 0:
    
    day_i,month_i,year = get_date(date)

  else: 
    day_i = 5
    month_i = 1
    date = '01/05/2015'
  choice = metric == 'relative'
  # return data frame from GetPrediction_ByDate_FlaskLm.py
  df_big_sorted = get_results(month_i,day_i,metric)

  return render_template("output_project.html", df_big_sorted = df_big_sorted, d=date, m=choice)
