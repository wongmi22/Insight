
from flask import render_template, request
from app import app
import pymysql as mdb
from a_Model import ModelIt
from GetPrediction_ByDate_Flask import get_results
import pandas as pd
import re
db = mdb.connect(user="root", host="localhost", db="world_innodb", charset='utf8')

@app.route('/')
@app.route('/index')
def index():
    return render_template("index.html",
       title = 'Home', user = { 'nickname': 'Miguel' },
       )

@app.route('/db')
def cities_page():
    with db:
        cur = db.cursor()
        cur.execute("SELECT Name FROM City LIMIT 15;")
        query_results = cur.fetchall()
    cities = ""
    for result in query_results:
        cities += result[0]
        cities += "<br>"
    return cities

@app.route("/db_fancy")
def cities_page_fancy():
    with db:
        cur = db.cursor()
        cur.execute("SELECT Name, CountryCode, Population FROM City ORDER BY Population LIMIT 15;")

        query_results = cur.fetchall()
    cities = []
    for result in query_results:
        cities.append(dict(name=result[0], country=result[1], population=result[2]))
    return render_template('cities.html', cities=cities)

@app.route('/input')
def cities_input():
  return render_template("input_project.html")

  
@app.route('/output')
def cities_output():
  #pull 'ID' from input field and store it
  month = request.args.get('month')
  date = request.args.get('date')
  metric = request.args.get('metric')
  date = request.args.get('date')
  match = re.search('(\w\w)/(\w\w)/(\w\w\w\w)', date)
  day = match.group(1)
  if day[0] == 0:
    day = day[1]
  month = match.group(2)
  if month[0] == 0:
    month = month[1]
  year = match.group(3)
  print day, month, year
  #home = request.args.get('Home')
  
  
  # with db:
#     cur = db.cursor()
#     #just select the city from the world_innodb that the user inputs
#     cur.execute("SELECT Name, CountryCode, Population FROM City WHERE Name='%s';" % city)
#     query_results = cur.fetchall()
# 
#   cities = []
#   for result in query_results:
#     cities.append(dict(name=result[0], country=result[1], population=result[2]))
#   #call a function from a_Model package. note we are only pulling one result in the query
#   pop_input = cities[0]['population']
  # data = get_results(team, player,days, home)
  # pre_rf = list(prediction_rf[0])
  # pre_rf.insert(0, 'Random Forest')
  # pre_lm = list(prediction_lm)
  # pre_lm.insert(0, 'Linear Model')
  # avg = list(avg_stats)
  # avg.insert(0, 'Average')
  # out_rf = list(outlook_rf)
  # out_rf.insert(0, 'Performance (Rf)')
  # out_lm = list(outlook_lm)
  # out_lm.insert(0, 'Performance (Lm)')
  
  # Fanduel = [['Prediction',predict_fanduel],['Average', average_fanduel],['Performance',outlook_fanduel]]
  # if outlook_fanduel > 0:
  #   Diagnosis= 'OUTPERFORM!'
  # elif outlook_fanduel <0:
  #   Diagnosis= 'UNDERPERFORM!'
  # else:
  #   Diagnosis= 'Average Performance' 

  df_big_sorted = get_results(month,day,metric)

  
  return render_template("output_project.html", df_big_sorted = df_big_sorted, d=date)
