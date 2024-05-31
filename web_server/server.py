



from flask import Flask, render_template
import pickle
from datetime import datetime, timedelta




with open('../predictions/pl_predictions.csv', 'rb') as myFile:
    pl_pred = pickle.load(myFile)
    
with open('../prem_clean_fixtures_and_dataframes/2019_2020_2021_2022_2023_additional_stats_dict.txt', 'rb') as myFile:
    additional_stats_dict = pickle.load(myFile)    



temp_current_date = datetime(2024,3,1)
current_date = temp_current_date.strftime('%Y-%m-%d')
for j in range(len(pl_pred)):
    game_date = pl_pred['Game Date'].loc[j]
    if game_date < current_date:
        pl_pred = pl_pred.drop([j], axis=0)
pl_pred = pl_pred.reset_index(drop=True)        



max_display_games = 40
iterator_len = len(pl_pred) - 1
if iterator_len > max_display_games:
    iterator_len = max_display_games
iterator = range(iterator_len)


max_additional_display_games = 5
dict_keys = list(additional_stats_dict.keys())
min_length = 100
for i in dict_keys:
    df_len = len(additional_stats_dict[i])
    if df_len < min_length:
        min_length = df_len
if max_additional_display_games > min_length:
    max_additional_display_games = min_length
iterator2 = range(max_additional_display_games)



app = Flask(__name__, static_url_path='/static')

@app.route('/')
def pass_game_1():
    return render_template('index.html',
                           pl_pred=pl_pred, 
                           iterator=iterator,
                           iterator2=iterator2,
                           additional_stats_dict=additional_stats_dict)

if __name__ == '__main__':
    
    app.run(host = '0.0.0.0', port = 5000)
    