print('\n\n ---------------- START ---------------- \n')



import time
start=time.time()

import pandas as pd
import math
import pickle






fixtures_saved_name = '2015_2016_2017_2018_2019_2020_2021_2022_2023_premier_league_fixtures_df.csv'
stats_dict_output_name = '2015_2016_2017_2018_2019_2020_2021_2022_2023_prem_all_stats_dict.txt'






fixtures_clean = pd.read_csv(f'prem_clean_fixtures_and_dataframes/{fixtures_saved_name}')


fixtures_clean_ID_index = pd.Index(fixtures_clean['Fixture ID'])


team_id_list = (fixtures_clean['Home Team ID'].unique()).tolist()


all_stats_dict = {}


for team in team_id_list:
    
    
    team_fixture_list = []
    for i in fixtures_clean.index[:]:
        if fixtures_clean['Home Team ID'].iloc[i] == team:
            if math.isnan(fixtures_clean['Home Team Goals'].iloc[i]) == False:
                team_fixture_list.append(fixtures_clean['Fixture ID'].iloc[i])
    all_stats_dict[team] = {}
    for j in team_fixture_list:
        
        df = pd.read_json('prem_game_stats_json_files/' + str(j) + '.json', orient='values')
        
        df['Ball Possession'] = df['Ball Possession'].str.replace('%', '').astype(int)
        df['Passes %'] = df['Passes %'].str.replace('%', '').astype(int)
        
        temp_index = fixtures_clean_ID_index.get_loc(j)
        home_goals = fixtures_clean['Home Team Goals'].iloc[temp_index]
        away_goals = fixtures_clean['Away Team Goals'].iloc[temp_index]
        df['Goals'] = [home_goals, away_goals]
        
        if home_goals > away_goals:
            df['Points'] = [2,0]
        elif home_goals == away_goals:
            df['Points'] = [1,1]
        elif home_goals < away_goals:
            df['Points'] = [0,2]
        else:
            df['Points'] = ['nan', 'nan']
        
        df['Team Identifier'] = [1,2]
        
        df['Team ID'] = [team, fixtures_clean['Away Team ID'].iloc[temp_index]]
        
        gd = fixtures_clean['Game Date'].iloc[temp_index]
        df['Game Date'] = [gd, gd]
        
        sub_dict_1 = {j:df}
        all_stats_dict[team].update(sub_dict_1)
        
    
    team_fixture_list = []    
    for i in fixtures_clean.index[:]:
        if fixtures_clean['Away Team ID'].iloc[i] == team:
            if math.isnan(fixtures_clean['Away Team Goals'].iloc[i]) == False:
                team_fixture_list.append(fixtures_clean['Fixture ID'].iloc[i])
    for j in team_fixture_list:
        
        print('./prem_game_stats_json_files/' + str(j) + '.json')
        df = pd.read_json('./prem_game_stats_json_files/' + str(j) + '.json', orient='values')
        
        df['Ball Possession'] = df['Ball Possession'].str.replace('%', '').astype(int)
        df['Passes %'] = df['Passes %'].str.replace('%', '').astype(int)
        
        temp_index = fixtures_clean_ID_index.get_loc(j)
        home_goals = fixtures_clean['Home Team Goals'].iloc[temp_index]
        away_goals = fixtures_clean['Away Team Goals'].iloc[temp_index]
        df['Goals'] = [home_goals, away_goals]
        
        if home_goals > away_goals:
            df['Points'] = [2,0]
        elif home_goals == away_goals:
            df['Points'] = [1,1]
        elif home_goals < away_goals:
            df['Points'] = [0,2]
        else:
            df['Points'] = ['nan', 'nan']
        
        df['Team Identifier'] = [2,1]       
        
        df['Team ID'] = [fixtures_clean['Home Team ID'].iloc[temp_index], team]
        
        gd = fixtures_clean['Game Date'].iloc[temp_index]
        df['Game Date'] = [gd, gd]
        
        sub_dict_1 = {j:df}
        all_stats_dict[team].update(sub_dict_1)
        



with open(f'prem_clean_fixtures_and_dataframes/{stats_dict_output_name}', 'wb') as myFile:
    pickle.dump(all_stats_dict, myFile)

with open(f'prem_clean_fixtures_and_dataframes/{stats_dict_output_name}', 'rb') as myFile:
    loaded_dict_test = pickle.load(myFile)




print('\n', 'Script runtime:', round(((time.time()-start)/60), 2), 'minutes')
print(' ----------------- END ----------------- \n')
