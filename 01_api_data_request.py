print('\n\n ---------------- START ---------------- \n')



import time
start=time.time()

import requests
import pandas as pd
import math
from os import listdir








YEAR = 2023
YEAR_str = str(YEAR)

request_league_ids = True
request_fixtures = True
request_missing_game_stats = True




api_key ='66e6132dc27bfd2d090b8e9e8b3951b3'


def get_api_data(base_url, end_url):
    url = base_url + end_url
    headers = {'X-RapidAPI-Key': api_key}
    res = requests.get(url, headers=headers)
    if res.status_code != 200:
        raise RuntimeError(f'error {res.status_code}')
    res_t = res.text
    return res_t


def slice_api(api_str_output, start_char, end_char):
  e = len(api_str_output) - end_char
  s = start_char
  output = api_str_output[s:e]
  return output


def save_api_output(save_name, jason_data, json_data_path=''):
    writeFile = open(json_data_path + save_name + '.json', 'w')
    writeFile.write(jason_data)
    writeFile.close()
    
    
def read_json_as_pd_df(json_data, json_data_path='', orient_def='records'):
    output = pd.read_json(json_data_path + json_data, orient=orient_def)
    return output
  
    


base_url = 'https://v2.api-football.com/'


def req_prem_fixtures_id(season_code, year=YEAR_str):
    
    premier_league_fixtures_raw = get_api_data(base_url, f'/fixtures/league/{season_code}/')

    
    premier_league_fixtures_sliced = slice_api(premier_league_fixtures_raw, 33, 2)

    
    save_api_output(f'{year}_premier_league_fixtures', premier_league_fixtures_sliced, json_data_path = 'prem_clean_fixtures_and_dataframes/')

    
    premier_league_fixtures_df = read_json_as_pd_df(f'{year}_premier_league_fixtures.json', json_data_path='prem_clean_fixtures_and_dataframes/')
    return premier_league_fixtures_df



if request_league_ids:
    leagues = premier_league_fixtures_raw = get_api_data(base_url, 'leagues/country/England/2023')

if YEAR == 2019:
    season_id = 524
elif YEAR == 2020:
    season_id = 2790
elif YEAR == 2021:
    season_id = 3456
elif YEAR == 2022:
    season_id = 4335
elif YEAR == 2023:
    season_id = 5267
else:
    print('please lookup season id and specify this as season_id variable')


if request_fixtures:
    fixtures = req_prem_fixtures_id(season_id, YEAR_str)
    

def load_prem_fixtures_id(year=YEAR_str):
    premier_league_fixtures_df = read_json_as_pd_df(f'{year}_premier_league_fixtures.json', json_data_path='prem_clean_fixtures_and_dataframes/')
    return premier_league_fixtures_df


fixtures = load_prem_fixtures_id()




fixtures = pd.read_json(f'prem_clean_fixtures_and_dataframes/{YEAR_str}_premier_league_fixtures.json', orient='records')



for i in fixtures.index:
    x1 = str(fixtures['homeTeam'].iloc[i])[12:14]
    x = int(x1)
    fixtures.at[i, 'HomeTeamID'] = x

for i in fixtures.index:
    x1 = str(fixtures['awayTeam'].iloc[i])[12:14]
    x = int(x1)
    fixtures.at[i, 'AwayTeamID'] = x

for i in fixtures.index:
    x = str(fixtures['event_date'].iloc[i])[:10] 
    fixtures.at[i, 'Game Date'] = x

for i in fixtures.index:
    x = str(fixtures['homeTeam'][i]['team_name']) 
    fixtures.at[i, 'Home Team'] = x
    
for i in fixtures.index:
    x = str(fixtures['awayTeam'][i]['team_name']) 
    fixtures.at[i, 'Away Team'] = x
    
for i in fixtures.index:
    x = str(fixtures['homeTeam'][i]['logo']) 
    fixtures.at[i, 'Home Team Logo'] = x
    
for i in fixtures.index:
    x = str(fixtures['awayTeam'][i]['logo']) 
    fixtures.at[i, 'Away Team Logo'] = x
 
fixtures_clean = pd.DataFrame({'Fixture ID': fixtures['fixture_id'], 'Game Date': fixtures['Game Date'], 'Home Team ID': fixtures['HomeTeamID'], 'Away Team ID': fixtures['AwayTeamID'], 'Home Team Goals': fixtures['goalsHomeTeam'], 'Away Team Goals': fixtures['goalsAwayTeam'], 'Venue': fixtures['venue'], 'Home Team': fixtures['Home Team'], 'Away Team': fixtures['Away Team'], 'Home Team Logo': fixtures['Home Team Logo'], 'Away Team Logo': fixtures['Away Team Logo']})

fixtures_clean.to_csv(f'prem_clean_fixtures_and_dataframes/{YEAR_str}_premier_league_fixtures_df.csv', index=False)






fixtures_clean_2019_2020_2021_2022 = pd.read_csv('prem_clean_fixtures_and_dataframes/2019_2020_2021_2022_premier_league_fixtures_df.csv')

fixtures_clean_2023 = pd.read_csv('prem_clean_fixtures_and_dataframes/2023_premier_league_fixtures_df.csv')

fixtures_clean_combined = pd.concat([fixtures_clean_2019_2020_2021_2022, fixtures_clean_2023])
fixtures_clean_combined = fixtures_clean_combined.reset_index(drop=True)

fixtures_clean_combined.to_csv('prem_clean_fixtures_and_dataframes/2019_2020_2021_2022_2023_premier_league_fixtures_df.csv', index=False)




fixtures_clean = pd.read_csv(f'prem_clean_fixtures_and_dataframes/{YEAR_str}_premier_league_fixtures_df.csv')
    

def req_prem_stats(start_index, end_index):
    for i in fixtures_clean.index[start_index:end_index]:
        if math.isnan(fixtures_clean['Home Team Goals'].iloc[i]) == False:
            fix_id = str(fixtures_clean['Fixture ID'].iloc[i])
            fixture_raw = get_api_data(base_url, '/statistics/fixture/' + fix_id + '/')
            fixture_sliced = slice_api(fixture_raw, 34, 2)
            save_api_output('prem_game_stats_json_files/' + fix_id, fixture_sliced)
    


 






existing_data_raw = listdir('prem_game_stats_json_files/')


existing_data = []
for i in existing_data_raw:
    if i.endswith('.json'):
        try:
            existing_data.append(int(i[:-5]))
        except ValueError:
            print(f"Skipping invalid entry: {i}")
    else:
        print(f"Skipping non-.json entry: {i}")


missing_data = []
for i in fixtures_clean.index:
    fix_id = fixtures_clean['Fixture ID'].iloc[i]
    if fix_id not in existing_data:
        if math.isnan(fixtures_clean['Home Team Goals'].iloc[i]) == False:
            missing_data.append(fix_id)


def req_prem_stats_list(missing_data):
    if len(missing_data) > 100:
        print('This request exceeds 100 request limit and has not been completed')
    else:
        if len(missing_data) > 0:
            print('Data collected for the following fixtures:')
        for i, dat in enumerate(missing_data):
            pause_points = [(i*10)-1 for i in range(10)]
            if i in pause_points:
                print('sleeping for 1 minute - API only allows 10 requests per minute')
                time.sleep(60)
            print(dat)
            fix_id = str(dat)
            fixture_raw = get_api_data(base_url, '/statistics/fixture/' + fix_id + '/')
            fixture_sliced = slice_api(fixture_raw, 34, 2)
            save_api_output('prem_game_stats_json_files/' + fix_id, fixture_sliced)


if request_missing_game_stats:
    req_prem_stats_list(missing_data)




print('\n', 'Script runtime:', round(((time.time()-start)/60), 2), 'minutes')
print(' ----------------- END ----------------- \n')
