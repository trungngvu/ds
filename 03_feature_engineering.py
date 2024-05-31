print('\n\n ---------------- START ---------------- \n')



import time
start=time.time()

import pickle
from ml_functions.feature_engineering_functions import average_stats_df
from ml_functions.feature_engineering_functions import creating_ml_df






stats_dict_saved_name = '2015_2016_2017_2018_2019_2020_2021_2022_2023_prem_all_stats_dict.txt'

df_5_output_name = '2015_2016_2017_2018_2019_2020_2021_2022_2023_prem_df_for_ml_5_v2.txt'
df_10_output_name = '2015_2016_2017_2018_2019_2020_2021_2022_2023_prem_df_for_ml_10_v2.txt'




with open(f'prem_clean_fixtures_and_dataframes/{stats_dict_saved_name}', 'rb') as myFile:
    game_stats = pickle.load(myFile)


team_list = []
for key in game_stats.keys():
    team_list.append(key)
team_list.sort()


team_fixture_id_dict = {}
for team in team_list:
    fix_id_list = []
    for key in game_stats[team].keys():
        fix_id_list.append(key)
    fix_id_list.sort()
    sub_dict = {team:fix_id_list}
    team_fixture_id_dict.update(sub_dict)
        

for team in team_fixture_id_dict:
    team_fixture_id_dict[team].sort()




df_ml_5 = average_stats_df(5, team_list, team_fixture_id_dict, game_stats)


df_ml_10 = average_stats_df(10, team_list, team_fixture_id_dict, game_stats)
        

df_for_ml_5_v2 = creating_ml_df(df_ml_5)
with open(f'prem_clean_fixtures_and_dataframes/{df_5_output_name}', 'wb') as myFile:
    pickle.dump(df_for_ml_5_v2, myFile)


df_for_ml_10_v2 = creating_ml_df(df_ml_10)
with open(f'prem_clean_fixtures_and_dataframes/{df_10_output_name}', 'wb') as myFile:
    pickle.dump(df_for_ml_10_v2, myFile)



df_for_ml_10_v2.to_csv('prem_clean_fixtures_and_dataframes/df_for_powerbi.csv', index='False')





print('\n', 'Script runtime:', round(((time.time()-start)/60), 2), 'minutes')
print(' ----------------- END ----------------- \n')
