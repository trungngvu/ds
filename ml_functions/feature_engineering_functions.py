


import numpy as np
import pandas as pd






def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / float(N)



def average_stats_df(games_slide, team_list, team_fixture_id_dict, game_stats, making_predictions=False):
    if making_predictions:
        x = games_slide
        xx = 1
    else:
        x = -1
        xx = 0

  
    
    t_total_shots = []
    t_shots_inside_box = []     
    t_fouls = []
    t_corners = []
    t_posession = []
    t_pass_accuracy = []
    t_goals = []
    t_goals_target = []
    o_total_shots = []
    o_shots_inside_box = []     
    o_fouls = []
    o_corners = []
    o_posession = []
    o_pass_accuracy = []
    o_goals = []
    o_goals_target = []
    fix_id = []
    result_indicator = []
    o_team_ID = []
    
    for team_id in team_list[:]:
        team = game_stats[team_id] 
        
        
        if len(team_fixture_id_dict[team_id]) < games_slide:
            continue
        
        
        team_total_shots = []
        team_shots_inside_box = []     
        team_fouls = []
        team_corners = []
        team_posession = []
        team_pass_accuracy = []
        team_goals = []
        opponent_total_shots = []
        opponent_shots_inside_box = []     
        opponent_fouls = []
        opponent_corners = []
        opponent_posession = []
        opponent_pass_accuracy = []
        opponent_goals = []
        result_indicator_raw = []
        
        
        
        for game_id in team_fixture_id_dict[team_id]:
            game = team[game_id] 
            temp_index = pd.Index(game['Team Identifier'])
            team_ind = temp_index.get_loc(1)
            opponent_ind = temp_index.get_loc(2)
            
            
            team_total_shots.append(game['Total Shots'][team_ind])
            team_shots_inside_box.append(game['Shots insidebox'][team_ind])
            team_fouls.append(game['Fouls'][team_ind])
            team_corners.append(game['Corner Kicks'][team_ind])
            team_posession.append(game['Ball Possession'][team_ind])
            team_pass_accuracy.append(game['Passes %'][team_ind])
            team_goals.append(game['Goals'][team_ind])     
            opponent_total_shots.append(game['Total Shots'][opponent_ind])
            opponent_shots_inside_box.append(game['Shots insidebox'][opponent_ind])
            opponent_fouls.append(game['Fouls'][opponent_ind])
            opponent_corners.append(game['Corner Kicks'][opponent_ind])
            opponent_posession.append(game['Ball Possession'][opponent_ind])
            opponent_pass_accuracy.append(game['Passes %'][opponent_ind])
            opponent_goals.append(game['Goals'][opponent_ind])
            result_indicator_raw.append(game['Points'][team_ind])
             
        
        
        team_total_shots_slide = running_mean(team_total_shots, games_slide)[:x]
        team_shots_inside_box_slide = running_mean(team_shots_inside_box, games_slide)[:x]
        team_fouls_slide = running_mean(team_fouls, games_slide)[:x]
        team_corners_slide = running_mean(team_corners, games_slide)[:x]
        team_posession_slide = running_mean(team_posession, games_slide)[:x]
        team_pass_accuracy_slide = running_mean(team_pass_accuracy, games_slide)[:x]
        team_goals_slide = running_mean(team_goals, games_slide)[:x]
        team_goals_target = team_goals[games_slide-xx:]
        opponent_total_shots_slide = running_mean(opponent_total_shots, games_slide)[:x]
        opponent_shots_inside_box_slide = running_mean( opponent_shots_inside_box, games_slide)[:x]
        opponent_fouls_slide = running_mean(opponent_fouls, games_slide)[:x]
        opponent_corners_slide = running_mean(opponent_corners, games_slide)[:x]
        opponent_posession_slide = running_mean(opponent_posession, games_slide)[:x]
        opponent_pass_accuracy_slide = running_mean(opponent_pass_accuracy, games_slide)[:x]
        opponent_goals_slide = running_mean(opponent_goals, games_slide)[:x]
        opponent_goals_target = opponent_goals[games_slide-xx:]
        fix_id_slide = team_fixture_id_dict[team_id][games_slide-xx:]
        result_indicator_slide = result_indicator_raw[games_slide-xx:]

    
        
        t_total_shots.extend(team_total_shots_slide)
        t_shots_inside_box.extend(team_shots_inside_box_slide)
        t_fouls.extend(team_fouls_slide)
        t_corners.extend(team_corners_slide)
        t_posession.extend(team_posession_slide)
        t_pass_accuracy.extend(team_pass_accuracy_slide)
        t_goals.extend(team_goals_slide)
        t_goals_target.extend(team_goals_target)
        o_total_shots.extend(opponent_total_shots_slide)
        o_shots_inside_box.extend(opponent_shots_inside_box_slide)
        o_fouls.extend(opponent_fouls_slide)
        o_corners.extend(opponent_corners_slide)
        o_posession.extend(opponent_posession_slide)
        o_pass_accuracy.extend(opponent_pass_accuracy_slide)
        o_goals.extend(opponent_goals_slide)
        o_goals_target.extend(opponent_goals_target)
        fix_id.extend(fix_id_slide)
        result_indicator.extend(result_indicator_slide)
        o_team_ID.append(team_id)

    
    
    df_ready_for_ml = pd.DataFrame({})  
    df_ready_for_ml['Team Av Shots'] = t_total_shots
    df_ready_for_ml['Team Av Shots Inside Box'] = t_shots_inside_box
    df_ready_for_ml['Team Av Fouls'] = t_fouls
    df_ready_for_ml['Team Av Corners'] = t_corners
    df_ready_for_ml['Team Av Possession'] = t_posession
    df_ready_for_ml['Team Av Pass Accuracy'] = t_pass_accuracy
    df_ready_for_ml['Team Av Goals'] = t_goals
    df_ready_for_ml['Opponent Av Shots'] = o_total_shots
    df_ready_for_ml['Opponent Av Shots Inside Box'] = o_shots_inside_box
    df_ready_for_ml['Opponent Av Fouls'] = o_fouls
    df_ready_for_ml['Opponent Av Corners'] = o_corners
    df_ready_for_ml['Opponent Av Possession'] = o_posession
    df_ready_for_ml['Opponent Av Goals'] = o_goals
    df_ready_for_ml['Opponent Av Pass Accuracy'] = o_pass_accuracy

    df_ready_for_ml['Team Goal Target'] = t_goals_target
    df_ready_for_ml['Opponent Goal Target'] = o_goals_target
    df_ready_for_ml['Target Fixture ID'] = fix_id
    df_ready_for_ml['Result Indicator'] = result_indicator
    if making_predictions:
        df_ready_for_ml['team_id'] = o_team_ID 
    
    
    return df_ready_for_ml



def mod_df(df, making_predictions=False):
    df_sort = df.sort_values('Target Fixture ID')
    df_sort = df_sort.reset_index(drop=True)
    
    
    
    if not making_predictions:
    
        index_to_remove = []
        
        for i in range(0, len(df_sort)-1):
            if i == 0:
                continue
        
            elif i == len(df_sort)-1:
                target_m1 = df_sort['Target Fixture ID'].loc[i-1]
                target = df_sort['Target Fixture ID'].loc[i]
                if target != target_m1:
                    index_to_remove.append(i)
                  
            else:
                target_m1 = df_sort['Target Fixture ID'].loc[i-1]
                target = df_sort['Target Fixture ID'].loc[i]
                target_p1 = df_sort['Target Fixture ID'].loc[i+1]
                if (target != target_m1) and (target != target_p1):
                    index_to_remove.append(i)
                else:
                    continue
                
        df_sort = df_sort.drop(df_sort.index[index_to_remove])    
    
    
    
    df_output = pd.DataFrame({})
    
    df_output['Av Shots Diff'] = df_sort['Team Av Shots'] - df_sort['Opponent Av Shots']
    df_output['Av Shots Inside Box Diff'] = df_sort['Team Av Shots Inside Box'] - df_sort['Opponent Av Shots Inside Box']
    df_output['Av Fouls Diff'] = df_sort['Team Av Fouls'] - df_sort['Opponent Av Fouls']
    df_output['Av Corners Diff'] = df_sort['Team Av Corners'] -df_sort['Opponent Av Corners']
    df_output['Av Possession Diff'] = df_sort['Team Av Possession'] - df_sort['Opponent Av Possession']
    df_output['Av Pass Accuracy Diff'] = df_sort['Team Av Pass Accuracy'] - df_sort['Opponent Av Pass Accuracy']
    df_output['Av Goal Difference'] = df_sort['Team Av Goals'] - df_sort['Opponent Av Goals']
    if not making_predictions:
        df_output['Fixture ID'] = df_sort['Target Fixture ID']
        df_output['Result Indicator'] = df_sort['Result Indicator']
    if making_predictions:
        df_output['Team ID'] = df_sort['team_id']
    
    return df_output



def combining_fixture_id(df): 
    odd_list = []
    for x in range(1, len(df), 2):
        odd_list.append(x)
    even_list = []
    for x in range(0, len(df)-1, 2):
        even_list.append(x)
        
    team_df = df.drop(df.index[odd_list])
    team_df = team_df.reset_index(drop=True)
    opponent_df = df.drop(df.index[even_list])
    opponent_df = opponent_df.reset_index(drop=True)
    
    df_output = pd.DataFrame({})
    df_output['Team Av Shots Diff'] = team_df['Av Shots Diff']
    df_output['Team Av Shots Inside Box Diff'] = team_df['Av Shots Inside Box Diff']
    df_output['Team Av Fouls Diff'] = team_df['Av Fouls Diff']
    df_output['Team Av Corners Diff'] = team_df['Av Corners Diff']
    df_output['Team Av Possession Diff'] = team_df['Av Possession Diff']
    df_output['Team Av Pass Accuracy Diff'] = team_df['Av Pass Accuracy Diff']
    df_output['Team Av Goal Diff'] = team_df['Av Goal Difference']
    df_output['Opponent Av Shots Diff'] = opponent_df['Av Shots Diff']
    df_output['Opponent Av Shots Inside Box Diff'] = opponent_df['Av Shots Inside Box Diff']
    df_output['Opponent Av Fouls Diff'] = opponent_df['Av Fouls Diff']
    df_output['Opponent Av Corners Diff'] = opponent_df['Av Corners Diff']
    df_output['Opponent Av Possession Diff'] = opponent_df['Av Possession Diff']
    df_output['Opponent Av Pass Accuracy Diff'] = opponent_df['Av Pass Accuracy Diff']
    df_output['Opponent Av Goal Diff'] = opponent_df['Av Goal Difference']
    df_output['Fixture ID'] = team_df['Fixture ID']
    df_output['Team Result Indicator'] = team_df['Result Indicator']
    df_output['Opponent Result Indicator'] = opponent_df['Result Indicator']
    
    return df_output



def creating_ml_df(df, making_predictions=False):
    modified_df = mod_df(df)
    df_output = combining_fixture_id(modified_df)
    return df_output




