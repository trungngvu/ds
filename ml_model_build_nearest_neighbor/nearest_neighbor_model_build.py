print('\n\n ---------------- START ---------------- \n')




from os.path import dirname, realpath, sep, pardir
import sys
sys.path.append(dirname(realpath(__file__)) + sep + pardir + sep)

import time
start=time.time()

from ml_functions.ml_model_eval import pred_proba_plot, plot_cross_val_confusion_matrix, plot_learning_curve
from ml_functions.data_processing import scale_df
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_score

plt.close('all')




df_5_saved_name = '2015_2016_2017_2018_2019_2020_2021_2022_2023_prem_df_for_ml_5_v2.txt'
df_10_saved_name = '2015_2016_2017_2018_2019_2020_2021_2022_2023_prem_df_for_ml_10_v2.txt'

test_n_neighbors = True

pred_prob_plot_df10 = True
save_pred_prob_plot_df10 = True
pred_prob_plot_df5 = True
save_pred_prob_plot_df5 = True

save_conf_matrix_df10 = True
save_conf_matrix_df5 = True

save_learning_curve_df10 = True
save_learning_curve_df5 = True

create_final_model = True





with open(f'../prem_clean_fixtures_and_dataframes/{df_5_saved_name}', 'rb') as myFile:
    df_ml_5 = pickle.load(myFile)

with open(f'../prem_clean_fixtures_and_dataframes/{df_10_saved_name}', 'rb') as myFile:
    df_ml_10 = pickle.load(myFile)


df_ml_10 = scale_df(df_ml_10, list(range(14)), [14,15,16])
df_ml_5 = scale_df(df_ml_5, list(range(14)), [14,15,16])

x_10 = df_ml_10.drop(['Fixture ID', 'Team Result Indicator', 'Opponent Result Indicator'], axis=1)
y_10 = df_ml_10['Team Result Indicator']

x_5 = df_ml_5.drop(['Fixture ID', 'Team Result Indicator', 'Opponent Result Indicator'], axis=1)
y_5 = df_ml_5['Team Result Indicator']





def k_nearest_neighbor_train(df, print_result=True, print_result_label=''):
    
    
    x = df.drop(['Fixture ID', 'Team Result Indicator', 'Opponent Result Indicator'], axis=1)
    y = df['Team Result Indicator']
    
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
      
    
    clf = KNeighborsClassifier(n_neighbors=11, weights='distance')
    
    
    clf.fit(x_train, y_train)
    
    if print_result:
        print(print_result_label)
        
        train_data_score = round(clf.score(x_train, y_train) * 100, 1)
        print(f'Training data score = {train_data_score}%')
        
        
        test_data_score = round(clf.score(x_test, y_test) * 100, 1)
        print(f'Test data score = {test_data_score}% \n')
    
    return clf, x_train, x_test, y_train, y_test


ml_10_knn, x10_train, x10_test, y10_train, y10_test = k_nearest_neighbor_train(df_ml_10, print_result_label='DF_ML_10')
ml_5_knn, x5_train, x5_test, y5_train, y5_test = k_nearest_neighbor_train(df_ml_5, print_result_label='DF_ML_5')







df_ml_5_dropto10 = df_ml_5.drop(list(range(0,50)))
ml_5_to10_knn, x5_to10_train, x5_to10_test, y5_to10_train, y5_to10_test = k_nearest_neighbor_train(df_ml_5_dropto10, print_result=False)


y_pred_ml10 = ml_10_knn.predict(x10_test)
y_pred_ml5to10 = ml_5_to10_knn.predict(x10_test)


pred_proba_ml10 = ml_10_knn.predict_proba(x10_test)
pred_proba_ml5_10 = ml_5_to10_knn.predict_proba(x10_test)


pred_proba_ml5and10 = (np.array(pred_proba_ml10) + np.array(pred_proba_ml5_10)) / 2.0
y_pred_ml5and10 = np.argmax(pred_proba_ml5and10, axis=1)


y_pred_ml10_accuracy = round(accuracy_score(y10_test, y_pred_ml10), 3) * 100
y_pred_ml5to10_accuracy = round(accuracy_score(y10_test, y_pred_ml5to10), 3) * 100
y_pred_ml5and10_accuracy = round(accuracy_score(y10_test, y_pred_ml5and10), 3) * 100

print('ENSEMBLE MODEL TESTING')
print(f'Accuracy of df_10 alone = {y_pred_ml10_accuracy}%')
print(confusion_matrix(y10_test, y_pred_ml10), '\n')
print(f'Accuracy of df_5 alone = {y_pred_ml5to10_accuracy}%')
print(confusion_matrix(y10_test, y_pred_ml5to10), '\n')
print(f'Accuracy of df_5 and df_10 combined = {y_pred_ml5and10_accuracy}%')
print(confusion_matrix(y10_test, y_pred_ml5and10), '\n\n')






if test_n_neighbors:

    test_accuracy_compiled = []
    for i in range(1, 10, 1):
        test_accuracy = []
        for n in range(1, 50, 1):
            x_train, x_test, y_train, y_test = train_test_split(x_10, y_10, test_size=0.2)
            clf = KNeighborsClassifier(n_neighbors=n, weights='uniform')
            clf.fit(x_train, y_train)
            test_accuracy.append(round(clf.score(x_test, y_test) * 100, 1))
        test_accuracy_compiled.append(test_accuracy)
    test_accuracy_compiled_np = np.transpose(np.array(test_accuracy_compiled))
    test_accuracy_compiled_av = np.mean(test_accuracy_compiled_np, axis=1)
    
    fig, ax = plt.subplots()
    ax.plot(range(1,50, 1), test_accuracy_compiled_av, label='Weights = Uniform')
    ax.set_xlabel('n_neighbors') 
    ax.set_ylabel('Accuracy Score %') 
    ax.set_title('Testing k values ml_10', y=1, fontsize=14, fontweight='bold');
    ax.legend(loc=4)
    plt.savefig('figures/ml_10_testing_k_values_uniform.png')

    
    test_accuracy_compiled = []
    for i in range(1, 10, 1):
        test_accuracy = []
        for n in range(1, 50, 1):
            x_train, x_test, y_train, y_test = train_test_split(x_10, y_10, test_size=0.2)
            clf = KNeighborsClassifier(n_neighbors=n, weights='distance')
            clf.fit(x_train, y_train)
            test_accuracy.append(round(clf.score(x_test, y_test) * 100, 1))
        test_accuracy_compiled.append(test_accuracy)
    test_accuracy_compiled_np = np.transpose(np.array(test_accuracy_compiled))
    test_accuracy_compiled_av = np.mean(test_accuracy_compiled_np, axis=1)

    fig, ax = plt.subplots()
    ax.plot(range(1,50, 1), test_accuracy_compiled_av, label='Weights = Distance')
    ax.set_xlabel('n_neighbors')
    ax.set_ylabel('Accuracy Score %')
    ax.set_title('Testing k values ml_10', y=1, fontsize=14, fontweight='bold');
    ax.legend(loc=4)
    plt.savefig('figures/ml_10_testing_k_values_distance.png')
    





skf = StratifiedKFold(n_splits=5, shuffle=True)

cv_score_av = round(np.mean(cross_val_score(ml_10_knn, x_10, y_10, cv=skf))*100,1)
print('Cross-Validation Accuracy Score ML10: ', cv_score_av, '%\n')

cv_score_av = round(np.mean(cross_val_score(ml_5_knn, x_5, y_5, cv=skf))*100,1)
print('Cross-Validation Accuracy Score ML5: ', cv_score_av, '%\n')




if pred_prob_plot_df10:
    fig = pred_proba_plot(ml_10_knn, 
                          x_10, 
                          y_10, 
                          no_iter=5, 
                          no_bins=36, 
                          x_min=0.3, 
                          classifier='Nearest Neighbor (ml_10)')
    if save_pred_prob_plot_df10:
        fig.savefig('figures/ml_10_nearest_neighbor_pred_proba.png')

if pred_prob_plot_df5:
    fig = pred_proba_plot(ml_5_knn, 
                          x_5, 
                          y_5, 
                          no_iter=50, 
                          no_bins=36, 
                          x_min=0.3, 
                          classifier='Nearest Neighbor (ml_5)')
    if save_pred_prob_plot_df10:
            fig.savefig('figures/ml_5_nearest_neighbor_pred_proba.png')




plot_cross_val_confusion_matrix(ml_10_knn, 
                                x_10, 
                                y_10, 
                                display_labels=('team loses', 'draw', 'team wins'), 
                                title='Nearest Neighbor Confusion Matrix ML10', 
                                cv=skf)
if save_conf_matrix_df10:
    plt.savefig('figures/ml_10_confusion_matrix_cross_val_nearest_neighbor.png')

plot_cross_val_confusion_matrix(ml_5_knn, 
                                x_5, 
                                y_5, 
                                display_labels=('team loses', 'draw', 'team wins'), 
                                title='Nearest Neighbor Confusion Matrix ML5', 
                                cv=skf)
if save_conf_matrix_df5:
    plt.savefig('figures/ml_5_confusion_matrix_cross_val_nearest_neighbor.png')




plot_learning_curve(ml_10_knn, 
                    x_10, 
                    y_10, 
                    training_set_size=10, 
                    x_max=2500, 
                    title='Learning Curve - Nearest Neighbor DF_10', 
                    leg_loc=1)
if save_learning_curve_df10:
    plt.savefig('figures/ml_10_nearest_neighbor_learning_curve.png')

plot_learning_curve(ml_5_knn, 
                    x_5, 
                    y_5, 
                    training_set_size=10, 
                    x_max=2500, 
                    title='Learning Curve - Nearest Neighbor DF_5', 
                    leg_loc=1)
if save_learning_curve_df5:
    plt.savefig('figures/ml_5_nearest_neighbor_learning_curve.png')

 




if create_final_model:
    
    
    ml_5_knn = KNeighborsClassifier(n_neighbors=11, weights='distance')
    ml_5_knn.fit(x_5, y_5)
    
    
    ml_10_knn = KNeighborsClassifier(n_neighbors=11, weights='distance')
    ml_10_knn.fit(x_10, y_10)
    
    with open('ml_models/knn_model_5.pk1', 'wb') as myFile:
        pickle.dump(ml_5_knn, myFile)

    with open('ml_models/knn_model_10.pk1', 'wb') as myFile:
        pickle.dump(ml_10_knn, myFile)

   


print('\n', 'Script runtime:', round(((time.time()-start)/60), 2), 'minutes')
print(' ----------------- END ----------------- \n')
