import pandas as pd
import numpy as np
import itertools
import time
from scipy.optimize import minimize, fmin_l_bfgs_b
from scipy.stats import norm
import cufflinks as cf
import sys
#%matplotlib inline
#%config InlineBackend.figure_format = 'retina'
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import math
from scipy.stats import mode

# for graph drawing
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

# use the most simple neural network
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error

def regression_function():
    # get input
    data_name = input('Enter the data file name with directory: ')
    sep_index = input("Select the data encoding format(1 = 'a b c' or 2 = 'a,b,c'): ")
    output_filename = input("Enter your output file name with directory: ")

    # find seperator
    if sep_index == '1':
        sep = ' '
    elif sep_index == '2':
        sep = ','
    else:
        raise('you should enter only 1 or 2')


    # make pandas data frame
    df = pd.read_table(data_name, sep = sep, header=None)
    df2 = df.assign(constant = 1)

    # choose y and X
    y_index = input("Enter the response variable y's column number(starts from 0) : ")
    y = df.loc[:,int(y_index)].values
    X = df2.loc[:,df2.columns != int(y_index)].values
    X_colname = df2.loc[:,df2.columns != int(y_index)].columns

    # calculate statistics
    Beta = np.matmul(np.matmul(np.linalg.inv(np.matmul( X.transpose(), X)) ,X.transpose()), y)
    Beta_names = ['Beta{}'.format(i+1) for i in range(len(Beta) - 1)]

    fitted_value = np.matmul(X, Beta)

    SSE =  sum((y - fitted_value)**2)
    MSE = SSE/len(y)
    SSTO = sum((y-y.mean())**2)
    R_square = 1-(SSE/SSTO)

    # print Coefficients
    print('Coefficients')
    print('--------------')
    print('Constant: {}'.format(np.round(Beta[X_colname == 'constant'],3)[0]))
    for i in range(len(Beta)-1):
        print('{}: {}'.format(Beta_names[i], np.round(Beta[X_colname != 'constant'],3)[i]))
    print('\n')

    # print ID, Actual values, Fitted values
    print('ID, Actual values, Fitted values')
    print('---------------------------------')
    for j in range(len(y)):
        print('{}, {}, {}'.format(j+1, y[j], np.round(fitted_value,3)[j]))
    print('\n')

    # print Model Summary
    print('Model Summary')
    print('-----------------')
    print('R-square: {}'.format(np.round(R_square,4)))
    print('MSE: {}'.format(np.round(MSE,4)))
    print('\n')

    # save output as output.txt
    f = open(output_filename , 'w')

    # f.write Coefficients
    f.write('Coefficients\n')
    f.write('--------------\n')
    f.write('Constant: {}\n'.format(np.round(Beta[X_colname == 'constant'],3)[0]))
    for i in range(len(Beta)-1):
        f.write('{}: {}\n'.format(Beta_names[i], np.round(Beta[X_colname != 'constant'],3)[i]))
    f.write('\n')
        
    # f.write ID, Actual values, Fitted values
    f.write('ID, Actual values, Fitted values\n')
    f.write('---------------------------------\n')
    for j in range(len(y)):
        f.write('{}, {}, {}\n'.format(j+1, y[j], np.round(fitted_value,3)[j]))
    f.write('\n')
        
    # f.write model summary
    f.write('Model Summary\n')
    f.write('--------------\n')
    f.write('R-square: {}\n'.format(np.round(R_square,4)))
    f.write('MSE: {}\n'.format(np.round(MSE,4)))
    f.write('\n')
    f.close()
    ################################
    ## end of Regression function ##
    ################################
    
    
def classification_function():
    
    # get input
    training_data = input('Enter the training data file name with directory: ')
    test_data = input('Enter the test data file name with directory(if Naive Bayes, just enter training data again): ')
    sep_index = input("Select the data encoding format(1 = 'a b c' or 2 = 'a,b,c'): ")
    output_filename = input("Enter your output file name with directory: ")
    
    # find seperator
    if sep_index == '1':
        sep = ' '
    elif sep_index == '2':
        sep = ','
    else:
        raise('you should enter only 1 or 2')
        
    # make pandas data frame
    df_train = pd.read_table(training_data, sep = sep, header=None)
    df_test = pd.read_table(test_data, sep = sep, header=None)

    # choose y and X
    y_index = input("Enter the response variable y's column number(starts from 0) : ")
    y_train = df_train.loc[:,int(y_index)]
    y_test = df_test.loc[:,int(y_index)]
    X_train = df_train.loc[:,df_train.columns != int(y_index)]
    X_test = df_test.loc[:,df_test.columns != int(y_index)]
    #X_colname = df.loc[:,df.columns != int(y_index)].columns
    
    # ask which classification model to use
    if len(y_train.unique()) > 2:
        classification_model = input('Which classification do you want(1 = LDA, 2 = QDA, 3 = RDA, 7 = Bagging_LDA) : ')
    elif len(y_train.unique()) == 2:
        classification_model = input('''Which classification do you want(1 = LDA, 2 = QDA, 3 = RDA, 
        4 = Logistic Regression, 5 = Naive_Bayes, 6 = Decision_Tree, 7 = Bagging_LDA ) : ''')
    else:
        raise('got only one class')
    

    def print_classification_result(y_train=None, y_test=None,predicted_classes_train=None,
                                    predicted_classes_test=None, accuracy_train=None, accuracy_test=None,
                                    confusion_train=None,confusion_test=None,
                                   prob_train=None, prob_test=None, 
                                    best_X_num=None, best_cutting_point=None, root_node_count=None,
                                    predicted_left=None, predicted_right=None, left_node_count=None, 
                                    right_node_count=None, model=None):
    
        if len(confusion_train.index) == 2:
            min_class = y_train.unique().min()
            sens_train = sum(predicted_classes_train[y_train == min_class] == min_class)
            spec_train = sum(predicted_classes_train[y_train == (min_class+1)] == (min_class+1))
            sensitivity_train = sens_train/confusion_train.sum(axis=1)[min_class]
            specificity_train = spec_train/confusion_train.sum(axis=1)[min_class+1]
            if model not in ['Naive_Bayes']:
                sens_test = sum(predicted_classes_test[y_test == min_class] == min_class)
                spec_test = sum(predicted_classes_test[y_test == (min_class+1)] == (min_class+1))
                sensitivity_test = sens_test/confusion_test.sum(axis=1)[min_class]
                specificity_test = spec_test/confusion_test.sum(axis=1)[min_class+1]
                
        if model in ['Decision_Tree']:
            print('Tree Structure')
            print('    Node 1: x{} <= {} {}'.format(best_X_num+1, best_cutting_point, root_node_count))
            print('      Node 2: {} {}'.format(predicted_left, left_node_count))
            print('      Node 3: {} {}'.format(predicted_right, right_node_count))
            print('\n')
            
        if model in ['Naive_Bayes', 'Logistic_Regression']:
            print('ID, Actual class, Resub pred, Pred Prob')
            print('-----------------------------')
            for i in range(len(y_train)):
                print('{}, {}, {}, {}'.format(i, y_train[i], predicted_classes_train[i], prob_train[i]))
            print('\n')
        elif model in ['Bagging_LDA']:
            print('(1) LDA - no bagging')
            print('ID, Actual class, LDA-nobagging pred')
            print('-----------------------------')
            for i in range(len(y_train)):
                print('{}, {}, {}'.format(i, y_train[i], predicted_classes_train[i]))
            print('\n')
        else:
            print('ID, Actual class, Resub pred')
            print('-----------------------------')
            for i in range(len(y_train)):
                print('{}, {}, {}'.format(i, y_train[i], predicted_classes_train[i]))
            print('\n')

        if model in ['Bagging_LDA']:
            print('Confusion Matrix (LDA - no bagging)')
        else:
            print('Confusion Matrix (Resubstitution)')
        print('----------------------------------')
        print(confusion_train)
        print('\n')

        if model in ['Bagging_LDA']:
            print('Model Summary (LDA - no bagging)')
        else:
            print('Model Summary (Resubstitution)')
        print('------------------------------')
        print('Overall accuracy = {}'.format(accuracy_train))

        if len(confusion_train.index) == 2:
            print('Sensitivity = {}'.format(sensitivity_train))
            print('Specificity = {}'.format(specificity_train))
        print('\n')
        
        if model in ['Logistic_Regression']:
            print('ID, Actual class, Test pred, Pred Prob')
            print('-----------------------------')
            for i in range(len(y_test)):
                print('{}, {}, {}, {}'.format(i, y_test[i], predicted_classes_test[i], prob_test[i]))
            print('\n')

        elif model in ['LDA', 'QDA', 'RDA', 'Decision_Tree']:
            print('ID, Actual class, Test pred')
            print('-----------------------------')
            for i in range(len(y_test)):
                print('{}, {}, {}'.format(i, y_test[i], predicted_classes_test[i]))
            print('\n')

        elif model in ['Bagging_LDA']:
            print('(2)  LDA - bagging')
            print('ID, Actual class, LDA-bagging pred')
            print('-----------------------------')
            for i in range(len(y_test)):
                print('{}, {}, {}'.format(i, y_test[i], predicted_classes_test[i]))
            print('\n')

        else:
            pass
            
        if model in ['LDA','QDA','RDA','Logistic_Regression', 'Decision_Tree']:
            print('Confusion Matrix (Test)')
            print('----------------------------------')
            print(confusion_test)
            print('\n')
            print('Model Summary (Test)')
            print('------------------------------')
            print('Overall accuracy = {}'.format(accuracy_test))
            if len(confusion_train.index) == 2:
                print('Sensitivity = {}'.format(sensitivity_test))
                print('Specificity = {}'.format(specificity_test))

        if model in['Bagging_LDA']:
            print('Confusion Matrix (LDA - bagging)')
            print('----------------------------------')
            print(confusion_test)
            print('\n')
            print('Model Summary (LDA - bagging)')
            print('------------------------------')
            print('Overall accuracy = {}'.format(accuracy_test))            
        

        # save output as output.txt
        f = open(output_filename , 'w')
        
        if model in ['Decision_Tree']:
            f.write('Tree Structure\n')
            f.write('    Node 1: x{} <= {} {}\n'.format(best_X_num+1, best_cutting_point, root_node_count))
            f.write('      Node 2: {} {}\n'.format(predicted_left, left_node_count))
            f.write('      Node 3: {} {}\n'.format(predicted_right, right_node_count))
            f.write('\n')
            
        if model in ['Naive_Bayes', 'Logistic_Regression']:
            f.write('ID, Actual class, Resub pred, Pred Prob \n')
            f.write('-----------------------------\n')
            for i in range(len(y_train)):
                f.write('{}, {}, {}, {}\n'.format(i, y_train[i], predicted_classes_train[i], prob_train[i]))
            f.write('\n')
        elif model in ['Bagging_LDA']:
            f.write('(1) LDA - no bagging\n')
            f.write('ID, Actual class, LDA-nobagging pred\n')
            f.write('-----------------------------\n')
            for i in range(len(y_train)):
                f.write('{}, {}, {}\n'.format(i, y_train[i], predicted_classes_train[i]))
            f.write('\n')
        else:
            f.write('ID, Actual class, Resub pred\n')
            f.write('-----------------------------\n')
            for i in range(len(y_train)):
                f.write('{}, {}, {}\n'.format(i, y_train[i], predicted_classes_train[i]))
            f.write('\n')

        if model in ['Bagging_LDA']:
            f.write('Confusion Matrix (LDA - no bagging)\n')
        else:
            f.write('Confusion Matrix (Resubstitution)\n')
        f.write('----------------------------------\n')
        f.write(confusion_train.to_string())
        f.write('\n')
        f.write('\n')
        if model in ['Bagging_LDA']:
            f.write('Model Summary (LDA - no bagging)\n')
        else:
            f.write('Model Summary (Resubstitution)\n')
        f.write('------------------------------\n')
        f.write('Overall accuracy = {}\n'.format(accuracy_train))
        if len(confusion_train.index) == 2:
            f.write('Sensitivity = {}\n'.format(sensitivity_train))
            f.write('Specificity = {}\n'.format(specificity_train))
        f.write('\n')
        
        if model in ['Logistic_Regression']:
            f.write('ID, Actual class, Test pred, Pred Prob\n')
            f.write('-----------------------------\n')
            for i in range(len(y_test)):
                f.write('{}, {}, {}, {}\n'.format(i, y_test[i], predicted_classes_test[i], prob_test[i]))
            f.write('\n')
        elif model in ['LDA', 'QDA', 'RDA', 'Decision_Tree']:
            f.write('ID, Actual class, Test pred\n')
            f.write('-----------------------------\n')
            for i in range(len(y_test)):
                f.write('{}, {}, {}\n'.format(i, y_test[i], predicted_classes_test[i]))
            f.write('\n')
        elif model in ['Bagging_LDA']:
            f.write('(2)  LDA - bagging\n')
            f.write('ID, Actual class, LDA-bagging pred\n')
            f.write('-----------------------------\n')
            for i in range(len(y_test)):
                f.write('{}, {}, {}\n'.format(i, y_test[i], predicted_classes_test[i]))
            f.write('\n')
        else:
            pass
            
        if model in ['LDA','QDA','RDA','Logistic_Regression', 'Decision_Tree']:
            f.write('Confusion Matrix (Test)\n')
            f.write('----------------------------------\n')
            f.write(confusion_test.to_string())
            f.write('\n')
            f.write('\n')
            f.write('Model Summary (Test)\n')
            f.write('------------------------------\n')
            f.write('Overall accuracy = {}\n'.format(accuracy_test))
            if len(confusion_train.index) == 2:
                f.write('Sensitivity = {}\n'.format(sensitivity_test))
                f.write('Specificity = {}\n'.format(specificity_test))

        if model in['Bagging_LDA']:
            f.write('Confusion Matrix (LDA - bagging)\n')
            f.write('----------------------------------\n')
            f.write(confusion_test.to_string())
            f.write('\n')
            f.write('\n')
            f.write('Model Summary (LDA - bagging)\n')
            f.write('------------------------------\n')
            f.write('Overall accuracy = {}\n'.format(accuracy_test))
        f.close()
        ###########################
        ## end of print function ##
        ###########################
    
    def LDA(X_train,y_train,X_test,y_test):
        
        # get variables to calculate LDF
        classes = np.sort(y_train.unique())
        X_bar_by_class = dict()
        n_class = dict()  # number of samples by classes(안씀)
        S_by_class = dict() # sample covariance by classes(안씀)
        equation_for_S_p = np.zeros((len(X_train.columns),len(X_train.columns)))
        for class_num in classes:
            X_k = X_train.loc[y_train==class_num,:]
            X_bar_by_class['X_bar_{}'.format(class_num)] = X_k.mean() # 독립변수의 표본평균 벡터
            n_class['n_{}'.format(class_num)] = len(X_k)  # 클래스에 해당하는 샘플의 갯수
            S_by_class['S_{}'.format(class_num)] = X_k.cov()  # 클래스에 해당하는 샘플의 독립변수별 표본분산
            equation_for_S_p += np.array(X_k.cov() * (len(X_k)-1))  # pooled covariance 편하게 계산 위해 미리 만들어둠
            

        # calculate S_p (diagonal matrix)
        S_p = equation_for_S_p/(len(X_train)-len(classes))
        
        # calculate LDF for train data
        fianl_d_train = np.zeros((len(X_train),len(classes)))
        for class_num in classes:
            X_name = 'X_bar_{}'.format(class_num)
            LDF_1_train = np.matmul(np.matmul(X_bar_by_class[X_name],np.linalg.inv(S_p)), X_train.transpose())
            LDF_2_train = np.matmul(np.matmul(X_bar_by_class[X_name],np.linalg.inv(S_p)) ,X_bar_by_class[X_name])/2
            d_k_train = LDF_1_train - LDF_2_train
            fianl_d_train[:,(class_num-1)] += d_k_train

        # get predicted classes for train data => caution : column starts from 0, while class starts from 1
        final_d_dataframe_train = pd.DataFrame(fianl_d_train)
        final_d_dataframe_train.columns = classes
        predicted_classes_train = final_d_dataframe_train.idxmax(axis=1)
        
        ##### test data ######
        # calculate LDF for test data
        fianl_d_test = np.zeros((len(X_test),len(classes)))
        for class_num in classes:
            X_name = 'X_bar_{}'.format(class_num)
            LDF_1_test = np.matmul(np.matmul(X_bar_by_class[X_name],np.linalg.inv(S_p)), X_test.transpose())
            LDF_2_test = np.matmul(np.matmul(X_bar_by_class[X_name],np.linalg.inv(S_p)) ,X_bar_by_class[X_name])/2
            d_k_test = LDF_1_test - LDF_2_test
            fianl_d_test[:,(class_num-1)] += d_k_test
            
        # get predicted classes for test data => caution : column starts from 0, while class starts from 1
        final_d_dataframe_test = pd.DataFrame(fianl_d_test)
        final_d_dataframe_test.columns = classes
        predicted_classes_test = final_d_dataframe_test.idxmax(axis=1)
        
        # make confusion matrix
        actual_predicted_train = pd.concat([y_train,predicted_classes_train],axis=1)
        actual_predicted_train.columns = ['Actual class', 'Predicted Class']
        actual_predicted_test = pd.concat([y_test,predicted_classes_test],axis=1)
        actual_predicted_test.columns = ['Actual class', 'Predicted Class']
        confusion_train = actual_predicted_train.groupby(['Actual class', 'Predicted Class']).size().unstack().fillna(0)
        confusion_test = actual_predicted_test.groupby(['Actual class', 'Predicted Class']).size().unstack().fillna(0)
        
        # calculate model accuracy
        accuracy_train = sum(y_train==predicted_classes_train)/len(y_train)
        accuracy_test = sum(y_test==predicted_classes_test)/len(y_test)
        
        # print result
        print_classification_result(y_train=y_train, y_test=y_test,predicted_classes_train=predicted_classes_train,
                                    predicted_classes_test=predicted_classes_test, accuracy_train=accuracy_train, 
                                    accuracy_test=accuracy_test,confusion_train=confusion_train,
                                    confusion_test=confusion_test,prob_train=None, prob_test=None, 
                                    best_X_num=None, best_cutting_point=None, root_node_count=None,
                                    predicted_left=None, predicted_right=None, left_node_count=None, 
                                    right_node_count=None, model='LDA')
        #########################
        ## end of LDA function ##
        #########################
        
        
    def QDA(X_train,y_train,X_test,y_test):
        # get variables to calculate QDF
        classes = np.sort(y_train.unique())
        X_bar_by_class = dict()
        n_class = dict()  # number of samples by classes(안씀)
        S_by_class = dict() # sample covariance by classes
        for class_num in classes:
            X_k = X_train.loc[y_train==class_num,:]
            X_bar_by_class['X_bar_{}'.format(class_num)] = X_k.mean() # 독립변수의 표본평균 벡터
            n_class['n_{}'.format(class_num)] = len(X_k)  # 클래스에 해당하는 샘플의 갯수
            S_by_class['S_{}'.format(class_num)] = X_k.cov()  # 클래스에 해당하는 샘플의 독립변수별 표본분산
        
        
        # calculate QDF for train data
        fianl_d_train = np.zeros((len(X_train),len(classes)))
        for class_num in classes:
            X_name = 'X_bar_{}'.format(class_num)
            S_name = 'S_{}'.format(class_num)
            
            S_k = S_by_class[S_name]
            X_bar_k = X_bar_by_class[X_name]
            X_train_minus_X_bar = X_train - X_bar_k
            
            QDF_1_train = (-1/2)*np.log(np.linalg.det(S_k))  
            QDF_2_train = np.diag((1/2)*np.matmul(np.matmul(X_train_minus_X_bar,np.linalg.inv(S_k))
                                           ,X_train_minus_X_bar.transpose()))  # 한번에 행렬곱하므로 대각성분이 결과값
            d_k_train = QDF_1_train - QDF_2_train  # equal prior
            fianl_d_train[:,(class_num-1)] += d_k_train
        
        
        # get predicted classes for train data => caution : column starts from 0, while class starts from 1
        final_d_dataframe_train = pd.DataFrame(fianl_d_train)
        final_d_dataframe_train.columns = classes
        predicted_classes_train = final_d_dataframe_train.idxmax(axis=1)
        
        ##### test data ######
        # calculate QDF for test data
        fianl_d_test = np.zeros((len(X_test),len(classes)))
        for class_num in classes:
            X_name = 'X_bar_{}'.format(class_num)
            S_name = 'S_{}'.format(class_num)
            
            S_k = S_by_class[S_name]
            X_bar_k = X_bar_by_class[X_name]
            X_test_minus_X_bar = X_test - X_bar_k
            
            QDF_1_test = (-1/2)*np.log(np.linalg.det(S_k))
            QDF_2_test = np.diag((1/2)*np.matmul(np.matmul(X_test_minus_X_bar,np.linalg.inv(S_k))
                                          ,X_test_minus_X_bar.transpose()))
            d_k_test = QDF_1_test - QDF_2_test
            fianl_d_test[:,(class_num-1)] += d_k_test
        
            
        # get predicted classes for test data => caution : column starts from 0, while class starts from 1
        final_d_dataframe_test = pd.DataFrame(fianl_d_test)
        final_d_dataframe_test.columns = classes
        predicted_classes_test = final_d_dataframe_test.idxmax(axis=1)
        
        # make confusion matrix
        actual_predicted_train = pd.concat([y_train,predicted_classes_train],axis=1)
        actual_predicted_train.columns = ['Actual class', 'Predicted Class']
        actual_predicted_test = pd.concat([y_test,predicted_classes_test],axis=1)
        actual_predicted_test.columns = ['Actual class', 'Predicted Class']
        confusion_train = actual_predicted_train.groupby(['Actual class', 'Predicted Class']).size().unstack().fillna(0)
        confusion_test = actual_predicted_test.groupby(['Actual class', 'Predicted Class']).size().unstack().fillna(0)
        
        # calculate model accuracy
        accuracy_train = sum(y_train==predicted_classes_train)/len(y_train)
        accuracy_test = sum(y_test==predicted_classes_test)/len(y_test)

        # print result
        print_classification_result(y_train=y_train, y_test=y_test, predicted_classes_train=predicted_classes_train,
                                    predicted_classes_test=predicted_classes_test, accuracy_train=accuracy_train, 
                                    accuracy_test=accuracy_test,confusion_train=confusion_train,
                                    confusion_test=confusion_test,prob_train=None, prob_test=None, 
                                    best_X_num=None, best_cutting_point=None, root_node_count=None,
                                    predicted_left=None, predicted_right=None, left_node_count=None, 
                                    right_node_count=None, model='QDA')
        
        #########################
        ## end of QDA function ##
        #########################
        
        
    def RDA(X_train,y_train,X_test,y_test):
        # get variables to calculate RDF
        classes = np.sort(y_train.unique())
        X_bar_by_class = dict()
        n_class = dict()  # number of samples by classes(안씀)
        S_by_class = dict() # sample covariance by classes
        equation_for_S_p = np.zeros((len(X_train.columns),len(X_train.columns)))
        for class_num in classes:
            X_k = X_train.loc[y_train==class_num,:]
            X_bar_by_class['X_bar_{}'.format(class_num)] = X_k.mean() # 독립변수의 표본평균 벡터
            n_class['n_{}'.format(class_num)] = len(X_k)  # 클래스에 해당하는 샘플의 갯수
            S_by_class['S_{}'.format(class_num)] = X_k.cov()  # 클래스에 해당하는 샘플의 독립변수별 표본분산
            equation_for_S_p += np.array(X_k.cov() * (len(X_k)-1))  # pooled covariance 편하게 계산 위해 미리 만들어둠
        
        # calculate S_p (diagonal matrix)
        S_p = equation_for_S_p/(len(X_train)-len(classes))
        
        # calculate sigma^2 * I => diagonal matrix of S_p
        sigma_square_I = np.diag(np.diag(S_p))
            
        # 목표: a, r을 컬럼으로 가진 데이터프레임에서 각각의 accuracy 계산해서 열로 붙이고 마지막에 최종 프린트본은 하나만 만들자
        a_list = [a for a in np.arange(0.0,1.05,0.05)]
        
        # [(a, r), (a,r), ....]
        a_r_combi = pd.Series([a for a in itertools.permutations(a_list,2)])
        
        
        ## 3d plot function
        def RDA_3d_plot(a_r_combi, accuracy_result ):
            fig = plt.figure(figsize=(15,10))
            ax = fig.gca(projection='3d')
            fig.suptitle('Alpha Gamma Accuracy', fontsize=30, fontweight='bold')
        
            # Make data.
            X = np.array([i[0] for i in a_r_combi])
            Y = np.array([i[1] for i in a_r_combi])
            Z = accuracy_result
            X.shape = (21,-1)
            Y.shape = (21,-1)
            Z.shape = (21,-1)
        
            # Plot the surface.
            surf = ax.plot_surface(X, Y, Z,rstride=1, cstride=1, cmap=cm.coolwarm,
                                   linewidth=0, antialiased=False)
        
            # Customize the z axis.
            ax.set_xlim(-0.01, 1.01)
            ax.set_ylim(-0.01, 1.01)
            ax.set_zlim(-0.01, 1.01)
            ax.zaxis.set_major_locator(LinearLocator(10))
            ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
            ax.set_xlabel('alpha')
            ax.set_ylabel('gamma')
            ax.set_zlabel('accuracy')
        
            # Add a color bar which maps values to colors.
            fig.colorbar(surf, shrink=0.5, aspect=5)
        
            # show and save figure
            plt.show()
            fig.savefig('RDA_3d.png')   # save the figure to file
            plt.close(fig) 
        
        def accuracy_calculator(a_r_combi, map_or_final):
            '''
            if map_or_final = 'map', run function to get overall accuracy
            if map_or_final = 'final', fun functino to get one best output
            '''
            
            fianl_d_train = np.zeros((len(X_train),len(classes)))
            fianl_d_test = np.zeros((len(X_test),len(classes)))
            
            # 각 행의 성분
            a = a_r_combi[0]
            r = a_r_combi[1]
            
            for class_num in classes:
                X_name = 'X_bar_{}'.format(class_num)
                S_name = 'S_{}'.format(class_num)
        
                S_k = S_by_class[S_name]
                X_bar_k = X_bar_by_class[X_name]
                X_train_minus_X_bar = X_train - X_bar_k
                X_test_minus_X_bar = X_test - X_bar_k
                # RDF's special point
                S_k_a_r = a*S_k + (1-a)*(r*S_p + (1-r)*sigma_square_I)
        
                # calculate RDF for train data
                RDF_1_train = (-1/2)*np.log(np.linalg.det(S_k_a_r))  
                RDF_2_train = np.diag((1/2)*np.matmul(np.matmul(X_train_minus_X_bar,np.linalg.inv(S_k_a_r))
                                               ,X_train_minus_X_bar.transpose()))  # 한번에 행렬곱하므로 대각성분이 결과값
                d_k_train = RDF_1_train - RDF_2_train  # equal prior
                fianl_d_train[:,(class_num-1)] += d_k_train
        
        
                # calculate RDF for test data
                RDF_1_test = (-1/2)*np.log(np.linalg.det(S_k_a_r))
                RDF_2_test = np.diag((1/2)*np.matmul(np.matmul(X_test_minus_X_bar,np.linalg.inv(S_k_a_r))
                                              ,X_test_minus_X_bar.transpose()))
                d_k_test = RDF_1_test - RDF_2_test
                fianl_d_test[:,(class_num-1)] += d_k_test
        
            # get predicted classes for train data => caution : column starts from 0, while class starts from 1
            final_d_dataframe_train = pd.DataFrame(fianl_d_train)
            final_d_dataframe_train.columns = classes
            predicted_classes_train = final_d_dataframe_train.idxmax(axis=1)
        
            # get predicted classes for test data => caution : column starts from 0, while class starts from 1
            final_d_dataframe_test = pd.DataFrame(fianl_d_test)
            final_d_dataframe_test.columns = classes
            predicted_classes_test = final_d_dataframe_test.idxmax(axis=1)
                
            # calculate model accuracy
            accuracy_train = sum(y_train==predicted_classes_train)/len(y_train)
            accuracy_test = sum(y_test==predicted_classes_test)/len(y_test)
            
            # two types of return => to prevent hard coding
            if map_or_final == 'map':
                return accuracy_test
            elif map_or_final == 'final':
                # make confusion matrix
                actual_predicted_train = pd.concat([y_train,predicted_classes_train],axis=1)
                actual_predicted_train.columns = ['Actual class', 'Predicted Class']
                actual_predicted_test = pd.concat([y_test,predicted_classes_test],axis=1)
                actual_predicted_test.columns = ['Actual class', 'Predicted Class']
                confusion_train = actual_predicted_train.groupby(['Actual class', 'Predicted Class']).size().unstack().fillna(0)
                confusion_test = actual_predicted_test.groupby(['Actual class', 'Predicted Class']).size().unstack().fillna(0)
                    
                # calculate model accuracy
                accuracy_train = sum(y_train==predicted_classes_train)/len(y_train)
                accuracy_test = sum(y_test==predicted_classes_test)/len(y_test)
            
                return (predicted_classes_train, predicted_classes_test , accuracy_train, accuracy_test,
                        confusion_train, confusion_test)
            else:
                raise('Only map or final is allowed')
            
        
        # run the function for the best result
        accuracy_result = a_r_combi.map(lambda x: accuracy_calculator(x, 'map'))
        best_model = a_r_combi[accuracy_result.idxmax()]   #(a,r)
        
        # get the result of final best model to print 
        best_model_result = accuracy_calculator(best_model, 'final')
                            
        print_classification_result(y_train=y_train, y_test=y_test, predicted_classes_train=best_model_result[0],
                                    predicted_classes_test=best_model_result[1], accuracy_train=best_model_result[2], 
                                    accuracy_test=best_model_result[3],confusion_train=best_model_result[4],
                                    confusion_test=best_model_result[5],prob_train=None, prob_test=None, 
                                    best_X_num=None, best_cutting_point=None, root_node_count=None,
                                    predicted_left=None, predicted_right=None, left_node_count=None, 
                                    right_node_count=None, model='RDA')

        
                
        RDA_3d_plot(a_r_combi, np.array(accuracy_result))
        #########################
        ## end of RDA function ##
        #########################
        
    def Logistic_regression(X_train,y_train,X_test,y_test):
        # let's find how many classes out there
        classes = np.sort(y_train.unique())
        
        # calculate difference from 0 to y_classes => for logistic regression
        y_diff = classes[0]
        y_train_for_cost = y_train - y_diff
        y_test_for_cost = y_test - y_diff
        
        # define logit function
        def logit_function(theta, X):
            # theta * X
            z_train = np.squeeze(np.dot(np.array(X), theta.transpose()))
            # logit function
            logit_result = 1/(1+np.exp(-z_train))
            
            return logit_result
        
        # define cost function
        def cost_function(theta, X, y):
            # logit function
            logit_result = logit_function(theta, X)
            # cost
            cost = (-sum(y*np.log(logit_result) + (1-y)*np.log(1.1-logit_result)/len(y)))
            
            return cost
        
        # define gradient function
        def logistic_grad(theta, X, y):
            # logit function
            logit_result = logit_function(theta, X)
            # calculate gradient
            final_grad = np.array((logit_result - y).T.dot(X))
            
            return final_grad
        
        # define predict function
        def predict_values(theta, X, thres):
            # logit function
            logit_result = logit_function(theta, X)
            # get predicted value
            pred_value = np.where(logit_result >= thres, 1, 0)
        
            return pred_value
        
        # initiate coordinates
        coordinate_name = ['X_{}'.format(i) for i in range(len(X_train.columns))]
        coordinate_train = np.random.randn(1,X_train.shape[1])
        
        # normalize data
        norm_X_train = (X_train - np.mean(X_train, axis=0)) / np.std(X_train, axis=0)
        norm_X_test = (X_test - np.mean(X_test, axis=0)) / np.std(X_test, axis=0)
        
        # run optimize function
        hyper_parm = (norm_X_train, y_train_for_cost)
        opt_result = fmin_l_bfgs_b(cost_function, x0=coordinate_train, args=hyper_parm, fprime=logistic_grad)
        coordinate_train = opt_result[0]
        
        # find best threshold
        threshold_candidate = np.arange(0.01,1,0.01)
        accuracy_candidate = np.zeros(len(threshold_candidate))
        iii = 0
        for thres in threshold_candidate:
            predict_train = predict_values(coordinate_train, norm_X_train, thres)
            accuracy_train = sum(y_train==(predict_train + 1))/len(y_train)
            accuracy_candidate[iii] += accuracy_train
            iii += 1
        threshold = threshold_candidate[accuracy_candidate.argmax()]
        
        # predict
        predict_train = pd.Series(predict_values(coordinate_train, norm_X_train, threshold))
        predict_test = pd.Series(predict_values(coordinate_train, norm_X_test, threshold))
        
        # make confusion matrix
        actual_predicted_train = pd.concat([y_train,(predict_train + y_diff)],axis=1)
        actual_predicted_train.columns = ['Actual class', 'Predicted Class']
        actual_predicted_test = pd.concat([y_test,(predict_test + y_diff)],axis=1)
        actual_predicted_test.columns = ['Actual class', 'Predicted Class']
        confusion_train = pd.crosstab(actual_predicted_train.loc[:,'Actual class'], actual_predicted_train.loc[:,'Predicted Class'])
        confusion_test = pd.crosstab(actual_predicted_test.loc[:,'Actual class'], actual_predicted_test.loc[:,'Predicted Class'])
        
        # calculate model accuracy
        accuracy_train = sum(y_train==(predict_train + y_diff))/len(y_train)
        accuracy_test = sum(y_test==(predict_test + y_diff))/len(y_test)
        
        #prob to print
        prob_print_train = 1 - logit_function(coordinate_train, norm_X_train)
        prob_print_test = 1 - logit_function(coordinate_train, norm_X_test)
        
        # print result
        print_classification_result(y_train=y_train, y_test=y_test, predicted_classes_train=(predict_train + y_diff),
                                    predicted_classes_test=(predict_test + y_diff), accuracy_train=accuracy_train, 
                                    accuracy_test=accuracy_test, confusion_train=confusion_train,
                                    confusion_test=confusion_test, prob_train=prob_print_train, prob_test=prob_print_test, 
                                    best_X_num=None, best_cutting_point=None, root_node_count=None,
                                    predicted_left=None, predicted_right=None, left_node_count=None, 
                                    right_node_count=None, model='Logistic_Regression')
        
        ##############################
        ## end of logistic function ##
        ##############################
        
    def Naive_Bayes(df_train, y_index):
    
        # replace '?' to None
        df_train2 = df_train.copy()
        for col in list(df_train.columns[df_train.isin(['?']).any()]):
            df_train2.loc[df_train2.loc[:,col].isin(['?']),col] = None
        
        # choose y and X
        y_index = int(y_index)
        y_train = df_train2.loc[:,int(y_index)]
        X_train = df_train2.loc[:,df_train.columns != int(y_index)]
        
        # let's find how many classes out there
        classes = np.sort(y_train.unique())
        categoric = [1,2,5,6,8,10,11,12,y_index]
        numeric = [0,3,4,7,9, y_index]
            
        # make two types of dataframe(categorical & numerical)
        df_train_cat = df_train2.loc[:,categoric]
        df_train_num = df_train2.loc[:,numeric]
        
        # calculate difference from 0 to y_classes => for logistic regression
        y_diff = classes[0]
        y_train_for_cost = y_train - y_diff
        
        # calculate total prob
        P_1 = sum(y_train == classes[0]) / len(y_train)
        P_2 = sum(y_train == classes[1]) / len(y_train)
        
        # make empty array to contain conditional probability
        conditional_1 = np.zeros(df_train2.shape)
        conditional_2 = np.zeros(df_train2.shape)
        
        # categorical value
        # replace every elements with probabilities
        # if element is null, replace to 1 (multiply 1 has no effect)
        for col in df_train_cat.columns:
            if col == y_index:
                continue
            cond_prob = (df_train_cat.groupby([y_index,col]).size() / df_train_cat.groupby([y_index])[col].count())
            y_1 = cond_prob[classes[0]]  # conditional class 1
            y_2 = cond_prob[classes[1]]  # conditional class 2
            conditional_1[:,col] = df_train_cat.loc[:,col].map(lambda x: y_1[x] if x != None else 1)
            conditional_2[:,col] = df_train_cat.loc[:,col].map(lambda x: y_2[x] if x != None else 1)
        
        # numerical value
        # calculate conditional mean and variance by groupby
        col_std = df_train_num.groupby(y_index).std()
        col_mean = df_train_num.groupby(y_index).mean()
        for col in df_train_num.columns:
            if col == y_index:
                continue
            mean_1 = col_mean[col][classes[0]]
            mean_2 = col_mean[col][classes[1]]
            std_1 = col_std[col][classes[0]]
            std_2 = col_std[col][classes[1]]
            conditional_1[:,col] = df_train_num.loc[:, col].map(
                lambda x: norm.pdf(x, loc=mean_1, scale=std_1) if x != None else 1)
            conditional_2[:,col] = df_train_num.loc[:, col].map(
                lambda x: norm.pdf(x, loc=mean_2, scale=std_2) if x != None else 1)
        
        # append y probability
        conditional_1[:,y_index] = P_1
        conditional_2[:,y_index] = P_2
        
        # calculate likelihood and probability
        likelihood_1 = conditional_1.prod(axis=1)
        likelihood_2 = conditional_2.prod(axis=1)
        prob_1 = likelihood_1/(likelihood_1 + likelihood_2)
        prob_2 = likelihood_2/(likelihood_1 + likelihood_2)
        prob_print = pd.Series(np.round(prob_1,4))
        
        # predict
        predict_train = (prob_2>0.5).astype(int)
        
        # make confusion matrix
        actual_predicted_train = pd.concat([y_train,pd.Series(predict_train + y_diff)],axis=1)
        actual_predicted_train.columns = ['Actual class', 'Predicted Class']
        confusion_train = pd.crosstab(actual_predicted_train.loc[:,'Actual class'], 
                                      actual_predicted_train.loc[:,'Predicted Class'])
        
        # calculate model accuracy
        accuracy_train = sum(y_train==(predict_train + y_diff))/len(y_train)
        
        # print result
        print_classification_result(y_train=y_train, y_test=None, predicted_classes_train=(predict_train + y_diff),
                                    predicted_classes_test=None, accuracy_train=accuracy_train, 
                                    accuracy_test=None, confusion_train=confusion_train,
                                    confusion_test=None, prob_train=prob_print, prob_test=None, 
                                    best_X_num=None, best_cutting_point=None, root_node_count=None,
                                    predicted_left=None, predicted_right=None, left_node_count=None, 
                                    right_node_count=None, model='Naive_Bayes')
        ##############################
        ##### end of Naive Bayes #####
        ##############################
        
    def Decision_Tree(df_train, df_test, y_train, y_test, y_index):
        '''
        make decision tree by CART,  with one depth
        '''
        
        y_index = int(y_index)

        # let's find how many classes out there
        classes = np.sort(y_train.unique())

        # get n_row for each train and test
        row_len_train = len(y_train)
        row_len_test = len(y_test)

        # get index of X_p from df_train or df_test
        X_index_train = df_train.columns[df_train.columns != int(y_index)]
        X_index_test = df_test.columns[df_test.columns != int(y_index)]


        def full_impurity_calculator(training_df, X_index, y_index, row_len):

            '''
            get training dataframe, column index of X, column index of y as input
            make X_impurity array to save [best cutting point, impurity] of each X_i
            return X_impurity

            logic:
            by each X_i, make df_tmp to get best cutting poing with lowest impurity
            get unique values of X_i, calculate impurity by cutting X_i with unique value
            save impurity of X_i in X_i_impurity
            get best cutting point from X_i_impurity
            save it to X_impurity
            '''

            # make numpy array to save lowest impurity of each X_i(i=1,..,p) 
            # => [cutting point, impurity]
            X_impurity = np.zeros([len(X_index),2])

            for iii, X_ind in enumerate(X_index):
                # get df_tmp which contains original index, X_i_values, y_values
                df_tmp = training_df.sort_values(training_df.columns[X_ind]).loc[:,[X_ind,y_index]].reset_index()

                # get unique values of X_i to find seperating point
                X_i_unique = df_tmp[X_ind].unique()

                # make zero array to save impurity
                X_i_impurity = np.zeros([len(X_i_unique)-1, 2])
                X_i_impurity[:,0] =  X_i_unique[:-1]

                # Separate left and right by unique value
                for ii, x_value in enumerate(X_i_unique[:-1]):
                    tmp_left = df_tmp[df_tmp[X_ind] <= x_value]
                    tmp_right= df_tmp[df_tmp[X_ind] > x_value]

                    tmp_left_impurity = (1 - sum((tmp_left[y_index].value_counts()/len(tmp_left))**2))
                    tmp_right_impurity = (1 - sum((tmp_right[y_index].value_counts()/len(tmp_right))**2))
                    tmp_weighted_impurity = (tmp_left_impurity*(len(tmp_left)/row_len)
                                     + tmp_right_impurity*(len(tmp_right)/row_len))

                    # save impurity for each X_i_unique
                    X_i_impurity[ii, 1] = tmp_weighted_impurity

                # find lowest impurity for X_i
                X_i_best_point = X_i_impurity[np.argmin(X_i_impurity, axis=0)[1],:]

                # save best cutting point, impurity of X_i in X_impurity
                X_impurity[iii,:] = X_i_best_point

            return X_impurity

        def make_prediction(df_to_predict, cutting_point, best_X_num, y_index, y_values):
            '''
            given best X and cutting point, predict y_value
            '''
            # array to save predict result
            predicted_y = np.zeros(len(df_to_predict[y_index]))

            # best X_column to make node
            # we use best_row_num in case column index doesn't match with real X's num
            # ex) X_index = [1,2,3,   5, 6,7]
            best_column_num = X_index_train[best_X_num]

            # make left and right node
            X_values = df_to_predict[best_column_num]
            left_node = y_values[X_values<=cutting_point]
            right_node = y_values[X_values>cutting_point]

            # predict y and save it to predicted_y
            predicted_left = left_node.value_counts().argmax()
            predicted_right = right_node.value_counts().argmax()
            predicted_y[X_values<=cutting_point] = predicted_left
            predicted_y[X_values>cutting_point] = predicted_right

            return predicted_y, left_node, right_node, predicted_left, predicted_right

        # find best X to split
        full_impurity = full_impurity_calculator(df_train, X_index_train, y_index, row_len_train)
        best_X_num = np.argmin(full_impurity, axis=0)[1]
        best_cutting_point = full_impurity[best_X_num][0]
        
        # run make_prediction_function
        train_result = make_prediction(df_train, best_cutting_point, best_X_num, y_index, y_train)
        test_result = make_prediction(df_test, best_cutting_point, best_X_num, y_index, y_test)

        # make a prediction
        predict_train = pd.Series(train_result[0])
        predict_test = pd.Series(test_result[0])

        # count values from each node
        root_node_count = tuple(y_train.value_counts().sort_index())
        left_node_count = tuple(train_result[1].value_counts().sort_index())
        right_node_count = tuple(train_result[2].value_counts().sort_index())

        # predicted class left
        predicted_left = train_result[3]
        predicted_right = train_result[4]

        # make confusion matrix
        actual_predicted_train = pd.concat([y_train,predict_train],axis=1)
        actual_predicted_train.columns = ['Actual class', 'Predicted Class']
        actual_predicted_test = pd.concat([y_test,predict_test],axis=1)
        actual_predicted_test.columns = ['Actual class', 'Predicted Class']
        confusion_train = pd.crosstab(actual_predicted_train.loc[:,'Actual class'], actual_predicted_train.loc[:,'Predicted Class'])
        confusion_test = pd.crosstab(actual_predicted_test.loc[:,'Actual class'], actual_predicted_test.loc[:,'Predicted Class'])

        # calculate model accuracy
        accuracy_train = sum(y_train==predict_train)/len(y_train)
        accuracy_test = sum(y_test==predict_test)/len(y_test)

        # print result
        print_classification_result(y_train=y_train, y_test=y_test, predicted_classes_train=predict_train,
                                        predicted_classes_test=predict_test , accuracy_train=accuracy_train, 
                                        accuracy_test=accuracy_test, confusion_train=confusion_train,
                                        confusion_test=confusion_test, prob_train=None,prob_test=None, 
                                        best_X_num=best_X_num, best_cutting_point=best_cutting_point, 
                                        root_node_count=root_node_count,predicted_left=predicted_left, 
                                       predicted_right=predicted_right, left_node_count=left_node_count, 
                                        right_node_count=right_node_count, model='Decision_Tree')

        #########################################
        ##### end of Decision Tree function #####
        #########################################

    def Bagging_LDA(df_train, df_test,X_train,X_test, y_train, y_test, y_index):
        def LDA_simple(X_train,y_train,X_test,y_test):
            
            # get variables to calculate LDF
            classes = np.sort(y_train.unique())
            X_bar_by_class = dict()
            n_class = dict()  # number of samples by classes(안씀)
            S_by_class = dict() # sample covariance by classes(안씀)
            equation_for_S_p = np.zeros((len(X_train.columns),len(X_train.columns)))
            for class_num in classes:
                X_k = X_train.loc[y_train==class_num,:]
                X_bar_by_class['X_bar_{}'.format(class_num)] = X_k.mean() # 독립변수의 표본평균 벡터
                n_class['n_{}'.format(class_num)] = len(X_k)  # 클래스에 해당하는 샘플의 갯수
                S_by_class['S_{}'.format(class_num)] = X_k.cov()  # 클래스에 해당하는 샘플의 독립변수별 표본분산
                equation_for_S_p += np.array(X_k.cov() * (len(X_k)-1))  # pooled covariance 편하게 계산 위해 미리 만들어둠


            # calculate S_p (diagonal matrix)
            S_p = equation_for_S_p/(len(X_train)-len(classes))

            # calculate LDF for train data
            fianl_d_train = np.zeros((len(X_train),len(classes)))
            for class_num in classes:
                X_name = 'X_bar_{}'.format(class_num)
                LDF_1_train = np.matmul(np.matmul(X_bar_by_class[X_name],np.linalg.inv(S_p)), X_train.transpose())
                LDF_2_train = np.matmul(np.matmul(X_bar_by_class[X_name],np.linalg.inv(S_p)) ,X_bar_by_class[X_name])/2
                d_k_train = LDF_1_train - LDF_2_train
                fianl_d_train[:,(class_num-1)] += d_k_train

            # get predicted classes for train data => caution : column starts from 0, while class starts from 1
            final_d_dataframe_train = pd.DataFrame(fianl_d_train)
            final_d_dataframe_train.columns = classes
            predicted_classes_train = final_d_dataframe_train.idxmax(axis=1)

            ##### test data ######
            # calculate LDF for test data
            fianl_d_test = np.zeros((len(X_test),len(classes)))
            for class_num in classes:
                X_name = 'X_bar_{}'.format(class_num)
                LDF_1_test = np.matmul(np.matmul(X_bar_by_class[X_name],np.linalg.inv(S_p)), X_test.transpose())
                LDF_2_test = np.matmul(np.matmul(X_bar_by_class[X_name],np.linalg.inv(S_p)) ,X_bar_by_class[X_name])/2
                d_k_test = LDF_1_test - LDF_2_test
                fianl_d_test[:,(class_num-1)] += d_k_test

            # get predicted classes for test data => caution : column starts from 0, while class starts from 1
            final_d_dataframe_test = pd.DataFrame(fianl_d_test)
            final_d_dataframe_test.columns = classes
            predicted_classes_test = final_d_dataframe_test.idxmax(axis=1)

            return predicted_classes_test
            #########################
            ## end of LDA function ##
            #########################
            
            
        # make zeor array to save 51 bootstrap predict result
        ensemble_result = np.zeros((df_test.shape[0], 51))
        
        # resample and train LDA => save 51 predicted test value in ensemble_result array
        for i in range(0,51):
            df_bootstrap = df_train.sample(n=df_train.shape[0], replace=True).reset_index(drop=True)
            y_bootstrap = df_bootstrap.loc[:,int(y_index)]
            X_bootstrap = df_bootstrap.loc[:,df_bootstrap.columns != int(y_index)]

            LDA_predict = LDA_simple(X_bootstrap,y_bootstrap,X_test,y_test)

            ensemble_result[:,i] = LDA_predict

        # vote and get most predicted value by row
        predicted_classes_bagging = pd.Series(mode(ensemble_result, axis=1)[0].transpose()[0])

        # get no bagging result
        predicted_classes_no_bagging = pd.Series(LDA_simple(X_train,y_train,X_test,y_test))
        
        # make confusion matrix
        actual_predicted_no_bagging = pd.concat([y_test,predicted_classes_no_bagging],axis=1)
        actual_predicted_no_bagging.columns = ['Actual class', 'Predicted Class']
        actual_predicted_bagging = pd.concat([y_test,predicted_classes_bagging],axis=1)
        actual_predicted_bagging.columns = ['Actual class', 'Predicted Class']
        confusion_no_bagging = actual_predicted_no_bagging.groupby(['Actual class', 'Predicted Class']).size().unstack().fillna(0)
        confusion_bagging = actual_predicted_bagging.groupby(['Actual class', 'Predicted Class']).size().unstack().fillna(0)

        # calculate model accuracy
        accuracy_no_bagging = sum(y_test==predicted_classes_no_bagging)/len(y_test)
        accuracy_bagging = sum(y_test==predicted_classes_bagging)/len(y_test)

        # print result
        print_classification_result(y_train=y_test, y_test=y_test,predicted_classes_train=predicted_classes_no_bagging,
                                    predicted_classes_test=predicted_classes_bagging, accuracy_train=accuracy_no_bagging, 
                                    accuracy_test=accuracy_bagging,confusion_train=confusion_no_bagging,
                                    confusion_test=confusion_bagging,prob_train=None, prob_test=None, 
                                    best_X_num=None, best_cutting_point=None, root_node_count=None,
                                    predicted_left=None, predicted_right=None, left_node_count=None, 
                                    right_node_count=None, model='Bagging_LDA')

        
    if classification_model == '1':
        LDA(X_train,y_train,X_test,y_test)
    elif classification_model == '2':
        QDA(X_train,y_train,X_test,y_test)
    elif classification_model == '3':
        RDA(X_train,y_train,X_test,y_test)
    elif classification_model == '4':
        Logistic_regression(X_train,y_train,X_test,y_test)
    elif classification_model == '5':
        Naive_Bayes(df_train, y_index)
    elif classification_model == '6':
        Decision_Tree(df_train, df_test, y_train, y_test, y_index)
    elif classification_model == '7':
        Bagging_LDA(df_train, df_test,X_train,X_test, y_train, y_test, y_index)
    else:
        raise('You should enter only 1, 2, 3, 4, 5, 6, 7')
    ##########################################
    ##### end of Classification function #####
    ##########################################

if __name__ == "__main__":
    user_want = input(
        '''Which analysis do you wish to carry out? 
        1= regression, 2 = classification : ''')
    if user_want == '1':
        regression_function()
    elif user_want == '2':
        classification_function()
    else:
        raise('you should enter only 1 or 2')