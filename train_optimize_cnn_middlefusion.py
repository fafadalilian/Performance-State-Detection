
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 6 11:58:34 2023

Author: Fatemeh Dalilian
"""

from tensorflow import keras
from tensorflow.keras.models import Model
from keras.layers import Dense, Dropout, BatchNormalization, Conv2D, MaxPooling2D, Flatten, Input, concatenate
from keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
import scipy.io
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score, recall_score, accuracy_score, confusion_matrix, cohen_kappa_score, matthews_corrcoef, roc_auc_score, precision_score
from sklearn.utils import resample
import optuna
import pandas as pd
import tensorflow as tf
from create_stack_separate import give_tf

# Set random seed for reproducibility
np.random.seed(2)

# Define constants
overlap_percent = 0.2
window_step_length = 0.25
path1 = "/Users/fdalilian/argpn1112024"

# Load the MATLAB file
EEG_data = scipy.io.loadmat(path1 + '/Matlab/Mat/EEG_timeseries12.mat')['EEG_timeseries']
HR_data = scipy.io.loadmat(path1 + '/Matlab/Mat/HR_timeseries12.mat')['HR_timeseries']
ET_data = scipy.io.loadmat(path1 + '/Matlab/Mat/ET_timeseries12.mat')['ET_timeseries'][:, :, [0, 2, 3, 4]]
controller_data = scipy.io.loadmat(path1 + '/Matlab/Mat/controller_timeseries20srate.mat')['controller_timeseries']
Label = scipy.io.loadmat(path1 + '/Matlab/Mat/Label_timeseries12.mat')['Label']
participants = scipy.io.loadmat(path1 + '/Matlab/Mat/participant_timeseries12.mat')['participant']
condition = scipy.io.loadmat(path1 + '/Matlab/Mat/condition_timeseries12.mat')['condition']

train_val_y=Label

def normalize_data_by_frequency_bins_and_channels(data):
    """
    Normalize data by frequency bins and channels.

    Args:
        data (ndarray): Input data array.

    Returns:
        ndarray: Normalized data array.
    """
    feature_means = np.mean(data, axis=(0, 2))
    feature_stds = np.std(data, axis=(0, 2))
    normalized_data = np.copy(data)

    num_samples, frequency_bins, time_steps, num_channels = data.shape
    for f in range(frequency_bins):
        for c in range(num_channels):
            normalized_data[:, f, :, c] = (data[:, f, :, c] - feature_means[f, c]) / feature_stds[f, c]

    return normalized_data
#%%

def create_model(trial):
    """
    Create a CNN model with hyperparameter optimization using Optuna.

    Args:
        trial (optuna.trial.Trial): Optuna trial object.

    Returns:
        model: Compiled Keras model.
        train_val_EEG, train_val_ET, train_val_cont: Training and validation data for EEG, ET, and controller.
        learning_rate: Learning rate for the optimizer.
    """
    data_length = trial.suggest_int('data_length', 2, 10)
    num_layers_dense = trial.suggest_int('num_layers_dense', 1, 4)
    learning_rate = trial.suggest_categorical('learning_rate', [0.0001, 0.0005, 0.001, 0.01])

    normalized_EEG, ET_spectrograms, controller_spectrograms = give_tf(overlap_percent, window_step_length, data_length, EEG_data, HR_data, ET_data, controller_data, participants)
    n_normalized_EEG = normalize_data_by_frequency_bins_and_channels(normalized_EEG)
    n_ET_spectrograms = normalize_data_by_frequency_bins_and_channels(ET_spectrograms)
    n_controller_spectrograms = normalize_data_by_frequency_bins_and_channels(controller_spectrograms)
    
    train_val_EEG = n_normalized_EEG.transpose(0, 2, 1, 3)
    train_val_ET = n_ET_spectrograms.transpose(0, 2, 1, 3)
    train_val_cont = n_controller_spectrograms.transpose(0, 2, 1, 3)
    
    main_input_EEG = Input(shape=train_val_EEG.shape[1:], name='main_input_EEG')
    main_input_ET = Input(shape=train_val_ET.shape[1:], name='main_input_ET')
    main_input_cont = Input(shape=train_val_cont.shape[1:], name='main_input_cont')
    
    input_layers = [main_input_EEG, main_input_ET, main_input_cont]
    cnn_outputs = []
    
    for input_layer in input_layers:
        x = input_layer
        current_shape = input_layer.shape.as_list()[1:]
        num_cnn_layers = trial.suggest_int(f'num_cnn_layers_{input_layer.name}', 1, 4)
        
        for cnn_index in range(num_cnn_layers):
            filters = trial.suggest_categorical(f'filters_layer_{input_layer.name}_{cnn_index}', [8, 16, 64, 128])
            kernel_height = trial.suggest_int(f'kernel_height_layer_{input_layer.name}_{cnn_index}', 1, min(5, current_shape[0] - 1))
            kernel_width = trial.suggest_int(f'kernel_width_layer_{input_layer.name}_{cnn_index}', 1, min(5, current_shape[1] - 1))
            pool_height = trial.suggest_int(f'pool_height_layer_{input_layer.name}_{cnn_index}', 1, min(3, current_shape[0] // 2))
            pool_width = trial.suggest_int(f'pool_width_layer_{input_layer.name}_{cnn_index}', 1, min(3, current_shape[1] // 2))
            dropout_rate_conv = trial.suggest_categorical(f'dropout_rate_convlayer_{input_layer.name}_{cnn_index}', [0, 0.1, 0.2, 0.3, 0.4])
            l2_regularizer_strength_conv = trial.suggest_categorical(f'l2_regularizer_strength_convlayer_{input_layer.name}_{cnn_index}', [0, 0.0001, 0.001, 0.01, 0.1])
    
            x = Conv2D(filters=filters, kernel_size=(kernel_height, kernel_width), activation='relu', padding='same', kernel_regularizer=l2(l2_regularizer_strength_conv))(x)
            x = BatchNormalization()(x)
            x = MaxPooling2D(pool_size=(pool_height, pool_width))(x)
            x = Dropout(dropout_rate_conv)(x)
            current_shape = (current_shape[0] // pool_height, current_shape[1] // pool_width, filters)
    
        x = Flatten()(x)
        cnn_outputs.append(x)
    
    binary_input = Input(shape=(1,), name='binary_input')
    combined = concatenate(cnn_outputs + [binary_input])

    for dense_index in range(num_layers_dense):
        dense_nodes = trial.suggest_categorical(f'dense_node_layer{dense_index}', [16, 64, 128, 256])
        dropout_rate_dense = trial.suggest_categorical(f'dropout_rate_denselayer{dense_index}', [0, 0.1, 0.3, 0.4])
        l2_regularizer_strength_dense = trial.suggest_categorical(f'l2_regularizer_strength_dense{dense_index}', [0, 0.0001, 0.001, 0.01, 0.1])
        combined = Dense(dense_nodes, activation='relu', kernel_regularizer=l2(l2_regularizer_strength_dense))(combined)
        combined = Dropout(dropout_rate_dense)(combined)

    output = Dense(1, activation='sigmoid')(combined)
    model = Model(inputs=[input_layers, binary_input], outputs=output)

    return model, train_val_EEG, train_val_ET, train_val_cont, learning_rate

#%%
def objective(trial):
    """
    Objective function for Optuna hyperparameter optimization.

    Args:
        trial (optuna.trial.Trial): Optuna trial object.

    Returns:
        float: Average Matthews correlation coefficient (MCC).
    """
    skf = StratifiedGroupKFold(n_splits=5)
    accuracy_scores, f1_scores, f11_scores, f10_scores, recall0_scores, recall1_scores = [], [], [], [], [], []
    mcc_scores, cohen_kappa_scores, roc_auc_scores = [], [], []
    precisions0_scores, precisions1_scores = [], []
    tp_scores, fp_scores, tn_scores, fn_scores = [], [], [], []

    model, train_val_EEG, train_val_ET, train_val_cont, learning_rate = create_model(trial)

    for train_index, val_index in skf.split(train_val_EEG, train_val_y, participants):
        clone_model = tf.keras.models.clone_model(model)
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        clone_model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['Recall'])
        
        EEG_train, EEG_val = train_val_EEG[train_index], train_val_EEG[val_index]
        ET_train, ET_val = train_val_ET[train_index], train_val_ET[val_index]
        cont_train, cont_val = train_val_cont[train_index], train_val_cont[val_index]
        y_train, y_val = train_val_y[train_index], train_val_y[val_index]
        condition_train, condition_val = condition[train_index], condition[val_index]
        
        EEG_majority, EEG_minority = EEG_train[y_train == 0], EEG_train[y_train == 1]
        ET_majority, ET_minority = ET_train[y_train == 0], ET_train[y_train == 1]
        cont_majority, cont_minority = cont_train[y_train == 0], cont_train[y_train == 1]
        y_majority, y_minority = y_train[y_train == 0], y_train[y_train == 1]
        condition_majority, condition_minority = condition_train[y_train == 0], condition_train[y_train == 1]
        
        EEG_minority_upsampled, ET_minority_upsampled, cont_minority_upsampled, condition_minority_upsampled, y_minority_upsampled = resample(
            EEG_minority, ET_minority, cont_minority, condition_minority, y_minority,
            replace=True, n_samples=EEG_majority.shape[0], random_state=2
        )

        EEG_train_upsampled = np.vstack((EEG_majority, EEG_minority_upsampled))
        ET_train_upsampled = np.vstack((ET_majority, ET_minority_upsampled))
        cont_train_upsampled = np.vstack((cont_majority, cont_minority_upsampled))
        y_train_upsampled = np.concatenate((y_majority, y_minority_upsampled))
        condition_train_upsampled = np.concatenate((condition_majority, condition_minority_upsampled))
        
        early_stopping = EarlyStopping(patience=3, monitor='val_loss', mode='min')

        clone_model.fit([EEG_train_upsampled, ET_train_upsampled, cont_train_upsampled, condition_train_upsampled], y_train_upsampled, batch_size=400, epochs=100, validation_data=([EEG_val, ET_val, cont_val, condition_val], y_val), callbacks=[early_stopping])

        y_pred = (clone_model.predict([EEG_val, ET_val, cont_val, condition_val]) > 0.499).astype(np.uint8)

        accuracy_scores.append(accuracy_score(y_val, y_pred))
        f1_scores.append(f1_score(y_val, y_pred))
        f11_scores.append(f1_score(y_val, y_pred, pos_label=1))
        f10_scores.append(f1_score(y_val, y_pred, pos_label=0))
        recall0_scores.append(recall_score(y_val, y_pred, pos_label=0))
        recall1_scores.append(recall_score(y_val, y_pred, pos_label=1))
        mcc_scores.append(matthews_corrcoef(y_val, y_pred))
        cohen_kappa_scores.append(cohen_kappa_score(y_val, y_pred))
        roc_auc_scores.append(roc_auc_score(y_val, y_pred))
        precisions0_scores.append(precision_score(y_val, y_pred, pos_label=0))
        precisions1_scores.append(precision_score(y_val, y_pred, pos_label=1))
        
        cm = confusion_matrix(y_val, y_pred)
        TN, FP, FN, TP = cm.ravel()
        tp_scores.append(TP)
        fp_scores.append(FP)
        tn_scores.append(TN)
        fn_scores.append(FN)


    avg_f1_score, std_f1 = np.mean(f1_scores), np.std(f1_scores)
    avg_accuracy, std_accuracy = np.mean(accuracy_scores), np.std(accuracy_scores)
    avg_f11, std_f11 = np.mean(f11_scores), np.std(f11_scores)
    avg_f10, std_f10 = np.mean(f10_scores), np.std(f10_scores)
    avg_recall0, std_recall0 = np.mean(recall0_scores), np.std(recall0_scores)
    avg_recall1, std_recall1 = np.mean(recall1_scores), np.std(recall1_scores)
    avg_mcc, std_mcc = np.mean(mcc_scores), np.std(mcc_scores)
    avg_cohen_kappa, std_cohen_kappa = np.mean(cohen_kappa_scores), np.std(cohen_kappa_scores)
    avg_roc_auc, std_roc_auc = np.mean(roc_auc_scores), np.std(roc_auc_scores)
    avg_precisions0, std_precisions0 = np.mean(precisions0_scores), np.std(precisions0_scores)
    avg_precisions1, std_precisions1 = np.mean(precisions1_scores), np.std(precisions1_scores)

    trial.set_user_attr('avg_f1_score', avg_f1_score)
    trial.set_user_attr('std_f1', std_f1)
    trial.set_user_attr('avg_accuracy', avg_accuracy)
    trial.set_user_attr('std_accuracy', std_accuracy)
    trial.set_user_attr('avg_f11', avg_f11)
    trial.set_user_attr('std_f11', std_f11)
    trial.set_user_attr('avg_f10', avg_f10)
    trial.set_user_attr('std_f10', std_f10)
    trial.set_user_attr('avg_recall0', avg_recall0)
    trial.set_user_attr('std_recall0', std_recall0)
    trial.set_user_attr('avg_recall1', avg_recall1)
    trial.set_user_attr('std_recall1', std_recall1)
    trial.set_user_attr('avg_mcc', avg_mcc)
    trial.set_user_attr('std_mcc', std_mcc)
    trial.set_user_attr('avg_cohen_kappa', avg_cohen_kappa)
    trial.set_user_attr('std_cohen_kappa', std_cohen_kappa)
    trial.set_user_attr('avg_roc_auc', avg_roc_auc)
    trial.set_user_attr('std_roc_auc', std_roc_auc)
    trial.set_user_attr('avg_precisions0', avg_precisions0)
    trial.set_user_attr('std_precisions0', std_precisions0)
    trial.set_user_attr('avg_precisions1', avg_precisions1)
    trial.set_user_attr('std_precisions1', std_precisions1)
    trial.set_user_attr('avg_TP', np.mean(tp_scores))
    trial.set_user_attr('std_TP', np.std(tp_scores))
    trial.set_user_attr('avg_FP', np.mean(fp_scores))
    trial.set_user_attr('std_FP', np.std(fp_scores))
    trial.set_user_attr('avg_TN', np.mean(tn_scores))
    trial.set_user_attr('std_TN', np.std(tn_scores))
    trial.set_user_attr('avg_FN', np.mean(fn_scores))
    trial.set_user_attr('std_FN', np.std(fn_scores))

    trial_data = {
        'trial_number': trial.number,
        'avg_f1_score': avg_f1_score,
        'std_f1': std_f1,
        'avg_accuracy': avg_accuracy,
        'std_accuracy': std_accuracy,
        'avg_f11': avg_f11,
        'std_f11': std_f11,
        'avg_f10': avg_f10,
        'std_f10': std_f10,
        'avg_recall0': avg_recall0,
        'std_recall0': std_recall0,
        'avg_recall1': avg_recall1,
        'std_recall1': std_recall1,
        'avg_mcc': avg_mcc,
        'std_mcc': std_mcc,
        'avg_cohen_kappa': avg_cohen_kappa,
        'std_cohen_kappa': std_cohen_kappa,
        'avg_roc_auc': avg_roc_auc,
        'std_roc_auc': std_roc_auc,
        'avg_precisions0': avg_precisions0,
        'std_precisions0': std_precisions0,
        'avg_precisions1': avg_precisions1,
        'std_precisions1': std_precisions1,
        'avg_TP': np.mean(tp_scores),
        'std_TP': np.std(tp_scores),
        'avg_FP': np.mean(fp_scores),
        'std_FP': np.std(fp_scores),
        'avg_TN': np.mean(tn_scores),
        'std_TN': np.std(tn_scores),
        'avg_FN': np.mean(fn_scores),
        'std_FN': np.std(fn_scores)
    }

    trial_data.update(trial.params)
    trial_data_list.append(trial_data)

    trial_df = pd.DataFrame(trial_data_list)
    trial_df.to_excel(path1 + '/separate_CNN_middle_fusion_results.xlsx', index=False)

    return avg_mcc

# Initialize an empty list to store trial data
trial_data_list = []
# Run Optuna optimization
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=1000)

# Print the best parameters and best score
print("Best Average MCC Score:", study.best_value)
print("Best Parameters:", study.best_params)

# Save the best parameters and best average MCC score to a file
with open(path1 + "separate_CNN_middle_fusion_best_results.txt", "w") as file:
    file.write("Best Average MCC Score: " + str(study.best_value) + "\n")
    file.write("Best Parameters: " + str(study.best_params) + "\n")
