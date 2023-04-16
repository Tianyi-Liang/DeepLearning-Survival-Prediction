from flask import Flask, render_template, url_for, request
import numpy as np
from pycox.preprocessing.feature_transforms import OrderedCategoricalLong
import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pycox.models import LogisticHazard
from sklearn_pandas import DataFrameMapper
from sklearn.preprocessing import StandardScaler
import torch
import torchtuples as tt
import math
import glob
import random

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=["POST"])
def Survival_analyze():

    ana_method = request.form['ana_sele']

    # get the exact values from the webpage!
    if ana_method == 'CSS':
        Age = request.form['age']
        BMI = request.form['BMI']
        Alb = request.form['alb']
        Alt = request.form['alt']
        Ast = request.form['ast']
        GGT = request.form['ggt']
        TB = request.form['tb']
        Hbv_dna = request.form['hbv-dna']
        Neu = request.form['neu']
        Plt = request.form['plt']
        Lym = request.form['lym']
        AFP = request.form['afp']
        Tumor_size = request.form['tumor-size']
        Operation_time = request.form['operation-time']
        Blood_loss = request.form['blood-loss']

        Gender = request.form['gender']
        T2DM = request.form['t2dm']
        Cirrhosis = request.form['cirrhosis']
        Child_Pugh = request.form['child-pugh']
        Smoking = request.form['smoking']
        Alcohol = request.form['alcohol']
        Enhanced_tace = request.form['enhanced-tace']
        TACE = request.form['tace']
        Re_resection = request.form['re-resection']
        RFA = request.form['RFA']
        Hypertension = request.form['hypertension']
        Tumor_number = request.form['tumor-number']
        Macroinvasion = request.form['macroinvasion']
        MVI = request.form['mvi']
        Pathological_grade = request.form['pathological_grade']

        # transform the raw value to the right form
        NLR = float(Neu) / float(Lym)
        PLR = float(Plt) / float(Lym)

        Operation_time = float(Operation_time)/10
        Blood_loss = float(Blood_loss)/100

        sample_data = [Age, BMI, Alb, Alt, Ast, GGT, TB, Hbv_dna, NLR, PLR,
                       AFP, Tumor_size, Blood_loss, Operation_time, Child_Pugh, Gender, T2DM, Cirrhosis, Smoking, Alcohol,
                       Enhanced_tace, Hypertension, Tumor_number, Macroinvasion, MVI, Pathological_grade, TACE, Re_resection, RFA]

        # transform to float
        convert_data = [float(i) for i in sample_data]

        # transform hbv_dna to logarithmic form
        convert_data[7] = math.log10(convert_data[7])

        # making the input data a matrix
        pred = pd.DataFrame(convert_data)
        pred = pred.T

        pred.columns = ['Age', 'BMI', 'Alb', 'Alt', 'Ast', 'GGT', 'TB', 'HBV_DNA', 'NLR', 'PLR',
                        'AFP', 'Tumor_size', 'Blood_loss', 'Operation_time', 'Child_Pugh', 'Gender', 'T2DM', 'Cirrhosis', 'Smoking', 'Alcohol',
                        'Enhance_TACE', 'Hypertension', 'Tumor_number', 'Macroinvasion', 'MVI', 'Pathological_grade', 'TACE', 'Re_resection', 'RFA']

        # transform to float32 (perhaps this is an unnecessary processing, or it may be necessary for neural networks, and we have not further explored it)
        pred = pred.astype('float32')

        # we will use the training set to further transform the input data
        df_train = pd.read_csv('static//train.csv')

        # making a split of the parameters
        cols_standardize = ['Age', 'BMI', 'Alb', 'Alt', 'Ast', 'GGT', 'TB', 'HBV_DNA', 'NLR', 'PLR', 'AFP', 'Tumor_size', 'Operation_time', 'Blood_loss']
        cols_leave = ['Gender', 'T2DM', 'Cirrhosis', 'Child_Pugh', 'Smoking', 'Alcohol', 'Enhance_TACE', 'TACE', 'Re_resection', 'RFA', 'Hypertension', 'Tumor_number', 'Macroinvasion', 'MVI']
        cols_categorical = ['Pathological_grade']

        standardize = [([col], StandardScaler()) for col in cols_standardize]
        leave = [(col, None) for col in cols_leave]
        categorical = [(col, OrderedCategoricalLong()) for col in cols_categorical]

        x_mapper_float = DataFrameMapper(standardize+ leave)
        x_mapper_long = DataFrameMapper(categorical)  # we need a separate mapper to ensure the data type 'int64'

        x_fit_transform = lambda df: tt.tuplefy(x_mapper_float.fit_transform(df), x_mapper_long.fit_transform(df))
        x_transform = lambda df: tt.tuplefy(x_mapper_float.transform(df), x_mapper_long.transform(df))

        x_train = x_fit_transform(df_train)
        x_pred = x_transform(pred)

        num_durations = 20
        scheme = 'quantiles'
        labtrans = LogisticHazard.label_transform(num_durations, scheme)

        get_target = lambda df: (df['duration_css'].values, df['event_css'].values)
        y_train = labtrans.fit_transform(*get_target(df_train))

        num_embeddings = x_train[1].max(0) + 1
        embedding_dims = num_embeddings // 2

        # design a same neural network structure, without dropout
        in_features = x_train[0].shape[1]
        out_features = labtrans.out_features
        num_nodes = [16, 16, 16, 16]

        net = tt.practical.MixedInputMLP(in_features, num_embeddings, embedding_dims, num_nodes, out_features)
        model = LogisticHazard(net, duration_index=labtrans.cuts)

        # load the trained model's weights, and to make a prediction on the new data
        model.load_model_weights('static/N-net-CSS.pth')

        # predicting on the input data
        surv = model.predict_surv_df(x_pred)

        # save the survival plot
        fig = plt.figure(figsize=(9, 6))
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(surv.index, surv[0], 'o-', color='#F00078', label='DeepSurv')
        ax.set_xlim(None, 95)

        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        # ax.set_title('Cancer Special Survival Prediction')
        plt.title('Cancer Special Survival Prediction', fontsize=15, color='Maroon', fontweight='bold', loc='center')
        plt.xlabel('Time After Hepatectomy (months)', fontsize=10, color='#3C3C3C', fontweight='bold', loc='center', labelpad=5, alpha=1)
        plt.ylabel('Predicted Survival probability ', fontsize=10, color='#3C3C3C', fontweight='bold', loc='center', labelpad=5, alpha=1)

        plt.axhline(y=0.5, c="#FF8040", linestyle='dashed')

        fig_path_li = glob.glob('static//CSS*.jpg')


        if not fig_path_li:
            plt.savefig('static//CSS{}.jpg'.format(random.randint(1, 999)), bbox_inches='tight', dpi=300)
        else:
            os.remove(fig_path_li[0])
            plt.savefig('static//CSS{}.jpg'.format(random.randint(1, 999)), bbox_inches='tight', dpi=300)

        fig_path = glob.glob('static//CSS*.jpg')[0]

        return render_template('predict.html', explain="Model was based on a cohort with 600 cases from The First Affiliated Hospital of Guangxi Medical University", url=fig_path)
    else:
        pass

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=8090)