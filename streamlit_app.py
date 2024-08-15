
import os
import tensorflow

from ai4water.backend import sklearn_models

from ngboost import NGBRegressor
sklearn_models['NGBRegressor'] = NGBRegressor


import joblib
from typing import Tuple

import xgboost as xgb
import streamlit as st
import pandas as pd
import numpy as np

from ai4water import Model
from ai4water.utils import TrainTestSplit
from tensorflow.keras.layers import Dense, Dropout, Input, Concatenate, BatchNormalization
from tensorflow.keras import Model as tens_Model
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.math import reduce_mean, log, square
import tensorflow_probability as tfp
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

from xgboostlss.model import XGBoostLSS

# %%
def _ohe_column(df:pd.DataFrame, col_name:str)->tuple:
    # function for OHE
    assert isinstance(col_name, str)

    # setting sparse to True will return a scipy.sparse.csr.csr_matrix
    # not a numpy array
    encoder = OneHotEncoder(sparse=False)
    ohe_cat = encoder.fit_transform(df[col_name].values.reshape(-1, 1))
    cols_added = [f"{col_name}_{i}" for i in range(ohe_cat.shape[-1])]

    df[cols_added] = ohe_cat

    df.pop(col_name)

    return df, cols_added, encoder

def le_column(df:pd.DataFrame, col_name)->tuple:
    """label encode a column in dataframe"""
    encoder = LabelEncoder()
    df[col_name] = encoder.fit_transform(df[col_name])
    return df, encoder

# %%

def _load_data(
        input_features:list=None,
)->Tuple[pd.DataFrame, pd.DataFrame, list]:

    default_input_features = ['Adsorbent', 'Feedstock', 'Pyrolysis_temp',
                              'Heating rate (oC)', 'Pyrolysis_time (min)',
                              'C', 'O', 'Surface area', 'Adsorption_time (min)',
                              'Ci_ppm', 'solution pH', 'rpm', 'Volume (L)',
                              'loading (g)', 'adsorption_temp',
                              'Ion Concentration (mM)', 'ion_type']

    if input_features is None:
        input_features = default_input_features

    # read excel
    # our data is on the first sheet
    dirname = os.path.dirname(__file__)
    org_data = pd.read_excel(os.path.join(dirname, 'master_sheet_0802.xlsx'))
    data = org_data
    data.dropna()

    # removing final concentration and rpm from our data. As rpm only
    # contains one unique value and final concentration is used for
    # calculating the true value of target.

    #data = data.drop(columns=['Cf', 'rpm'])

    #removing original index of both dataframes and assigning a new index
    data = data.reset_index(drop=True)

    target = ['qe']

    if input_features is None:
        input_features = data.columns.tolist()[0:-1]
    else:
        assert isinstance(input_features, list)
        assert all([feature in data.columns for feature in input_features])

    data = data[input_features + target]
    data = data.dropna()

    data['Feedstock'] = data['Feedstock'].replace('coagulation–flocculation sludge',
                                                  'CF Sludge')

    data['Feedstock'] = data['Feedstock'].replace('bamboo (Phyllostachys pubescens)',
                                                  'bamboo (PP)')

    return org_data, data, input_features

def make_data(
        input_features:list = None,
        encoding:str = None,
)->Tuple[pd.DataFrame, list, dict]:

    _, data, input_features = _load_data(input_features)

    adsorbent_encoder, fs_encoder, it_encoder  = None, None, None
    if encoding=="ohe":
        # applying One Hot Encoding
        data, _, adsorbent_encoder = _ohe_column(data, 'Adsorbent')
        data, _, fs_encoder = _ohe_column(data, 'Feedstock')
        data, _, it_encoder = _ohe_column(data, 'ion_type')

    elif encoding == "le":
        # applying Label Encoding
        data, adsorbent_encoder = le_column(data, 'Adsorbent')
        data, fs_encoder = le_column(data, 'Feedstock')
        data, it_encoder = le_column(data, 'ion_type')

    # moving target to last
    target = data.pop('qe')
    data['qe'] = target

    encoders = {
        "Adsorbent": adsorbent_encoder,
        "Feedstock": fs_encoder,
        "ion_type": it_encoder
    }
    return data, input_features, encoders

# %%

def build_monte_carlo_model():

    inp = Input(shape= (17,) )
    d1 = Dense(32)(inp)
    drop1 = Dropout(0.3)(d1)
    d2 = Dense(32)(drop1)
    drop2 = Dropout(0.3)(d2)
    d3 = Dense(32, activation="relu")(drop2)
    drop3 = Dropout(0.3)(d3)
    out = Dense(1)(drop3)

    model = tens_Model(inp, out)
    model.compile(loss="mse", optimizer=Adam(lr=0.0001))
    return model

def build_bayesian_nn_model():

    # Define the negative log likelihood loss function
    def negative_loglikelihood(targets, estimated_distribution):
        return -estimated_distribution.log_prob(targets)

    # Define the model architecture
    hidden_units= [19, 19]
    inputs = Input(shape=(17,), name='Inputs')

    features = BatchNormalization()(inputs)
    for units in hidden_units:
        features = Dense(units, activation='sigmoid')(features)

    distribution_params = Dense(units=2)(features)
    outputs = tfp.layers.IndependentNormal(1)(distribution_params)

    model = tens_Model(inputs=inputs, outputs=outputs)

    # Compile the model
    model.compile(optimizer= RMSprop(learning_rate=0.002850512),
                  loss= negative_loglikelihood)

    return model

class ModelHandler(object):

    weights = {
        'NGBoost': 'NGBRegressor',
        'Monte Carlo Dropout': 'weights',
        'Probabilistic_Neural_Network': 'weights'
    }

    def __init__(self, model_name):
        self.name = model_name
        self.model = self.load_model()

        self.train_data, _, self.encoders = make_data(encoding='le')
        splitter = TrainTestSplit(seed=313)
        X_train, X_test, y_train, y_test = splitter.random_split_by_groups(
            x=self.train_data.iloc[:, 0:-1],
            y=self.train_data.iloc[:, -1],
            groups=self.train_data['Adsorbent'])

    def load_model(self):

        model_path = os.path.join(os.getcwd(), "models", self.name)

        if self.name == "XGBoostLSS":
            
            model_ = joblib.load(os.path.join(model_path, "XGBoostLSS"))
            model_.num_ins = 17
            model_.input_features = ['Adsorbent', 'Feedstock', 'Pyrolysis_temp', 'Heating rate (oC)',
       'Pyrolysis_time (min)', 'C', 'O', 'Surface area',
       'Adsorption_time (min)', 'Ci_ppm', 'solution pH', 'rpm', 'Volume (L)',
       'loading (g)', 'adsorption_temp', 'Ion Concentration (mM)', 'ion_type']
        elif self.name == "Monte Carlo Dropout":

            # Load the model architecture
            model_ = build_monte_carlo_model()

            model_.num_ins = 17
            model_weights = os.path.join(model_path, "weights.hdf5")
            model_.load_weights(model_weights)
            model_.input_features = ['Adsorbent', 'Feedstock', 'Pyrolysis_temp', 'Heating rate (oC)',
                                     'Pyrolysis_time (min)', 'C', 'O', 'Surface area',
                                     'Adsorption_time (min)', 'Ci_ppm', 'solution pH', 'rpm', 'Volume (L)',
                                     'loading (g)', 'adsorption_temp', 'Ion Concentration (mM)', 'ion_type']

        elif self.name == "Probabilistic_Neural_Network":

            # Load the model architecture
            model_ = build_bayesian_nn_model()

            model_.num_ins = 17
            model_weights = os.path.join(model_path, "weights.hdf5")
            model_.load_weights(model_weights)
            model_.input_features = ['Adsorbent', 'Feedstock', 'Pyrolysis_temp', 'Heating rate (oC)',
                                     'Pyrolysis_time (min)', 'C', 'O', 'Surface area',
                                     'Adsorption_time (min)', 'Ci_ppm', 'solution pH', 'rpm', 'Volume (L)',
                                     'loading (g)', 'adsorption_temp', 'Ion Concentration (mM)', 'ion_type']

        else:
            conf_path = os.path.join(model_path, "config.json")
            model_ = Model.from_config_file(conf_path)
            model_.update_weights(os.path.join(model_path, self.weights[self.name]))

        return model_

    def make_prediction(self, inputs:list):

        assert len(inputs) == self.model.num_ins

        df = pd.DataFrame(
            np.array(inputs).reshape(1,-1),
            columns=self.model.input_features
        )
        df.loc[0, 'Adsorbent'] = self.transform_categorical('Adsorbent', df.loc[0, 'Adsorbent'])
        df.loc[0, 'Feedstock'] = self.transform_categorical('Feedstock', df.loc[0, 'Feedstock'])
        df.loc[0, 'ion_type'] = self.transform_categorical('ion_type', df.loc[0, 'ion_type'])

        # Ensure that all columns in df are of numeric type
        df = df.astype(float)

        if self.name == "XGBoostLSS":
            dtrain = xgb.DMatrix(df.values, nthread=1)
            pred_samples = self.model.predict(dtrain,
                              pred_type="samples",
                              n_samples=1000,
                              seed=123)

            preds = np.exp(pred_samples)
            if preds.shape[1] == 1:
                pred = preds.mean(axis=0)
            else:
                pred = preds.mean(axis=1)
            print(type(pred), pred.shape, preds.shape, type(preds))
            # Calculating upper and lower limit
            max_limit_ = pred + 5
            min_limit_ = pred - 5
        elif self.name == "Monte Carlo Dropout":
            inputs = df.values.reshape(1, -1)

            n = 100
            train_pred = np.full(shape=(n, inputs.shape[0]), fill_value=np.nan)
            for i in range(n):
                train_pred[i] = self.model(inputs, training=True).numpy().reshape(-1, )
            tr_mean = np.mean(train_pred, axis=0)

            # Calculating Upper Limit
            max_limit_ = np.max(train_pred, axis=0)

            # Calculating Lower Limit
            min_limit_ = np.min(train_pred, axis=0)

            print('limits: ', min_limit_, max_limit_)

            pred = float(tr_mean)

        elif self.name == "Probabilistic_Neural_Network":
            inputs = df.values.reshape(1, -1)

            train_dist = self.model(inputs)
            train_mean = train_dist.mean().numpy().reshape(-1, )
            train_std = train_dist.stddev().numpy().reshape(-1, )

            # Calculating Upper Limit
            max_limit_ = train_mean + train_std

            # Calculating Lower Limit
            min_limit_ = train_mean - train_std

            print('limits: ', min_limit_, max_limit_)

            pred = float(train_mean)

        else:

            pred = self.model.predict(df.values.reshape(1, -1))
            max_limit_ = pred + 5
            min_limit_ = pred - 5
        if isinstance(pred, np.ndarray):
            if len(pred) == 1:
                pred = pred[0]
            else:
                raise ValueError
        return pred, float(max_limit_), float(min_limit_)

    def transform_categorical(self, feature: str, category: str) -> int:
        assert isinstance(category, str)
        category_as_num = self.encoders[feature].transform(np.array([category]).reshape(-1, 1))
        assert len(category_as_num) == 1
        return category_as_num[0]
# %%

st.set_page_config(
    page_title="Probabilitic Prediction",
    #initial_sidebar_state="collapsed"
)

st.title('ML-based prediction of $q_{e}$ of Biochar for $PO_{4}$')

# Load data
with st.sidebar.expander("**Model Selection**", expanded=True):
    seleted_model = st.selectbox(
        "Select a Model",
        options=['NGBoost', 'Monte Carlo Dropout', 'XGBoostLSS', 'Probabilistic_Neural_Network'],
        help='select the machine learning model which will be used for prediction',
    )


with st.sidebar.expander("**Batch Prediction**", expanded=True):
    X = st.file_uploader(
    "Upload a csv file", type="csv", help="""Load the csv file which contains your own data.
    The machine learning model will make prediction on the whole data from csv file.! 
    """
    )


with st.sidebar.expander("**Retrain**", expanded=True):
    new_data = st.file_uploader(
    "Upload a csv file", type="csv", help="""Load the csv file which contains your own data.
    The data in this file will be combined the our dataset and the machine learning model will be
    trained again! 
    """
    )

with st.form('key1'):
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        adsorbent = st.text_input("Adsorbent", value='Mg-600')
    with col2:
        feedstock = st.text_input("Feedstock", value='bamboo (PP)')
    with col3:
        pyrol_temp = st.number_input('Pyrolysis Temperature', value=600)
    with col4:
        heat_rate = st.number_input("Heating Rate (C)", value=10)

    col5, col6, col7, col8 = st.columns(4)
    with col5:
        pyrol_time = st.number_input("Pyrolysis Time (min)", value=60)
    with col6:
        c = st.number_input("Carbon (%)", value=21.1)
    with col7:
        o = st.number_input("O (%)", value=29.1)
    with col8:
        surface_area = st.number_input("Surface Area", value=399)

    col9, col10, col11, col12 = st.columns(4)
    with col9:
        ads_time = st.number_input("Adsorption Time (min)", value=5400)
    with col10:
        ci = st.number_input("Initial Concentration", value=500)
    with col11:
        sol_ph = st.number_input("Solution pH", value=2.68)
    with col12:
        rpm = st.number_input("rpm", value=100)

    col13, col14, col15, col16 = st.columns(4)
    with col13:
        vol = st.number_input("Volume (L)", value=0.1)
    with col14:
        loading = st.number_input("Loading (g)", value=0.1)
    with col15:
        ads_temp = st.number_input("Adsorption Temp (C)", value=25)
    with col16:
        ion_conc = st.number_input("Ion Concentration (mM)", value=0.0)

    col17, _ = st.columns(2)
    with col17:
        ion_type = st.text_input("Ion Type", value='FREE')

    st.form_submit_button(label="Predict")

# %%

mh = ModelHandler(seleted_model)

point_prediction, max_limit, min_limit = mh.make_prediction(
    [adsorbent, feedstock, pyrol_temp, heat_rate, pyrol_time,
     c, o, surface_area, ads_time, ci, sol_ph,
     rpm, vol, loading, ads_temp, ion_conc, ion_type]
)
point_prediction = round(point_prediction, 4)
upper_limit = round(max_limit, 4)
lower_limit = round(min_limit, 4)

# %%

colors=["#002244", "#ff0066", "#66cccc", "#ff9933", "#337788",
          "#429e79", "#474747", "#f7d126", "#ee5eab", "#b8b8b8"]

col1, col2, col3 = st.columns(3)
col1.markdown(
    f"<p style='color: {colors[1]}; "
    f"font-weight: bold; font-size: 20px;'> Point Prediction</p>",
    unsafe_allow_html=True,
)
col1.text(point_prediction)
col2.markdown(
    f"<p style='color: {colors[1]}; "
    f"font-weight: bold; font-size: 20px;'> Upper Limit</p>",
    unsafe_allow_html=True,
)
col2.text(upper_limit)
col3.markdown(
    f"<p style='color: {colors[1]}; "
    f"font-weight: bold; font-size: 20px;'> Lower Limit</p>",
    unsafe_allow_html=True,
)
col3.text(lower_limit)
