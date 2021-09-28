import dask.dataframe as dd
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from sklearn.utils.validation import column_or_1d
import streamlit as st
import shap

st.set_page_config(
    page_title="Boston Hause Predict",#titulo
    page_icon="🏠",#Icono
    layout="wide",
    initial_sidebar_state="expanded",
)

st.write("""
# Aplicacion para predecir el precio de las casas de Boston
---
la aplicacion va a predecir el precio de las casas segun un dataset(BostonHause dataset), se usaran el modelo del random forest de sklern
""")

boston = datasets.load_boston()
X = pd.DataFrame(boston.data, columns = boston.feature_names)
y = pd.DataFrame(boston.target, columns = ['MEDV'])
st.sidebar.header('Especifique los parametros de entrada para realizar su predicion')

crim = st.sidebar.slider('CRIM',float(X.CRIM.min()),float(X.CRIM.max()),float(X.CRIM.mean()))
zn = st.sidebar.slider('ZN',float(X.ZN.min()),float(X.ZN.max()),float(X.ZN.mean()))
indus = st.sidebar.slider('INDUS',float(X.INDUS.min()),float(X.INDUS.max()),float(X.INDUS.mean()))
chas = st.sidebar.slider('CHAS',float(X.CHAS.min()),float(X.CHAS.max()),float(X.CHAS.mean()))
nox = st.sidebar.slider('NOX',float(X.NOX.min()),float(X.NOX.max()),float(X.NOX.mean()))
rm = st.sidebar.slider('RM',float(X.RM.min()),float(X.RM.max()),float(X.RM.mean()))
age = st.sidebar.slider('AGE',float(X.AGE.min()),float(X.AGE.max()),float(X.AGE.mean()))
dis = st.sidebar.slider('DIS',float(X.DIS.min()),float(X.DIS.max()),float(X.DIS.mean()))
rad = st.sidebar.slider('RAD',float(X.RAD.min()),float(X.RAD.max()),float(X.RAD.mean()))
tax = st.sidebar.slider('TAX',float(X.TAX.min()),float(X.TAX.max()),float(X.TAX.mean()))
ptratio = st.sidebar.slider('PTRATIO',float(X.PTRATIO.min()),float(X.PTRATIO.max()),float(X.PTRATIO.mean()))
b = st.sidebar.slider('B',float(X.B.min()),float(X.B.max()),float(X.B.mean()))
lstat = st.sidebar.slider('LSTAT',float(X.LSTAT.min()),float(X.LSTAT.max()),float(X.LSTAT.mean()))

dicCaracteristicas = {
    'CRIM':crim,
    'ZN':zn,
    'INDUS':indus,
    'CHAS':chas,
    'NOX':nox,
    'RM':rm,
    'AGE':age,
    'DIS':dis,
    'RAD':rad,
    'TAX':tax,
    'PTRATIO':ptratio,
    'B':b,
    'LSTAT':lstat,
}
st.header('Parametros de entrada')
df_new = pd.DataFrame(dicCaracteristicas,index=[0])
st.write(df_new)
st.write("---")
modelo = RandomForestRegressor()
modelo.fit(X,y)
predict = modelo.predict(df_new)
st.header('Predicion realizada por el usuario')
st.write(predict)

explainer = shap.TreeExplainer(modelo)
valoresIncidentes = explainer.shap_values(X)
st.write("---")
st.header('Importancia de Parametros segun SHAP')
shap.summary_plot(valoresIncidentes,X)
st.pyplot(bbox_inches='tight')
st.write("---")
st.header('Importancia de Los parametros de entrada con un grfico de barras')
shap.summary_plot(valoresIncidentes,X,plot_type = 'bar')
st.pyplot(bbox_inches='tight')
