import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import boto3
from smart_open import open 
from glob import glob


st.title('Astorga Tomotherapy Blue Physics Analysis')

s3 = boto3.client('s3')

response = s3.list_objects_v2(Bucket='astorgacolombia', Delimiter='/')

listofdirectories = []

for common_prefix in response.get('CommonPrefixes', []):
        # Extract the folder name from the CommonPrefixes
        folder_name = common_prefix['Prefix'].rstrip('/')
        listofdirectories.append(folder_name)

directory1 = st.selectbox('Select Directory', listofdirectories)

response2 = s3.list_objects_v2(Bucket='astorgacolombia', Prefix=directory1)

#list of files in directory except

listoffiles = [file['Key'] for file in response2.get('Contents', [])]

@st.cache_data
def read_dataframe(file):
    path = f's3://astorgacolombia/{file}'
    df0 = pd.read_csv(path, skiprows = 4)
    #df0 = pd.read_csv(file, skiprows = 4) 
    return df0

filenow = st.selectbox('Select File to draw', listoffiles)
st.write('Raw data file. Can be download in csv format')
df0 = read_dataframe(filenow)
zeros = df0.loc[(df0.time < 10), 'ch0':].mean()
dfzeros = df0.loc[:, 'ch0':] - zeros
dfzeros.columns = ['ch0z', 'ch1z']
dfz = pd.concat([df0, dfzeros], axis = 1)
dfz['dose'] = dfz.ch0z - dfz.ch1z
st.write(dfz.loc[:,['number', 'time', 'ch0z', 'ch1z', 'dose']])
df1 = dfz.loc[:, ['number', 'time', 'ch0z']]
df1.columns = ['number', 'time', 'reading']
df1['ch'] = 'sensor'
df2 = dfz.loc[:, ['number', 'time', 'ch1z']]
df2.columns = ['number', 'time', 'reading']
df2['ch'] = 'cerenkov'
df3 = dfz.loc[:, ['number', 'time', 'dose']]
df3.columns = ['number', 'time', 'reading']
df3['ch'] = 'dose'
dftp = pd.concat([df1, df2, df3]) 
fig0 = px.scatter(dftp, x='time', y='reading', color='ch', title='Raw Data Plot')
fig0.update_layout(
        xaxis_title = "time (s)",
        yaxis_title = "Voltage accumulated every 750 %ss (V)" %u"\u00B5"
        )
st.plotly_chart(fig0)

integrate = st.checkbox('Integrate')
if integrate:
    df1['chunk'] = df1.number // 300
    df1g = df1.groupby('chunk').agg({'time':np.median, 'reading':np.sum})
    df1g = df1g.iloc[:-1, :]
    df1g['readingdiff'] = df1g.reading.diff()
    cutoff = st.slider('Chose cut-off to autodetect limits', min_value = 20, max_value = 200, value = 50)
    starttimes = df1g.loc[df1g.readingdiff > cutoff, 'time'].to_list()
    sts = [starttimes[0]] + [v for i, v in list(enumerate(starttimes))[1:] if abs(starttimes[i-1]-v)>1]
    stsg = [t - 0.2 for t in sts]
    finishtimes = df1g.loc[df1g.readingdiff < -cutoff, 'time'].to_list()
    fts = [finishtimes[0]] + [v for i, v in list(enumerate(finishtimes))[1:] if abs(finishtimes[i-1]-v)>1]
    ftsg = [t + 0.2 for t in fts]

    zeros = df0.loc[(df0.time < stsg[0]) | (df0.time > ftsg[-1]), 'ch0':].mean()
    dfzeros = df0.loc[:, 'ch0':] - zeros
    dfzeros.columns = ['ch0z', 'ch1z']
    dfz = pd.concat([df0, dfzeros], axis = 1)

    maxzeros = dfz.loc[(dfz.time < stsg[0]) | (dfz.time > ftsg[-1]), 'ch0z'].max()
    pulsethreshold = st.slider('Chose threshold for pulses', min_value = 1, max_value = 20, value = 5)
    dfz['pulse'] = dfz.ch0z > maxzeros * (1 + pulsethreshold/100)

    dfz['dose'] = dfz.ch0z - dfz.ch1z
    
    df1z = dfz.loc[:, ['number', 'time', 'ch0z']]
    df1z.columns = ['number', 'time', 'reading']
    df1z['ch'] = 'sensor'
    df2z = dfz.loc[:, ['number', 'time', 'ch1z']]
    df2z.columns = ['number', 'time', 'reading']
    df2z['ch'] = 'cerenkov'
    df3z = dfz.loc[:, ['number', 'time', 'dose']]
    df3z.columns = ['number', 'time', 'reading']
    df3z['ch'] = 'dose'
    dfpz = dfz.loc[:, ['number', 'time', 'pulse']]
    dfpz.columns = ['number', 'time', 'reading']
    dfpz['ch'] = 'pulse'
    dfztp = pd.concat([df1z, df2z, df3z, dfpz]) 
    dfztp['readingC'] = dfztp.reading * 0.03 
    

    fig1 = px.scatter(dfztp, x='time', y='readingC', color = 'ch', title = 'Data set to zero')
    fig1.update_layout(
        xaxis_title = "time (s)",
        yaxis_title = "Charge accumulated every 750 %ss (nC)" %u"\u00B5"
        )
    for n,(s, f) in enumerate(zip(stsg, ftsg)):
        fig1.add_vline(s, line_color = 'green', line_dash = 'dash')
        fig1.add_vline(f, line_color = 'red', line_dash = 'dash')
        dfz.loc[(dfz.time > s) & (dfz.time < f), 'shot'] = n
        

    st.plotly_chart(fig1)

    
    dfz['charge'] = dfz.ch0z * 0.03
    gr = dfz.groupby('shot')
    dfi = gr.agg({'time':np.min, 'charge':np.sum, 'pulse':np.sum, 'dose':np.sum})
    dfi.columns = ['start_time(s)', 'charge(nC)', 'pulses', 'dose(cGy)']
    dfi = dfi[['charge(nC)', 'pulses', 'start_time(s)', 'dose(cGy)']]
    dfi['end_time(s)'] = gr['time'].max()
    dfi['duration(s)'] = dfi['end_time(s)'] - dfi['start_time(s)']
    dfi['Avg_charge_per_pulse(pC)'] = dfi['charge(nC)'] / dfi.pulses * 1000
    dfi['Avg_dose_per_pulse(cGy)'] = dfi['dose(cGy)'] / dfi.pulses
    
    
    st.write('Result of integrals')
    st.write(dfi.round(2))

