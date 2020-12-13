
import altair as alt
import pandas as pd

from sklearn.preprocessing import MinMaxScaler

import matplotlib.pyplot as plt

def get_csv(dataset_name, filename):
    data_folder = f'data/{dataset_name}'
    df = pd.read_csv(
        f'{data_folder}/{filename}', 
        sep=',',
        header=0,
    )
    
    return df


def put_csv(df, dataset_name, filename):
    data_folder = f'data/{dataset_name}'
    df.to_csv(f"{data_folder}/{filename}", header=True, index=False)


def scale_minmax(df, col_name, min_val, max_val):
    scaler = MinMaxScaler(feature_range=(min_val, max_val))
    df_scaled = scaler.fit_transform(df[col_name].values.reshape(-1,1))

    return df_scaled



def calculate_quantile(df, col_name, *, quantiles=None):
    df_tmp = df[col_name].value_counts()
    if quantiles is None:
        quantiles = [0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99, 1]

    qntl = df_tmp.quantile(quantiles)
    df_qntl = qntl.to_frame('Observation')
    df_qntl.index.rename('%', inplace=True)
    
    return df_qntl
    
    
def draw_quantile(df, col_name, *, quantiles=None):
    df_qntl = calculate_quantile(df, col_name, quantiles=quantiles)

    df_qntl['Quantiles'] = df_qntl.index


    chart = alt.Chart(df_qntl).mark_line().encode(
        x=alt.X(
            "Quantiles:Q",
            axis=alt.Axis(
                tickCount=df_qntl.shape[0],
                grid=False,
                labelExpr="datum.value % 1 ? null : datum.label",
            )
        ),
        y='Observation'
    ).properties(
        title=f'quantile on observations / {col_name}',
        width=600,
        height=300,
    )
    
    chart.configure_title(
        fontSize=20,
        font='Courier',
        anchor='start',
        color='gray'
    )
    
    return chart
    
def draw_long_tail(df, col_name):
    import recmetrics
    
    fig = plt.figure(figsize=(15, 7))
    return recmetrics.long_tail_plot(
        df=df, 
        item_id_column=col_name, 
        interaction_type="interaction", 
        percentage=0.5,
        x_labels=False
    )