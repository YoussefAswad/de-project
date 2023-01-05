from airflow import DAG
from airflow.utils.dates import days_ago
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy import stats
import datetime
from bs4 import BeautifulSoup
import requests
import os
import plotly.express as px
import plotly.tools as tls

import dash
import dash_core_components as dcc
import dash_html_components as html
from sqlalchemy import create_engine

data_path = "/opt/airflow/data/"
dataset = '1980_Accidents_UK.csv'
dataset_clean = '1980_Accidents_UK_clean.csv'
dataset_ex = '1980_Accidents_UK_ex.csv'


def isNowInTimePeriod(startTime, endTime, nowTime):
    if startTime < endTime:
        return nowTime >= startTime and nowTime <= endTime
    else:
        # Over midnight:
        return nowTime >= startTime or nowTime <= endTime


def clean(filename):
    df_accidents_1980 = pd.read_csv(filename, index_col=None)

    df_accidents_1980_clean = df_accidents_1980.replace(
        'Data missing or out of range', np.nan)
    df_accidents_1980_clean = df_accidents_1980_clean.replace(
        -1, np.nan)
    df_accidents_1980_clean = df_accidents_1980_clean.replace(
        '-1', np.nan)
    df_accidents_1980_clean = df_accidents_1980_clean.replace(
        'first_road_class is C or Unclassified. These roads do not have official numbers so recorded as zero ', 0)
    df_accidents_1980_clean = df_accidents_1980_clean.dropna(
        axis='columns', how='all')
    df_accidents_1980_clean = df_accidents_1980_clean.drop(
        'accident_index', axis=1)  # dropping the accident_index
    df_accidents_1980_clean = df_accidents_1980_clean.set_index(
        'accident_reference')
    df_accidents_1980_clean = df_accidents_1980_clean.dropna(
        axis='index', how='any', subset=['location_easting_osgr', 'location_easting_osgr', 'junction_detail', 'first_road_number', 'light_conditions', 'weather_conditions', 'road_surface_conditions', 'carriageway_hazards'])
    df_filter = df_accidents_1980_clean[(df_accidents_1980_clean['second_road_class'].notna()) & ~(
        df_accidents_1980_clean['second_road_number'].notna())].index
    df_accidents_1980_clean = df_accidents_1980_clean.drop(df_filter)
    df_filter = df_accidents_1980_clean[((df_accidents_1980_clean['junction_detail'] != 'Not at junction or within 20 metres') & ~(
        df_accidents_1980_clean['junction_control'].notna())) | ((df_accidents_1980_clean['junction_detail'].isna()) & (
            df_accidents_1980_clean['junction_control'].notna()))].index
    df_accidents_1980_clean = df_accidents_1980_clean.drop(df_filter)
    data = df_accidents_1980_clean[['pedestrian_crossing_human_control',
                                    'pedestrian_crossing_physical_facilities']].dropna()
    data['pedestrian_crossing_physical_facilities'] = data['pedestrian_crossing_physical_facilities'].astype(
        'category')
    data['pedestrian_crossing_physical_facilities'] = data['pedestrian_crossing_physical_facilities'].cat.codes

    data['pedestrian_crossing_human_control'] = data['pedestrian_crossing_human_control'].astype(
        'category')
    data['pedestrian_crossing_human_control'] = data['pedestrian_crossing_human_control'].cat.codes
    data["pedestrian_crossing_human_control"].median()
    ind = data[data["pedestrian_crossing_human_control"] ==
               data["pedestrian_crossing_human_control"].median()].index[0]
    pedesMedian = df_accidents_1980_clean["pedestrian_crossing_human_control"][ind]
    df_accidents_1980_clean["pedestrian_crossing_human_control"] = df_accidents_1980_clean["pedestrian_crossing_human_control"].replace(
        np.nan, pedesMedian)
    data["pedestrian_crossing_physical_facilities"].median()
    ind = data[data["pedestrian_crossing_physical_facilities"] ==
               data["pedestrian_crossing_physical_facilities"].median()].index[0]
    pedesMedian = df_accidents_1980_clean["pedestrian_crossing_physical_facilities"][ind]
    df_accidents_1980_clean["pedestrian_crossing_physical_facilities"] = df_accidents_1980_clean[
        "pedestrian_crossing_physical_facilities"].replace(np.nan, pedesMedian)
    df_accidents_1980_clean["special_conditions_at_site"].unique()
    data = df_accidents_1980_clean[['special_conditions_at_site']].dropna()
    data['special_conditions_at_site'] = data['special_conditions_at_site'].astype(
        'category')
    data['special_conditions_at_site'] = data['special_conditions_at_site'].cat.codes
    data["special_conditions_at_site"].median()
    ind = data[data["special_conditions_at_site"] ==
               data["special_conditions_at_site"].median()].index[0]
    specMedian = df_accidents_1980_clean["special_conditions_at_site"][ind]
    df_accidents_1980_clean["special_conditions_at_site"] = df_accidents_1980_clean["special_conditions_at_site"].replace(
        np.nan, pedesMedian)
    data = df_accidents_1980_clean[['road_type']].dropna()
    data['road_type'] = data['road_type'].astype(
        'category')
    data['road_type'] = data['road_type'].cat.codes
    sns.kdeplot(data["road_type"])
    data["road_type"].median()
    ind = data[data["road_type"] == data["road_type"].median()].index[0]
    specMedian = df_accidents_1980_clean["road_type"][ind]
    df_accidents_1980_clean["road_type"] = df_accidents_1980_clean["road_type"].replace(
        np.nan, pedesMedian)
    df_accidents_1980_clean["junction_control"] = df_accidents_1980_clean["junction_control"].replace(
        np.nan, "None")

    df_accidents_1980_clean = df_accidents_1980_clean.drop(
        'second_road_class', axis=1)
    df_accidents_1980_clean = df_accidents_1980_clean.drop(
        'second_road_number', axis=1)
    df_accidents_1980_clean = df_accidents_1980_clean.drop(
        'accident_year', axis=1)
    z = np.abs(stats.zscore(df_accidents_1980_clean['number_of_vehicles']))
    veh_filtered_entries = z < 3
    veh_med = df_accidents_1980_clean[veh_filtered_entries]['number_of_vehicles'].median(
    )
    df_accidents_1980_clean['number_of_vehicles'] = df_accidents_1980_clean['number_of_vehicles'].where(
        veh_filtered_entries, other=veh_med)
    df_accidents_1980_clean[~veh_filtered_entries]
    z = np.abs(stats.zscore(df_accidents_1980_clean['number_of_casualties']))
    cas_filtered_entries = z < 4
    cas_med = df_accidents_1980_clean[cas_filtered_entries]['number_of_casualties'].median(
    )
    df_accidents_1980_clean['number_of_casualties'] = df_accidents_1980_clean['number_of_casualties'].where(
        cas_filtered_entries, other=cas_med)
    df_accidents_1980_clean[~cas_filtered_entries]
    df_accidents_1980_clean['date'] = pd.to_datetime(df_accidents_1980_clean['date'].apply(
        lambda x: datetime.datetime.strptime(x, '%d/%m/%Y')))
    df_accidents_1980_clean['time'] = pd.to_datetime(df_accidents_1980_clean['time'].apply(
        lambda x: (datetime.datetime.strptime(x, '%H:%M'))))

    df_accidents_1980_clean['week_number'] = df_accidents_1980_clean['date'].apply(
        lambda x: x.isocalendar().week)
    enc_cols = ['police_force', 'accident_severity', 'day_of_week', 'local_authority_district', 'first_road_class', 'road_type', 'junction_detail', 'junction_control', 'pedestrian_crossing_human_control',
                'pedestrian_crossing_physical_facilities', 'light_conditions', 'weather_conditions', 'road_surface_conditions', 'special_conditions_at_site', 'carriageway_hazards']
    df_lookup = pd.DataFrame()
    for i, col in enumerate(enc_cols):
        col_cat = df_accidents_1980_clean[col].astype('category')
        col_look = pd.DataFrame({col: df_accidents_1980_clean[col].unique()})
        df_accidents_1980_clean[col] = col_cat.cat.codes
        df_lookup = pd.concat([df_lookup, col_look], axis=1)
    df_lookup.head(20)
    z = np.array((df_accidents_1980_clean['day_of_week']))

    is_workday = np.logical_and(z != 2, z != 6)
    df_accidents_1980_clean['is_workday'] = is_workday
    is_midnight = np.array((df_accidents_1980_clean['time'].apply(
        lambda x: isNowInTimePeriod(datetime.time(0, 0), datetime.time(3, 0), x.time()))))
    df_accidents_1980_clean['is_midnight'] = is_midnight

    df_lookup.to_csv(data_path + 'lookup.csv')
    df_accidents_1980_clean.to_csv(
        data_path + dataset_clean)
    pass


def extract(filename):
    web = 'https://www.britannica.com/topic/list-of-cities-and-towns-in-the-United-Kingdom-2034188'
    response = requests.get(web)
    content = response.text
    soup = BeautifulSoup(content)
    district = soup.find_all('ul', class_='topic-list')

    df_accidents_1980_clean = pd.read_csv(
        filename, index_col='accident_reference')
    df_lookup = pd.read_csv(data_path +
                            'lookup.csv', index_col=0)

    len(district)
    df_lookup
    df_accidents_1980_clean['area_type'] = "N/A"
    for area in district:
        type = (area.findChild("div", recursive=True))
        city = (type.findChild("a", recursive=False))
        index = type.getText().find(")")
        type = type.getText()[:index + 1]
        index = type.find("(")
        type = type[index + 1:-1]
        if (city is not None):
            city = city.get_text()
            # print(df_lookup['local_authority_district'][df_accidents_1980_clean['local_authority_district']])
            mask = df_lookup['local_authority_district'][df_accidents_1980_clean['local_authority_district']].str.lower(
            ).str.replace(" ", "") == city.lower().replace(" ", "")
            # df_accidents_1980_clean.loc[df_lookup['local_authority_district'] [df_accidents_1980_clean['local_authority_district']] == city, 'area_type'] = type\
            # print( np.asarray(mask))
            df_accidents_1980_clean['area_type'] = df_accidents_1980_clean['area_type'].where(
                np.asarray(~mask), type)
    col = "area_type"
    col_cat = df_accidents_1980_clean[col].astype('category')
    col_look = pd.DataFrame({col: df_accidents_1980_clean[col].unique()})
    df_accidents_1980_clean[col] = col_cat.cat.codes
    df_lookup = pd.concat([df_lookup, col_look], axis=1)
    df_lookup.to_csv('lookup.csv')
    df_accidents_1980_clean.to_csv(
        data_path + dataset_ex)
    pass


def cleanup():
    os.system('rm ' + data_path + 'lookup.csv')
    os.system('rm ' + data_path + dataset_clean)
    os.system('rm ' + data_path + dataset_ex)


def create_dashboard(filename):
    engine = create_engine(
        'postgresql://root:root@pgdatabase_accidents:5432/accidents_uk')
    df = pd.read_sql('select * from "1980_Accidents_UK"',
                     engine, index_col="accident_reference")

    df_lookup = pd.read_sql(
        'select * from "lookup_table"', engine)
    df['accident_severity'] = df['accident_severity'].apply(
        lambda x: df_lookup['accident_severity'][x])
    df['light_conditions'] = df['light_conditions'].apply(
        lambda x: df_lookup['light_conditions'][x])
    # sdf['time'] = df['time'].apply(lambda x: datetime.datetime.strptime(x, "%Y-%m-%d %H:%M:%S").time())
    df['time'] = df['time'].apply(lambda x: int(x[-8:-6]))
    print(df['time'])
    df['date'] = df['date'].apply(lambda x: int(x[5:7]))
    print(df['date'])
    histo_time = px.histogram(df, x="time", nbins=24).update_xaxes(
        categoryorder="total ascending")

    histo_date = px.histogram(df, x="date", nbins=12).update_xaxes(
        categoryorder="total ascending")
    histo_date.update_layout(
        xaxis=dict(
            tickmode='array',
            tickvals=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
            ticktext=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'June',
                      'July', 'Aug', 'Sept', 'Oct', 'Nov', 'Dec']
        )
    )
    histo_light = px.histogram(df, x="light_conditions")
    tt1 = sns.barplot(x="accident_severity", y="number_of_casualties",
                      data=df)
    mpl_fig = plt.gcf()
    bar_sev = tls.mpl_to_plotly(mpl_fig)
    bar_sev.update_layout(
        xaxis=dict(
            tickmode='array',
            tickvals=[0, 1, 2],
            ticktext=["slight", "serious", "fatal"]
        )
    )
    tt2 = sns.barplot(x="accident_severity", y="speed_limit",
                      data=df)
    mpl_fig2 = plt.gcf()
    bar_sev2 = tls.mpl_to_plotly(mpl_fig2)
    bar_sev2.update_layout(
        xaxis=dict(
            tickmode='array',
            tickvals=[0, 1, 2],
            ticktext=["slight", "serious", "fatal"]
        )
    )
    app = dash.Dash()
    app.layout = html.Div(
        children=[
            html.H1(children="UK Accidents 1980 dataset",),
            # html.P(
            #     children="Age vs Survived Titanic dataset",
            # ),
            # dcc.Graph(
            #     figure={
            #         "data": [
            #             {
            #                 "x": df["Age"],
            #                 "y": df["Survived"],
            #                 "type": "lines",
            #             },
            #         ],
            #         "layout": {"title": "Age vs Survived"},
            #     },
            # ),
            html.P(
                children="Number of accidents in relation to time of day",
            ),
            dcc.Graph(
                figure=histo_time
            ),
            html.P(
                children="Number of accidents in relation to month",
            ),
            dcc.Graph(
                figure=histo_date
            ),
            html.P(
                children="Number of accidents in relation to light conditions",
            ),
            dcc.Graph(
                figure=histo_light
            ),
            html.P(
                children="Relation between accident_severity and number_of_casualties",
            ),
            dcc.Graph(
                figure=bar_sev
            ),
            html.P(
                children="Relation between accident_severity and speedlimit",
            ),
            dcc.Graph(
                figure=bar_sev2
            ),
        ]
    )
    app.run_server(host='0.0.0.0')
    print('dashboard is successful and running on port 8000')


def load_to_postgres(filename):
    df_accidents_1980_clean = pd.read_csv(
        filename, index_col='accident_reference')
    df_lookup = pd.read_csv("/opt/airflow/data/" +
                            'lookup.csv', index_col=0)
    engine = create_engine(
        'postgresql://root:root@pgdatabase_accidents:5432/accidents_uk')
    if (engine.connect()):
        print('connected succesfully')
    else:
        print('failed to connect')
    df_accidents_1980_clean.to_sql(
        name='1980_Accidents_UK', con=engine, if_exists='replace')
    df_lookup.to_sql(name='lookup_table', con=engine, if_exists='replace')


default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    'start_date': days_ago(2),
    "retries": 1,
}

dag = DAG(
    'accidents_uk_pipeline',
    default_args=default_args,
    description='Uk accidents',
)
with DAG(
    dag_id='accidents_uk_pipeline',
    schedule_interval='@once',
    default_args=default_args,
    tags=['accidents-pipeline'],
)as dag:
    clean_task = PythonOperator(
        task_id='clean',
        python_callable=clean,
        op_kwargs={
            "filename": data_path + dataset
        },
    )
    extract_task = PythonOperator(
        task_id='extract',
        python_callable=extract,
        op_kwargs={
            "filename": data_path + dataset_clean
        },
    )
    load_to_postgres_task = PythonOperator(
        task_id='load_to_postgres',
        python_callable=load_to_postgres,
        op_kwargs={
            "filename": data_path + dataset_ex
        },
    )
    create_dashboard_task = PythonOperator(
        task_id='create_dashboard_task',
        python_callable=create_dashboard,
        op_kwargs={
            "filename": data_path + dataset_ex
        },
    )
    cleanup_task = PythonOperator(
        task_id='cleanup',
        python_callable=cleanup,
    )

    clean_task >> extract_task >> load_to_postgres_task >> cleanup_task >> create_dashboard_task
