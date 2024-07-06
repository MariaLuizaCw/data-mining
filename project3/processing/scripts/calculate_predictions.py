import psycopg2
import pandas as pd
import matplotlib.pyplot as plt
import folium
from geopy.distance import geodesic
from pyproj import Geod
import numpy as np
from shapely.wkt import loads
from geopy.distance import geodesic
from shapely.geometry import Point, Polygon
from sqlalchemy import create_engine
from shapely.geometry import Point, Polygon, LineString
import geopandas as gpd
import importlib
from pyproj import CRS, Transformer
from scipy.integrate import quad
from scipy.integrate import cumtrapz


from pyproj import Geod
from sklearn.metrics import mean_squared_error
geod = Geod(ellps="WGS84")
from shapely.ops import nearest_points
from geopy.distance import geodesic
import mplleaflet
import seaborn as sns
from shapely.wkt import loads
from psycopg2.extras import execute_values

import sys
import os
# Defina o caminho do diretório manualmente
module_path = os.path.abspath(os.path.join('..', 'modules'))

# Adicione o diretório ao sys.path
sys.path.append(module_path)
from calculate_mean_speed_modules import *


from predict_modules import *


# Função para reconstruir a trajetória
def get_routes(conn):
    query = f"""
    SELECT
        linha,
        week_day,
        ST_AsText(start_point) as start_point,
        ST_AsText(end_point) as end_point,
        ST_AsText(route_line_out) as route_line_out,
        ST_AsText(route_line_back) as route_line_back
    FROM public.bus_routes
    """
    
    # Carregar dados em um GeoDataFrame
    df = pd.read_sql(query, conn)

    return df

# Função trazer os dados de um onibus
def get_bus_train_data_datahora(conn, linha):
    # Selecionar dados da tabela bus_routes para a linha específica
    query = f'''select
                a.*,
                b.latitude as latitude_resposta,
                b.longitude as longitude_resposta,
                b.filename as filename_resposta
                from public.vehicle_tracking_treino_datahora a left join 
                public.vehicle_tracking_treino_latlong_resposta b on 
                (a.id = b.id and split_part(a.filename, 'treino-', 2) = split_part(b.filename, 'resposta-', 2))
                where
                linha = '{linha}'
            '''
        
    # Carregar dados em um GeoDataFrame
    df = pd.read_sql(query, conn)

    return df


# Função trazer os dados de um onibus
def get_bus_train_data_latlong(conn, linha):
    # Selecionar dados da tabela bus_routes para a linha específica
    query = f'''select
                a.*,
                     b.datahora_epoch as datahora_epoch_resposta,
                     b.filename as filename_resposta
                from public.vehicle_tracking_treino_latlong a left join 
                public.vehicle_tracking_treino_datahora_resposta b on (a.id = b.id  and split_part(a.filename, 'treino-', 2) = split_part(b.filename, 'resposta-', 2))
                where
                linha = '{linha}'
            '''
        
    # Carregar dados em um GeoDataFrame
    gdf = gpd.read_postgis(query, conn, geom_col='geom', crs='EPSG:4326')

    return gdf

def get_bus_train_data(conn, linha):
    query = f'''select 
                    *, 
                    ST_Transform(geom::geometry, 4326) AS geometry 
                from vehicle_tracking_treino_base where 
                linha = '{linha}'
            '''
        
    # Carregar dados em um GeoDataFrame
    gdf = gpd.read_postgis(query, conn, geom_col='geometry', crs='EPSG:4326')

    return gdf


def get_speed_model(conn, linha):
    query = f'''select 
                    *
                from bus_speed_model where 
                linha = '{linha}'
            '''
        
    # Carregar dados em um GeoDataFrame
    df = pd.read_sql(query, conn)

    return df


linha = sys.argv[1] 


# Conectar ao banco de dados PostgreSQL
conn = psycopg2.connect(
    dbname="postgres",
    user="postgres",
    password="postgres",
    host="localhost",
    port="5432"
)


print('Downloading Bus Info', linha)

routes = get_routes(conn)
bus_speed_df = get_speed_model(conn, linha)
bus_data_latlong = get_bus_train_data_latlong(conn, linha)
bus_data_datahora = get_bus_train_data_datahora(conn, linha)
bus_data_train = get_bus_train_data(conn, linha)



bus_data_train['datahora'] = pd.to_datetime(bus_data_train['datahora'])
bus_data_train = gpd.GeoDataFrame(bus_data_train, geometry='geometry', crs="EPSG:4326")


data = bus_data_latlong['filename'].str.split('-', expand=True)
data = data[1] + data[2] + data[3].str[:-5] + '00'
bus_data_latlong['filename_data'] = pd.to_datetime(data, format='%Y%m%d_%H%M')


data = bus_data_datahora['filename'].str.split('-', expand=True)
data = data[1] + data[2] + data[3].str[:-5] + '00'
bus_data_datahora['filename_data'] = pd.to_datetime(data, format='%Y%m%d_%H%M')
bus_data_datahora['point_resposta'] = bus_data_datahora.apply(lambda x: Point(x['longitude_resposta'], x['latitude_resposta']), axis=1)
bus_data_datahora = gpd.GeoDataFrame(bus_data_datahora, geometry='point_resposta', crs="EPSG:4326")


routes['route_line_out'] = routes['route_line_out'].map(lambda x: loads(x))
routes['route_line_back'] = routes['route_line_back'].map(lambda x: loads(x))
routes['end_point'] = routes['end_point'].map(lambda x: loads(x))
routes['start_point'] = routes['start_point'].map(lambda x: loads(x))
routes_gdf = gpd.GeoDataFrame(routes, geometry='start_point', crs="EPSG:4326")
routes.head()


insert_query_datahora = '''
    INSERT INTO vehicle_tracking_pred_datahora (id, filename, datahora_epoch_pred, datahora_epoch_resposta)
    VALUES %s
    '''


insert_query_latlong = '''
    INSERT INTO vehicle_tracking_pred_latlong (id, filename, latitude_pred, longitude_pred, latitude_resposta, longitude_resposta)
    VALUES %s
    '''

print('Starting Lat long predictions')
pred_lat_long = PredictLatLong()
for filename in bus_data_datahora['filename'].unique():
    print('Predicting filename', filename)
    ordens = bus_data_datahora.loc[bus_data_datahora['filename'] == filename, 'ordem'].unique()
    for ordem in ordens:
        pred_datahora = pred_lat_long.predict_lat_long(filename, ordem, routes_gdf, bus_speed_df, bus_data_datahora, bus_data_train, linha, handle_end=False)   
        if pred_datahora.shape[0] > 0:
            tuples = [tuple(x) for x in pred_datahora[['id_actual', 'filename', 'latitude_actual', 'longitude_actual', 'latitude_resposta', 'longitude_resposta']].to_numpy()]
            with conn.cursor() as cur:
                execute_values(cur, insert_query_latlong, tuples)
                conn.commit()

print('Starting time predictions')
pred_time = PredictTime()
for filename in bus_data_latlong['filename'].unique():
    print('Predicting filename', filename)
    ordens = bus_data_latlong.loc[bus_data_latlong['filename'] == filename, 'ordem'].unique()
    for ordem in ordens:
        pred_latlong_handle = pred_time.predict_epoch(filename, ordem, routes_gdf, bus_speed_df, bus_data_latlong, bus_data_train, linha, handle_end=True)    
        if pred_latlong_handle.shape[0] > 0:
            tuples = [tuple(x) for x in pred_latlong_handle[['id_actual', 'filename', 'datahora_epoch_actual', 'datahora_epoch_resposta']].to_numpy()]
            with conn.cursor() as cur:
                execute_values(cur, insert_query_datahora, tuples)
                conn.commit()


