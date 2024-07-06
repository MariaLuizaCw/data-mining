import psycopg2
import pandas as pd
import matplotlib.pyplot as plt
import folium
import numpy as np
from shapely.wkt import loads
from geopy.distance import geodesic
from shapely.geometry import Point, Polygon
from sqlalchemy import create_engine
from shapely.geometry import Point, Polygon, LineString
import geopandas as gpd

import warnings
warnings.filterwarnings('ignore')
from pyproj import Geod
geod = Geod(ellps="WGS84")
from shapely.ops import nearest_points
from geopy.distance import geodesic
import mplleaflet
import seaborn as sns
from shapely.wkt import loads
from shapely.ops import nearest_points
from shapely.ops import split
import psycopg2.extras as extras
import os
import sys
module_path = os.path.abspath(os.path.join('..', 'modules'))
sys.path.append(module_path)
from calculate_mean_speed_modules import *

linha = sys.argv[1]

# Conectar ao banco de dados PostgreSQL
conn = psycopg2.connect(
    dbname="postgres",
    user="postgres",
    password="postgres",
    host="localhost",
    port="5432"
)
cursor = conn.cursor()

print('Downloading bus data', linha)

routes = get_routes(conn)
bus_df = get_bus_data(conn, linha)


print('Preparing dataframes', linha)
## Conversões do bus_gdf
bus_df  = bus_df.sort_values(by='datahora')
bus_df['datahora'] = pd.to_datetime(bus_df['datahora'])
bus_df['datahoraservidor'] = pd.to_datetime(bus_df['datahoraservidor'])
bus_df['datahoraenvio'] = pd.to_datetime(bus_df['datahoraenvio'])
bus_gdf = gpd.GeoDataFrame(bus_df, geometry='geometry', crs="EPSG:4326")

## Conversões do routes_gdf
routes['route_line_out'] = routes['route_line_out'].map(lambda x: loads(x))
routes['route_line_back'] = routes['route_line_back'].map(lambda x: loads(x))
routes['end_point'] = routes['end_point'].map(lambda x: loads(x))
routes['start_point'] = routes['start_point'].map(lambda x: loads(x))
routes_gdf = gpd.GeoDataFrame(routes, geometry='start_point', crs="EPSG:4326")



print('Filtering far points', linha)
## Filtro das linhas que passam de 30 metros da trajetoria
bus_gdf_filtered = remove_far_away_points(linha, routes, bus_gdf)

print('Removing date outliers', linha)
## Filtro das linhas que descordam da hora do servidor
new_bus_gdf = remove_time_outliers(bus_gdf_filtered)



for day in ['Monday', 'Saturday', 'Sunday']:
    for hour in ['rush_manha', 'tarde', 'rush_noite', 'noite']:
        print('Processing speed model for', day, hour, linha)
        bus_vel = generate_vel_model(day, hour, routes_gdf, new_bus_gdf, linha)
        mean_df_ida, mean_df_volta = interpolate_mean_route(bus_vel)
        mean_df_ida['route'] = 'out'
        mean_df_volta['route'] = 'back'
        mean_df_ida['linha'] = linha
        mean_df_volta['linha'] = linha
        mean_df_ida['day_hour'] = hour
        mean_df_volta['day_hour'] = hour
        mean_df_ida['day_of_week'] = day
        mean_df_volta['day_of_week'] = day


        tuples = [tuple(x) for x in mean_df_ida.to_numpy()]
        cols = ','.join(list(mean_df_ida.columns))
        query = f"INSERT INTO bus_speed_model ({cols}) VALUES %s"
        extras.execute_values(cursor, query, tuples)

        tuples = [tuple(x) for x in mean_df_volta.to_numpy()]
        cols = ','.join(list(mean_df_volta.columns))
        query = f"INSERT INTO bus_speed_model ({cols}) VALUES %s"
        extras.execute_values(cursor, query, tuples)

        conn.commit()

conn.close()