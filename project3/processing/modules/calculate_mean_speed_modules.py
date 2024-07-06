import psycopg2
import pandas as pd
import matplotlib.pyplot as plt
import folium
import numpy as np
from scipy.interpolate import interp1d

from shapely.wkt import loads
from geopy.distance import geodesic
from shapely.geometry import Point, Polygon
from sqlalchemy import create_engine
from shapely.geometry import Point, Polygon, LineString
import geopandas as gpd
from shapely.geometry import mapping

from pyproj import Geod

geod = Geod(ellps="WGS84")
from shapely.ops import nearest_points
from geopy.distance import geodesic
import mplleaflet
import seaborn as sns
from shapely.wkt import loads

from shapely.ops import nearest_points
from shapely.ops import split



def project_point_to_line(point, line, start, end):
    """Projetar um ponto na linha."""
    if point.within(start):
        return Point(line.coords[0])

    if point.within(end):   
        return Point(line.coords[-1])
    
    point_on_line = line.interpolate(line.project(point))
    return point_on_line

def calculate_bus_routes(row, route_line_out, route_linha_back, start, end):
    if pd.isnull(row['geometry_before']):
        return None
    else:
        percurso = calculate_percurso(row['geometry_before'], row['geometry_actual'], route_line_out, route_linha_back, start, end)
        return percurso
    
# Função para remover outliers com base na IQR
def remove_outliers_from_bus(df, column, factor=2.5):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - factor * IQR
    upper_bound = Q3 + factor * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

def calculate_bus_distance(row, route_line_out, route_linha_back, start, end, geom='geometry_actual'):
    if row['route'] == 'ida':
        return calculate_distance_along_line(route_line_out, row[geom],  start, end)
    elif row['route'] == 'volta':
        return calculate_distance_along_line(route_linha_back,  row[geom], end, start)
    elif  row['route'] == 'ponto_inicial' or row['route'] == 'ponto_final':
        return 0
    return 0





def insert_point_in_linestring(line, point):
    coords = list(line.coords)
    for i in range(len(coords) - 1):
        segment = LineString([coords[i], coords[i + 1]])
        if  point.distance(segment) < 1e-9:
            return LineString(coords[:i + 1] + [(point.x, point.y)] + coords[i + 1:])
    return line

def start_end_buffered(routes_df, linha):
    route_row = routes_df.loc[routes_df['linha'] == linha]
    start_point = route_row['start_point'].values[0]
    end_point = route_row['end_point'].values[0]
    gdf_start = gpd.GeoDataFrame(geometry=[start_point], crs="EPSG:4326").to_crs(epsg=3857)
    gdf_end = gpd.GeoDataFrame(geometry=[end_point], crs="EPSG:4326").to_crs(epsg=3857)
    start_point_buffer = gdf_start.geometry.buffer(200).to_crs(epsg=4326)
    end_point_buffer = gdf_end.geometry.buffer(200).to_crs(epsg=4326)
    return start_point_buffer[0], end_point_buffer[0]
        
def calculate_precise_distance_point_to_line(point, line):
    nearest_pt = nearest_points(point, line)[1]
    return geodesic((point.y, point.x), (nearest_pt.y, nearest_pt.x)).meters

def calculate_percurso(point1, point2, route_line_out, route_line_back, start, end):
    # Calcular as distâncias ao longo da linha
    dist_i1_out = calculate_distance_along_line(route_line_out, point1, start, end)
    dist_i2_out = calculate_distance_along_line(route_line_out, point2, start, end)

    # Calcular as distâncias ao longo da linha
    dist_i1_back = calculate_distance_along_line( route_line_back, point1, end, start)
    dist_i2_back = calculate_distance_along_line( route_line_back, point2, end, start)

    dist_proj_out =  calculate_precise_distance_point_to_line(point2, route_line_out)
    dist_proj_back = calculate_precise_distance_point_to_line(point2, route_line_back)

    if point2.within(start): 
        return 'ponto_inicial'
    elif point2.within(end): 
        return 'ponto_final'
    elif dist_proj_out - dist_proj_back > 20:
        if dist_proj_out > dist_proj_back:
            return 'volta'
        else:
            return 'ida'
    elif point1.distance(point2) < 10**(-5):
        return 'sem_movimento'
    else:
        if dist_i1_out < dist_i2_out: 
            return 'ida'
        elif dist_i1_back < dist_i2_back:
            return 'volta'
    
        
    return 'sem_movimento'




def calculate_percurso_new(point1, point2, route_line_out, route_line_back, start, end):
    # Calcular as distâncias ao longo da linha
    dist_i1_out = calculate_distance_along_line(route_line_out, point1, start, end)
    dist_i2_out = calculate_distance_along_line(route_line_out, point2, start, end)

    # Calcular as distâncias ao longo da linha
    dist_i1_back = calculate_distance_along_line( route_line_back, point1, end, start)
    dist_i2_back = calculate_distance_along_line( route_line_back, point2, end, start)

    dist_proj_out =  calculate_precise_distance_point_to_line(point2, route_line_out)
    dist_proj_back = calculate_precise_distance_point_to_line(point2, route_line_back)

    if dist_proj_out - dist_proj_back > 20:
        if dist_proj_out > dist_proj_back:
            return 'volta'
        else:
            return 'ida'
    elif point1.distance(point2) < 10**(-5):
        return 'sem_movimento'
    elif point2.within(start): 
        return 'ponto_inicial'
    elif point2.within(end): 
        return 'ponto_final'
    else:
        if dist_i1_out < dist_i2_out: 
            return 'ida'
        elif dist_i1_back < dist_i2_back:
            return 'volta'
    
        
    return 'sem_movimento'



def calculate_distance_along_line(line, point1, start, end):
    # Projetando os pontos na linha para encontrar A e B
    if point1.within(start):
        return 0

    if point1.within(end):   
        return geod.geometry_length(line)


    proj_point1 = project_point_to_line(point1, line, start, end)
    # proj_point2 = project_point_to_line(point2, line)

    line_c = insert_point_in_linestring(line, proj_point1)
    # line_c = insert_point_in_linestring(line_c, proj_point2)
   

    # Criando LineStrings parciais
    split_line1 = split(line_c, proj_point1)
    # split_line2 = split(line_c, proj_point2)

    # distance_along_line = geod.geometry_length(split_line2.geoms[1]) - geod.geometry_length(split_line1.geoms[1])
    return geod.geometry_length(split_line1.geoms[0])



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
def get_bus_data(conn, linha):
    # Selecionar dados da tabela bus_routes para a linha específica
    query = f'''select 
                    *, 
                    ST_Transform(geom::geometry, 4326) AS geometry 
                from vehicle_tracking_filtered where 
                linha = '{linha}'
            '''
        
    # Carregar dados em um GeoDataFrame
    gdf = gpd.read_postgis(query, conn, geom_col='geometry', crs='EPSG:4326')

    return gdf

def remove_time_outliers(bus_gdf):
    # Convert to datetime
    bus_gdf['datahora'] = pd.to_datetime(bus_gdf['datahora'])
    bus_gdf['datahoraservidor'] = pd.to_datetime(bus_gdf['datahoraservidor'])
    
    # Calculate the time difference in seconds
    bus_gdf['time_diff'] = (bus_gdf['datahoraservidor'] - bus_gdf['datahora']).dt.total_seconds()
    
    # Calculate mean and standard deviation
    mean_diff = bus_gdf['time_diff'].mean()
    std_diff = bus_gdf['time_diff'].std()
    
    # Filter out the outliers
    filtered_gdf = bus_gdf[(bus_gdf['time_diff'] >= (mean_diff - 3 * std_diff)) & (bus_gdf['time_diff'] <= (mean_diff + 3 * std_diff))]
    
    return filtered_gdf


def remove_far_away_points(linha, routes_df, bus_gdf, datahora_column='datahora'):
    # Select the appropriate route based on the day of the week

    semana = {
        'Segunda': ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'],
        'Sabado': ['Saturday'],
        'Domingo': ['Sunday']
    }
    bus_gdf_filtered = pd.DataFrame()
    # bus_gdf_outside_buffer = pd.DataFrame()
    for dia, valores in semana.items():
        bus_gdf_dia = bus_gdf[bus_gdf[datahora_column].dt.day_name().isin(valores)]

        route_row  = routes_df[(routes_df['linha'] == linha) & (routes_df['week_day'] == dia)]

        route_linha_out = route_row['route_line_out'].values[0]
        route_linha_back = route_row['route_line_back'].values[0]
        start_point = route_row['start_point'].values[0]
        end_point = route_row['end_point'].values[0]
        
        # Create GeoDataFrames for the routes and reproject to a metric CRS
        gdf_out = gpd.GeoDataFrame(geometry=[route_linha_out], crs="EPSG:4326").to_crs(epsg=3857)
        gdf_back = gpd.GeoDataFrame(geometry=[route_linha_back], crs="EPSG:4326").to_crs(epsg=3857)
        gdf_start = gpd.GeoDataFrame(geometry=[start_point], crs="EPSG:4326").to_crs(epsg=3857)
        gdf_end = gpd.GeoDataFrame(geometry=[end_point], crs="EPSG:4326").to_crs(epsg=3857)
        
        
        
        # Reproject bus_gdf to metric CRS for spatial operations
        bus_gdf_metric = bus_gdf_dia.to_crs(epsg=3857)
        
        # Filter bus_gdf to remove points further than 30 meters from both route lines
        within_route_buffer = bus_gdf_metric.geometry.within(gdf_out.geometry.buffer(30).unary_union) | bus_gdf_metric.geometry.within(gdf_back.geometry.buffer(30).unary_union)
        
        # Filter bus_gdf to include points within 200 meters of start or end points
        within_points_buffer = bus_gdf_metric.geometry.within(gdf_start.geometry.buffer(200).unary_union) | bus_gdf_metric.geometry.within(gdf_end.geometry.buffer(200).unary_union)
        
        # Combine both conditions
        bus_gdf_filtered = pd.concat([bus_gdf_filtered, bus_gdf_metric[within_route_buffer | within_points_buffer].to_crs(epsg=4326)])

    return bus_gdf_filtered

import warnings
warnings.filterwarnings('ignore')
def generate_vel_model(day_of_week, hora, routes_gdf, bus_gdf, linha):
    ordens = bus_gdf['ordem'].unique()
    cat_horas = {
        'rush_manha': [8, 9, 10, 11],
        'tarde': [12, 13, 14, 15],
        'rush_noite': [16, 17, 18, 19],
        'noite': [20, 21, 22, 23]
    }
    semana = {
        'Monday': 'Segunda',
        'Tuesday': 'Segunda',
        'Wednesday': 'Segunda',
        'Thursday': 'Segunda',
        'Friday': 'Segunda',
        'Saturday': 'Sabado',
        'Sunday': 'Domingo'
    }
    rota_volta = routes_gdf.loc[(routes_gdf['linha'] == linha) & (routes_gdf['week_day'] == semana[day_of_week]), 'route_line_back'].values[0]
    rota_ida = routes_gdf.loc[(routes_gdf['linha'] == linha) & (routes_gdf['week_day'] == semana[day_of_week]), 'route_line_out'].values[0]
    start, end = start_end_buffered(routes_gdf, linha)
    bus_line_vel = pd.DataFrame()
    for ord in ordens:
        try:
            bus_sample = bus_gdf[
                (bus_gdf['ordem'] == ord) & 
                (bus_gdf['datahora'].dt.day_name() == day_of_week) &
                (bus_gdf['datahora'].dt.hour.isin(cat_horas[hora]))
            ].sort_values('datahora').reset_index(drop=True)
            if bus_sample.shape[0] < 4:
                continue

            before = bus_sample.loc[:, ['ordem', 'datahora', 'linha', 'geometry']].shift(1)
            actual = bus_sample.loc[:, ['ordem', 'datahora', 'linha', 'geometry']]

            before.columns = [f'{c}_before' for c in before.columns]
            actual.columns = [f'{c}_actual' for c in actual.columns]

            bus_samples_concat = pd.concat([actual, before], axis=1)
    
            ## Remove outliers
            bus_samples_concat['datahora_actual'] = pd.to_datetime(bus_samples_concat['datahora_actual'])
            bus_samples_concat['datahora_before'] = pd.to_datetime(bus_samples_concat['datahora_before'])
            # Converter as colunas geometry para objetos Shapely
            bus_samples_concat['geometry_actual'] = bus_samples_concat['geometry_actual']
            bus_samples_concat['geometry_before'] = bus_samples_concat['geometry_before']
            # Calcular a diferença de tempo em segundos
            bus_samples_concat['time_diff'] = (bus_samples_concat['datahora_actual'] - bus_samples_concat['datahora_before']).dt.total_seconds()
            # Calcular a diferença de distância em metros
            bus_samples_concat['distance_diff'] = bus_samples_concat.apply(lambda row: row['geometry_actual'].distance(row['geometry_before']), axis=1)

            # Remover outliers com base na diferença de tempo
            bus_samples_concat = remove_outliers_from_bus(bus_samples_concat, 'time_diff', factor=1.5)
            # Remover outliers com base na diferença de distância
            bus_samples_concat = remove_outliers_from_bus(bus_samples_concat, 'distance_diff', factor=1.5)

            bus_samples_concat = bus_samples_concat.reset_index()

            bus_samples_concat['route'] = bus_samples_concat.apply(lambda row: calculate_bus_routes(row, rota_ida, rota_volta, start, end), axis=1)
            bus_samples_concat['route'] = bus_samples_concat['route'].replace("sem_movimento", pd.NA)
            bus_samples_concat['route']  = bus_samples_concat['route'].fillna(method='ffill').fillna(method='bfill')

            bus_samples_concat['rel_pos'] =  bus_samples_concat.apply(lambda row: calculate_bus_distance(row, rota_ida, rota_volta, start, end), axis=1)

            bus_samples_concat['pos_diff'] = bus_samples_concat.groupby('route')['rel_pos'].diff()
            bus_samples_concat['time_diff'] = bus_samples_concat.groupby('route')['datahora_actual'].diff().dt.total_seconds() / 60  # Converter tempo minutos
            # Calcular a velocidade
            bus_samples_concat['velocidade'] = bus_samples_concat['pos_diff'] / bus_samples_concat['time_diff']


            # Remover as linhas onde a velocidade não pode ser calculada (primeiros elementos de cada grupo)
            bus_samples_concat = bus_samples_concat.dropna(subset=['velocidade'])

            bus_samples_concat = bus_samples_concat[bus_samples_concat['velocidade'] > 0]


            bus_line_vel = pd.concat([bus_line_vel, bus_samples_concat])
        except Exception as error:
            print(error)
            continue
    
    return bus_line_vel

def interpolate_data(x, y, new_x):
    f = interp1d(x, y, kind='linear', bounds_error=False, fill_value=np.nan)
    return f(new_x)

# Função para calcular a média móvel
def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

def interpolate_mean_route(bus_line_vel):
    combined_df_ida = pd.DataFrame()
    combined_df_volta = pd.DataFrame()
    ordens = bus_line_vel['ordem_actual'].unique()
    max_rel_ida = bus_line_vel.loc[bus_line_vel['route'] == 'ida', 'rel_pos'].max()
    max_rel_volta = bus_line_vel.loc[bus_line_vel['route'] == 'volta', 'rel_pos'].max()
    min_rel_ida = bus_line_vel.loc[bus_line_vel['route'] == 'ida', 'rel_pos'].min()
    min_rel_volta = bus_line_vel.loc[bus_line_vel['route'] == 'volta', 'rel_pos'].min()
    window_size = 10
    num_ida = max_rel_ida/20
    num_volta = max_rel_volta/20

    common_rel_pos_ida = np.linspace(min_rel_ida, max_rel_ida, round(num_ida))
    common_rel_pos_volta = np.linspace(min_rel_volta, max_rel_volta, round(num_volta))

    for ord in ordens:
        df_ida = bus_line_vel[
            (bus_line_vel['ordem_actual'] == ord) &
            (bus_line_vel['route'] == 'ida')
        ].sort_values(by='rel_pos')

        df_volta = bus_line_vel[
            (bus_line_vel['ordem_actual'] == ord) &
            (bus_line_vel['route'] == 'volta')
        ].sort_values(by='rel_pos')

        if df_ida.shape[0] != 0:
            interpolated_out1 = interpolate_data(df_ida['rel_pos'], df_ida['velocidade'], common_rel_pos_ida)
            ma_interpolated_out1 = moving_average(interpolated_out1, window_size)
           

            interpolated_df_ida = pd.DataFrame({
                'rel_pos': common_rel_pos_ida[:len(ma_interpolated_out1)],
                'speed': ma_interpolated_out1
            })
            combined_df_ida = pd.concat([combined_df_ida, interpolated_df_ida], ignore_index=True)
        
        if df_volta.shape[0] != 0:
            interpolated_out2 = interpolate_data(df_volta['rel_pos'], df_volta['velocidade'], common_rel_pos_volta)
            ma_interpolated_out2 = moving_average(interpolated_out2, window_size)
            interpolated_df_volta = pd.DataFrame({
                'rel_pos': common_rel_pos_volta[:len(ma_interpolated_out2)],
                'speed': ma_interpolated_out2
            })
            combined_df_volta = pd.concat([combined_df_volta, interpolated_df_volta], ignore_index=True)
            
    mean_df_ida = combined_df_ida.groupby('rel_pos').mean().reset_index()
    mean_df_volta = combined_df_volta.groupby('rel_pos').mean().reset_index()
    return mean_df_ida, mean_df_volta