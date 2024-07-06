import psycopg2
import pandas as pd
import matplotlib.pyplot as plt
import folium
import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import quad
from scipy.integrate import cumtrapz

from shapely.wkt import loads
from geopy.distance import geodesic
from shapely.geometry import Point, Polygon
from sqlalchemy import create_engine
from shapely.geometry import Point, Polygon, LineString
import geopandas as gpd
from pyproj import Geod
geod = Geod(ellps="WGS84")
from shapely.ops import nearest_points
from geopy.distance import geodesic
import mplleaflet
import seaborn as sns
from shapely.wkt import loads
from shapely.ops import nearest_points
from shapely.ops import split
from scipy.interpolate import interp1d
from scipy.integrate import quad
from calculate_mean_speed_modules import *


class Predict():
    
    def find_nearest_datetime(self, row, df, i):
        nearest = df.loc[df['datahora'] < row['filename_data']].sort_values(by='datahora', ascending=False).iloc[i]
        return nearest
    
        
    def substituir_proximidade(self, row, df):
        index = int(row.name)
        if index > 0 and index < len(df) - 1:
            if row['route'] == 'ida' and (df.at[index-1, 'route'] == 'volta' and df.at[index+1, 'route'] == 'volta'):
                return 'volta'
            elif row['route'] == 'volta' and (df.at[index-1, 'route'] == 'ida' and df.at[index+1, 'route'] == 'ida'):
                return 'ida'
        return row['route']

    def filter_route(self, routes_gdf, linha, day_of_week):
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

        return rota_ida, rota_volta


    def filter_speed(self, bus_speed_df, hour, day):
        if hour in [8, 9, 10, 11]:
            cat_hour =  'rush_manha'
        elif hour in [12, 13, 14, 15]:
            cat_hour = 'tarde'
        elif hour in [16, 17, 18, 19]:
            cat_hour = 'rush_noite'
        elif hour in [20, 21, 22, 23]:
            cat_hour = 'noite'

        if day in ['Tuesday', 'Wednesday', 'Thursday', 'Friday']:
            cat_day='Monday'
        else:
            cat_day = day

        if cat_day  not in bus_speed_df['day_of_week'].unique():
            cat_day = 'Monday'

        bus_speed_ida = bus_speed_df[
            (bus_speed_df['route'] == 'out') & 
            (bus_speed_df['day_hour'] == cat_hour) &
            (bus_speed_df['day_of_week'] == cat_day)
        ]
        bus_speed_volta = bus_speed_df[
            (bus_speed_df['route'] == 'back') & 
            (bus_speed_df['day_hour'] == cat_hour)&
            (bus_speed_df['day_of_week'] == cat_day)
        ]

        return bus_speed_ida, bus_speed_volta


class PredictTime(Predict):
    # Interpolação da velocidade ao longo da posição relativa
    def calcular_tempo_passado(self, row, bus_speed_ida, bus_speed_volta):
        interp_func_volta = interp1d(bus_speed_volta['rel_pos'], bus_speed_volta['speed'], kind='linear', fill_value='extrapolate')
        interp_func_ida = interp1d(bus_speed_ida['rel_pos'], bus_speed_ida['speed'], kind='linear', fill_value='extrapolate')

        if row['route'] == 'ida':
            interp_func = interp_func_ida
        elif row['route'] == 'volta':
            interp_func = interp_func_volta
        elif row['route'] == 'ponto_inicial':
            return 0
        elif row['route'] == 'ponto_final':
            return 0

        # Função de integrando para calcular o tempo
        def integrand(pos):
            return 1 / interp_func(pos)
        
        # Calcular a integral da função 1/velocidade entre as duas posições
        tempo_passado, _ = quad(integrand,  row['rel_pos_before'], row['rel_pos_actual'])
        
        return tempo_passado


    def add_column_route(self, df, rota_ida, rota_volta, start, end):
        df['route'] = df.apply(lambda row: calculate_bus_routes(row, rota_ida, rota_volta, start, end), axis=1)
        df['route'] = df['route'].replace("sem_movimento", pd.NA)
        df['route']  = df['route'].fillna(method='ffill').fillna(method='bfill')
        df['route'] = df.apply(lambda x: self.substituir_proximidade(x, df), axis=1)
        return df


    def generate_actual_before_df(self, df, df_reference, cols, common_cols):
        before = df.loc[:, cols].shift(1)
        actual = df.loc[:, cols + common_cols]
        before.columns = [f'{c}_before' for c in before.columns] 
        actual.columns = [f'{c}_actual' for c in actual.columns if c not in common_cols] + common_cols
        bus_samples_concat = pd.concat([actual, before], axis=1)
        bus_samples_concat = bus_samples_concat.reset_index(drop=True)
        reference_item = self.find_nearest_datetime(bus_samples_concat.iloc[0], df_reference, 0)


        bus_samples_concat.loc[0, 'latitude_before'] = reference_item['latitude']
        bus_samples_concat.loc[0, 'longitude_before'] = reference_item['longitude']
        bus_samples_concat.loc[0, 'geometry_before'] = Point(reference_item['longitude'],  reference_item['latitude'])
        bus_samples_concat.loc[0, 'datahora_epoch_before'] = reference_item['datahora_epoch']*1000

        return bus_samples_concat
    
        
    def add_column_rel_pos(self, df,rota_ida, rota_volta, start, end):
        ## Calculate REL_POS
        df['rel_pos_actual'] = df.apply(lambda row: calculate_bus_distance(row, rota_ida, rota_volta, start, end), axis=1)
        df['rel_pos_before'] = df.apply(lambda row: calculate_bus_distance(row, rota_ida, rota_volta, start, end, geom='geometry_before'), axis=1)
        return df

    def add_column_time_passed(self, df, bus_speed_ida, bus_speed_volta, handle_end=False):
        ## Calculate time_passed
        df['time_passed'] = df.apply(lambda row: self.calcular_tempo_passado(row, bus_speed_ida, bus_speed_volta), axis=1).astype(float)
        df.loc[df['time_passed'] < 0, 'time_passed'] = 0

        median_time_passed = df['time_passed'].median()

        if handle_end:
            df.loc[(df['route'] == 'ponto_inicial') | (df['route'] == 'ponto_final'), 'time_passed'] = median_time_passed

        for i, row in df.iterrows():
            if i == 0:
                continue
            df.loc[i, 'datahora_epoch_before'] = df.loc[i-1, 'datahora_epoch_before'] + df.loc[i-1, 'time_passed']*60*1000

        df['datahora_epoch_actual'] = df['datahora_epoch_before'] + df['time_passed']*60*1000
        return df


    def predict_epoch(self, filename, ordem, routes_gdf, bus_speed_df, bus_data_latlong, bus_data_train, linha, handle_end=False):
        try:
            day_of_week = bus_data_latlong.loc[bus_data_latlong['filename'] == filename, 'filename_data'].dt.day_name().iloc[0]
            day = bus_data_latlong.loc[bus_data_latlong['filename'] == filename, 'filename_data'].dt.day.iloc[0]

            ## Filter samples
            sample_train = bus_data_train[
                (bus_data_train['ordem'] == ordem) & 
                (bus_data_train['datahora'].dt.day == day) 
            ].rename(columns={'geom': 'geometry'})

            sample_latlong = bus_data_latlong[
                (bus_data_latlong['ordem'] == ordem) &
                (bus_data_latlong['filename'] == filename) 
                
            ].sort_values(by='id').rename(columns={'geom': 'geometry'})

            sample_latlong = sample_latlong.set_geometry('geometry')

            ## Get bus info
            rota_ida, rota_volta = self.filter_route(routes_gdf, linha, day_of_week)
            start, end = start_end_buffered(routes_gdf, linha)
            hour = bus_data_latlong.loc[bus_data_latlong['filename'] == filename, 'filename_data'].dt.hour.iloc[0]
            bus_speed_ida, bus_speed_volta = self.filter_speed(bus_speed_df, hour, day_of_week)

            sample_latlong = remove_far_away_points(linha, routes_gdf, sample_latlong, datahora_column='filename_data')
            sample_latlong = sample_latlong.reset_index(drop=True)


            bus_samples_concat = self.generate_actual_before_df(
                sample_latlong, 
                sample_train, 
                cols=['id', 'ordem', 'latitude', 'longitude', 'geometry'], 
                common_cols=['filename_data', 'datahora_epoch_resposta',  'filename']
            )

            bus_samples_concat = self.add_column_route(bus_samples_concat, rota_ida, rota_volta, start, end)
            bus_samples_concat = self.add_column_rel_pos(bus_samples_concat, rota_ida, rota_volta, start, end)
            bus_samples_concat = self.add_column_time_passed(bus_samples_concat, bus_speed_ida, bus_speed_volta, handle_end)

            return bus_samples_concat.loc[:, ['id_actual', 'ordem_actual', 'datahora_epoch_actual', 'datahora_epoch_resposta', 'filename', 'route']]
        except:
            return pd.DataFrame()



from scipy.integrate import quad
from scipy.interpolate import interp1d
class PredictLatLong(Predict):

    def generate_actual_before_df(self, df, df_reference, cols, common_cols):
        before = df.loc[:, cols].shift(1)
        actual = df.loc[:, cols + common_cols]
        before.columns = [f'{c}_before' for c in before.columns] 
        actual.columns = [f'{c}_actual' for c in actual.columns if c not in common_cols] + common_cols
        bus_samples_concat = pd.concat([actual, before], axis=1)
        bus_samples_concat = bus_samples_concat.reset_index(drop=True)
        reference_item_1 = self.find_nearest_datetime(bus_samples_concat.iloc[0], df_reference, 0)
        reference_item_2 = self.find_nearest_datetime(bus_samples_concat.iloc[0], df_reference, 1)
        
        for i  in range(df_reference.shape[0]):
            if reference_item_2['geometry'].values[0] != reference_item_1['geometry'].values[0]:
                break
            reference_item_2 = self.find_nearest_datetime(bus_samples_concat.iloc[0], df_reference, i)

        bus_samples_concat.loc[0, 'geometry_before'] = Point(reference_item_1['longitude'],  reference_item_1['latitude'])

        bus_samples_concat.loc[0, 'datahora_epoch_before'] = reference_item_1['datahora_epoch']*1000

        bus_samples_concat.loc[0, 'geometry_past_before'] = Point(reference_item_2['longitude'],  reference_item_2['latitude'])

        return bus_samples_concat
    
    def tempo_para_distancia(self, distancia, interp_func):
        def integrand(d):
            return 1 / interp_func(d)
        tempo_total, _ = quad(integrand, 0, distancia)
        return tempo_total

    def calcular_distancia(self, row, bus_speed_ida, bus_speed_volta):
     

        if row['route'] == 'ida' or row['route'] == 'ida_corrigida':
            interp_func_ida_rel = interp1d(bus_speed_ida['rel_pos'], bus_speed_ida['speed'], kind='linear', fill_value='extrapolate')
            times_ida = cumtrapz(1 / interp_func_ida_rel.y, interp_func_ida_rel.x, initial=0)
            interp_func_ida_time = interp1d(times_ida, interp_func_ida_rel.y, kind='linear', fill_value='extrapolate')
            interp_func = interp_func_ida_time

        elif row['route'] == 'volta' or row['route'] == 'volta_corrigida':
            interp_func_volta_rel = interp1d(bus_speed_volta['rel_pos'], bus_speed_volta['speed'], kind='linear', fill_value='extrapolate')
            times_volta = cumtrapz(1 / interp_func_volta_rel.y, interp_func_volta_rel.x, initial=0)
            interp_func_volta_time = interp1d(times_volta, interp_func_volta_rel.y, kind='linear', fill_value='extrapolate')
            interp_func = interp_func_volta_time
        elif row['route'] == 'ponto_inicial':
            return 0
        elif row['route'] == 'ponto_final':
            return 0
        else:
            return 0

        # Função de integrando para calcular o tempo
        def integrand(pos):
            return interp_func(pos)
        
        time1 = self.tempo_para_distancia(row['rel_pos_before'], interp_func)
       
        time2 = time1 + row['time_passed']/1000/60
        # Calcular a integral da função 1/velocidade entre as duas posições
        distancia, _ = quad(integrand,  time1, time2)
        
        return distancia
    
    def find_lat_long(self, row, rota_ida, rota_volta, start, end):
        if row['route'] == 'ida' or  row['route'] == 'ida_corrigida':
            norm_loc = row['rel_pos_actual']/geod.geometry_length(rota_ida)
            point = rota_ida.interpolate(norm_loc, normalized=True)

        elif row['route'] == 'volta' or row['route'] == 'volta_corrigida':
            norm_loc = row['rel_pos_actual']/geod.geometry_length(rota_volta)
            point = rota_volta.interpolate(norm_loc, normalized=True)
        elif row['route'] == 'ponto_inicial':
            point = start.centroid
        elif row['route'] == 'ponto_final':
            point = end.centroid
        else:
            return row['geometry_before']
        return point

    def add_column_geometry(self, df,  rota_ida, rota_volta, start, end, bus_speed_ida, bus_speed_volta, handle_end=False):
        
        for i, row in df.iterrows():
            if i == 0:
                route =  calculate_percurso(df['geometry_before'].loc[i], df['geometry_past_before'].loc[i], rota_ida, rota_volta, start, end)
                df.loc[i, 'route'] = route
                rel_pos = calculate_bus_distance(df.loc[i], rota_ida, rota_volta, start, end, geom='geometry_before')
                df.loc[i, 'rel_pos_before'] = rel_pos

                df.loc[i, 'distance']  = self.calcular_distancia(df.loc[i], bus_speed_ida, bus_speed_volta)     
                df.loc[i, 'rel_pos_actual'] = df.loc[i, 'rel_pos_before'] + df.loc[i, 'distance'] 
                df.loc[i, 'geometry_actual'] = self.find_lat_long(df.loc[i], rota_ida, rota_volta, start, end)
            else:
                df.loc[i, 'rel_pos_before'] =  df.loc[i-1, 'rel_pos_actual']
                df.loc[i, 'geometry_before'] =  df.loc[i-1, 'geometry_actual']
                df.loc[i, 'geometry_past_before'] =  df.loc[i-1, 'geometry_before']
                route =  calculate_percurso(df['geometry_before'].loc[i], df['geometry_past_before'].loc[i], rota_ida, rota_volta, start, end)


                if route == 'ponto_inicial' and  df.loc[i-1, 'route'] == 'volta' and handle_end:
                    df.loc[i, 'route'] = 'ida_corrigida'            
                    df.loc[i, 'rel_pos_before'] = 0
                    df.loc[i, 'rel_pos_actual'] = 10
                    df.loc[i, 'geometry_before'] =  Point(rota_ida.coords[0])
                    df.loc[i, 'geometry_actual'] = self.find_lat_long(df.loc[i], rota_ida, rota_volta, start, end)
                elif route == 'ponto_inicial' and  df.loc[i-1, 'route'] == 'ida_corrigida' and handle_end:
                    df.loc[i, 'route'] = 'ida_corrigida'            
                    df.loc[i, 'distance']  = self.calcular_distancia(df.loc[i], bus_speed_ida, bus_speed_volta)     
                    df.loc[i, 'rel_pos_actual'] = df.loc[i, 'rel_pos_before'] + df.loc[i, 'distance'] 
                    df.loc[i, 'geometry_actual'] = self.find_lat_long(df.loc[i], rota_ida, rota_volta, start, end)
                elif  route == 'ponto_final' and  df.loc[i-1, 'route'] == 'ida' and handle_end:
                    df.loc[i, 'route'] = 'volta_corrigida'            
                    df.loc[i, 'rel_pos_before'] = 0
                    df.loc[i, 'rel_pos_actual'] = 10
                    df.loc[i, 'geometry_before'] = Point(rota_volta.coords[0])
                    df.loc[i, 'geometry_actual'] = self.find_lat_long(df.loc[i], rota_ida, rota_volta, start, end)
                elif  route == 'ponto_final' and  df.loc[i-1, 'route'] == 'volta_corrigida' and handle_end:
                    df.loc[i, 'route'] = 'volta_corrigida'  
                    df.loc[i, 'distance']  = self.calcular_distancia(df.loc[i], bus_speed_ida, bus_speed_volta)     
                    df.loc[i, 'rel_pos_actual'] = df.loc[i, 'rel_pos_before'] + df.loc[i, 'distance'] 
                    df.loc[i, 'geometry_actual'] = self.find_lat_long(df.loc[i], rota_ida, rota_volta, start, end)
                else:
                    df.loc[i, 'route'] = route            
                    df.loc[i, 'distance']  = self.calcular_distancia(df.loc[i], bus_speed_ida, bus_speed_volta)     
                    df.loc[i, 'rel_pos_actual'] = df.loc[i, 'rel_pos_before'] + df.loc[i, 'distance'] 
                    df.loc[i, 'geometry_actual'] = self.find_lat_long(df.loc[i], rota_ida, rota_volta, start, end)
        return df
    
    def predict_lat_long(self, filename, ordem, routes_gdf, bus_speed_df, bus_data_datahora, bus_data_train, linha, handle_end=False):
        try:
            day_of_week = bus_data_datahora.loc[bus_data_datahora['filename'] == filename, 'filename_data'].dt.day_name().iloc[0]
            day = bus_data_datahora.loc[bus_data_datahora['filename'] == filename, 'filename_data'].dt.day.iloc[0]

            sample_train = bus_data_train[
                (bus_data_train['ordem'] == ordem) & 
                (bus_data_train['datahora'].dt.day == day) 
            ].rename(columns={'geom': 'geometry'})

            sample_datahora = bus_data_datahora[
                (bus_data_datahora['ordem'] == ordem) &
                (bus_data_datahora['filename'] == filename) 

            ].sort_values(by='id').rename(columns={'geom': 'geometry'})
            rota_ida, rota_volta = self.filter_route(routes_gdf, linha, day_of_week)
            start, end = start_end_buffered(routes_gdf, linha)
            hour = bus_data_datahora.loc[bus_data_datahora['filename'] == filename, 'filename_data'].dt.hour.iloc[0]
            bus_speed_ida, bus_speed_volta =  self.filter_speed(bus_speed_df, hour, day_of_week)

            bus_samples_concat = self.generate_actual_before_df(
                sample_datahora, 
                sample_train, 
                cols=['id', 'ordem', 'datahora_epoch'], 
                common_cols=['filename_data', 'latitude_resposta', 'longitude_resposta',  'filename']
            )

            bus_samples_concat['time_passed'] = bus_samples_concat['datahora_epoch_actual'] - bus_samples_concat['datahora_epoch_before']

            bus_samples_concat = bus_samples_concat[bus_samples_concat['time_passed'] > 0]
            bus_samples_concat = bus_samples_concat.reset_index(drop=True)
            bus_samples_concat = self.add_column_geometry(bus_samples_concat,  rota_ida, rota_volta, start, end, bus_speed_ida, bus_speed_volta, handle_end)
            bus_samples_concat['latitude_actual'] = bus_samples_concat['geometry_actual'].map(lambda row: row.y)
            bus_samples_concat['longitude_actual'] = bus_samples_concat['geometry_actual'].map(lambda row: row.x)
            return bus_samples_concat
        except Exception as error:
            print(error)
            return pd.DataFrame()