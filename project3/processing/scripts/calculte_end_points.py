import psycopg2
import pandas as pd
import matplotlib.pyplot as plt
import folium
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from geopy.distance import geodesic
from shapely.wkt import loads
from shapely.geometry import Point, Polygon
from sqlalchemy import create_engine
from shapely.geometry import Point, Polygon, LineString
import shapely
import warnings
warnings.filterwarnings('ignore')
import geopandas as gpd
import mplleaflet
import seaborn as sns


dias_da_semana = {
 'segunda': ['20240429', '20240506'],
 'terça': ['20240430', '20240507'],
 'quarta': ['20240424', '20240501', '20240508'],
 'quinta': ['20240425', '20240502', '20240509'],
 'sexta': ['20240426', '20240503', '20240510'],
 'sábado': ['20240427', '20240504', '20240511'],
 'domingo': ['20240428', '20240505']
}

dia = 'segunda'
def normalize_column(df, column_name):
    scaler = MinMaxScaler()
    df[[column_name]] = scaler.fit_transform(df[[column_name]])
    return df

# Função para escolher o ponto final
def choose_end_point(grids_stats):
    candidates = grids_stats[(grids_stats['median_velocidade'] == 0) & (grids_stats['median_time'] > 10) & (grids_stats['median_time'] < 17)]
    initial_point = candidates.sort_values(by='count', ascending=False).head(1)
    start_point = initial_point['centroid'].values[0]

    candidates['start_distance'] = candidates['centroid'].apply(lambda x: calculate_distance((x.y, x.x), (start_point.y, start_point.x)))

    candidates = candidates[candidates['grid_id'] != initial_point['grid_id'].values[0]]
    # Normalizar as colunas 'count' e 'start_distance'
    candidates = normalize_column(candidates, 'count')
    candidates = normalize_column(candidates, 'start_distance')
    # Ajustar a fórmula para calcular 'combined' com mais peso em 'count'
    candidates['combined'] = (0.7 * candidates['count']) + (0.3 * candidates['start_distance'])

    # Escolher o ponto final com base na coluna 'combined'
    end_point = candidates.sort_values(by='combined', ascending=False).head(1)

    return start_point, end_point['centroid'].values[0]

def stat_day_per_line(gdf, grid):
    gdf.loc[:, 'datahora'] = pd.to_datetime(gdf['datahora'])
    gdf.loc[:, 'hour'] = gdf.loc[:, 'datahora'].dt.hour
    gdf = gdf.set_geometry('geometry')
    gdf.loc[:, 'save_geometry'] = gdf.loc[:, 'geometry']
    grid = grid.set_geometry('geometry')
    grid_joined = grid.sjoin(gdf, how='inner', predicate='contains')
    aggregated = grid_joined.groupby(['grid_id', 'geometry']).agg(
        count=('geometry', 'size'),
        median_time=('hour', 'median'),
        median_velocidade=('velocidade', 'median'),
        centroid=('save_geometry', lambda x: Point(x.x.mean(), x.y.mean()))
    ).reset_index()
 
    return aggregated


def create_grid(gdf=None, bounds=None, n_cells=10, overlap=False, crs="EPSG:4326"):
    if bounds != None:
        xmin, ymin, xmax, ymax= bounds
    else:
        xmin, ymin, xmax, ymax= gdf.total_bounds

    # get cell size
    cell_size = (xmax-xmin)/n_cells
    # create the cells in a loop
    grid_cells = []
    for x0 in np.arange(xmin, xmax+cell_size, cell_size ):
        for y0 in np.arange(ymin, ymax+cell_size, cell_size):
            x1 = x0-cell_size
            y1 = y0+cell_size
            poly = shapely.geometry.box(x0, y0, x1, y1)
            #print (gdf.overlay(poly, how='intersection'))
            grid_cells.append( poly )

    cells = gpd.GeoDataFrame(grid_cells, columns=['geometry'],
                                     crs=crs)
    if overlap == True:
        cols = ['grid_id','geometry','grid_area']
        cells = cells.sjoin(gdf, how='inner').drop_duplicates('geometry')
    return cells

def calculate_distance(point1, point2):
    return geodesic(point1, point2).meters

print('Generating Grid')
rio_minx, rio_miny = -43.7955, -23.0824
rio_maxx, rio_maxy = -43.1039, -22.7448

grid = create_grid(bounds=(rio_minx, rio_miny, rio_maxx, rio_maxy), n_cells=1500)
grid = grid.reset_index(names='grid_id')

for linha_id in ('483', '864', '639', '3', '309', '774', '629', 
				  '371', '397', '100', '838', '315', '624', '388', 
				  '918', '665', '328', '497', '878', '355', '138', '606', '457', '550', 
				  '803', '917', '638', '2336', '399', '298', '867', '553', '565', '422', 
				  '756', '186012003', '292', '554', '634', '232', '415', '2803', '324', 
				  '852', '557', '759', '343', '779', '905', '108'):
    print('Processing line: ', linha_id)
    # Conectar ao banco de dados PostgreSQL
    conn = psycopg2.connect(
        dbname="postgres",
        user="postgres",
        password="postgres",
        host="localhost",
        port="5432"
    )

    between_clauses = " OR ".join([f"(datahora BETWEEN '{data} 00:00:00' AND '{data} 23:59:59')" for data in dias_da_semana[dia]])

    query = f'''select 
                    *, 
                    ST_Transform(geom::geometry, 4326) AS geometry 
                from vehicle_tracking_filtered where 
                linha = '{linha_id}' and ({between_clauses})
            '''
    
    # Load data into a GeoDataFrame
    gdf = gpd.read_postgis(query, conn, geom_col='geometry', crs='EPSG:4326')

    if gdf.shape[0] == 0:
        continue
    print('Datasoure shape: ', gdf.shape)
    grids_stats = stat_day_per_line(gdf, grid)
    grid_filtered = grids_stats[(grids_stats['count'] > grids_stats['count'].quantile(0.5))   & (grids_stats['median_time'] > 7) & (grids_stats['median_time'] < 20) & (grids_stats['median_velocidade'] > 0)]


    start_point, end_point = choose_end_point(grids_stats)




    map_center = [gdf['geometry'].y.mean(), gdf['geometry'].x.mean()]
    m = folium.Map(location=map_center, zoom_start=15)
    print('Grids Generation: ', grid_filtered.shape)
        
    # Add grid cells to the map
    for _, row in grid_filtered.iterrows():
        folium.GeoJson(row.geometry).add_to(m)
        folium.Marker(location=[row['geometry'].centroid.y, row['geometry'].centroid.x],
                        icon=folium.DivIcon(html=f'<div style="font-size: 5pt">{row["grid_id"]}</div>')).add_to(m)
        folium.Circle(location=[row.centroid.y, row.centroid.x],
                                radius=3,
                                color='red',
                                fill=True,
                                fill_color='red').add_to(m)
        
 
            
        
    # Add grid cells to the map

    folium.Circle(location=[start_point.y, start_point.x],
                            radius=30,
                            color='green',
                            fill=True,
                            fill_color='green').add_to(m)
        

    folium.Circle(location=[end_point.y, end_point.x],
                            radius=30,
                            color='purple',
                            fill=True,
                            fill_color='purple').add_to(m)


    print('Saving map: ', linha_id)
    m.save(f"maps/grid_endpoins_maps_{linha_id}.html")


    # Inserir os dados no banco de dados
    print('Saving on database: ', grid_filtered.shape)
  
    cursor = conn.cursor()
    start_point_wkt = start_point.wkt
    end_point_wkt = end_point.wkt



    cursor.execute(f"""
        INSERT INTO bus_end_points (linha, start_point, end_point)
        VALUES (%s, ST_GeomFromText(%s, 4326), ST_GeomFromText(%s, 4326))
        """, (linha_id, start_point_wkt, end_point_wkt))
    conn.commit()

    cursor.close()
    conn.close()