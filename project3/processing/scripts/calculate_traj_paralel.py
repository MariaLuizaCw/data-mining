import psycopg2
import pandas as pd
import matplotlib.pyplot as plt
import folium
import numpy as np

from shapely.wkt import loads
from shapely.geometry import Point, Polygon
from sqlalchemy import create_engine
from shapely.geometry import Point, Polygon, LineString
import shapely
from concurrent.futures import ThreadPoolExecutor
import warnings
warnings.filterwarnings('ignore')
import geopandas as gpd
# Função para suavizar a rota usando splines
from scipy.interpolate import splprep, splev
import mplleaflet
import seaborn as sns
from geopy.distance import geodesic
from pyproj import Geod

geod = Geod(ellps="WGS84")

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



# Função para calcular a distância entre dois pontos
def order_centroids(df, start_point, end_point, max_iter=200, max_dist=200, max_travel_dist=5000):
    # Encontrar o ponto mais próximo do ponto inicial
    df['start_distance'] = df['centroid'].apply(lambda x: calculate_distance((x.y, x.x), start_point))
    start_df = df.sort_values(by='start_distance').iloc[0]
    df['end_distance'] = df['centroid'].apply(lambda x: calculate_distance((x.y, x.x), end_point))

    if calculate_distance((start_df['centroid'].y, start_df['centroid'].x), start_point) >= max_dist:
        return ([], 0) 
    
    # Ordenar os pontos a partir do ponto inicial
    ordered_centroids = [(start_df['centroid'].y, start_df['centroid'].x)]
    current_point = start_df
    passed_points = {current_point['grid_id']}
    
    while max_iter:
        max_iter -= 1
        next_points = df[(df['datahora'] > current_point['datahora']) & 
                         (~df['grid_id'].isin(passed_points))].sort_values(by='datahora')
        if next_points.empty:
            return (ordered_centroids, 0)
       
        next_point = next_points.iloc[0]

        distance = calculate_distance((current_point['centroid'].y, current_point['centroid'].x), 
                                    (next_point['centroid'].y, next_point['centroid'].x))

        if distance > max_travel_dist:
            return (ordered_centroids, 0)
        if calculate_distance((next_point['centroid'].y, next_point['centroid'].x), end_point) <= max_dist:
            ordered_centroids.append((next_point['centroid'].y, next_point['centroid'].x))
            return (ordered_centroids, 1)

        passed_points.add(next_point['grid_id'])
        ordered_centroids.append((next_point['centroid'].y, next_point['centroid'].x))
        current_point = next_point

    return (ordered_centroids, 0)
    


def calculate_distance(point1, point2):
    return geodesic(point1, point2).meters


# Função para traçar segmentos de reta para um ônibus específico
def trace_routes(df, ordem_id, start_point, end_point):
    bus_df = df[df['ordem'] == ordem_id]
    ordered_centroids = order_centroids(bus_df, start_point, end_point)
    
    return ordered_centroids

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


# Função para criar um GeoDataFrame de rota como uma única linha
def create_route_gdf(route, complete, route_id):
    line = LineString([(point[1], point[0]) for point in route])
    return gpd.GeoDataFrame({'route_id': [route_id], 'geometry': [line], 'complete': [complete]})

def get_best_route(all_routes, df_grid):
    # Criar GeoDataFrame de todas as rotas
    all_routes_gdf = pd.concat([create_route_gdf(route['rota'], route['complete'], idx) for idx, route in all_routes.items()], ignore_index=True)
    grids_gdf = gpd.GeoDataFrame(df_grid, geometry='geometry')
    # Interseção usando overlay
    intersection_gdf = gpd.overlay(all_routes_gdf, grids_gdf, how='intersection')

    # Contar interseções por rota
    intersection_counts = intersection_gdf.groupby('route_id')['grid_id'].nunique()

    # Calcular o comprimento das linhas das rotas
    all_routes_gdf = all_routes_gdf.set_index('route_id')
    route_lengths = all_routes_gdf['geometry'].map(lambda x: geod.geometry_length(x))
    # Calcular a razão interseções/comprimento para cada rota
    ratio = intersection_counts.divide(route_lengths)
    all_routes_gdf['route_order'] = ratio
    all_routes_gdf = all_routes_gdf.sort_values(by=['complete', 'route_order'], ascending=False)
    route_id = all_routes_gdf.iloc[0].name
    print(route_id)

    return all_routes[route_id]['rota']



dias_da_semana = {
 'Segunda': ['20240429', '20240506'],
 'Terça': ['20240430', '20240507'],
 'Quarta': ['20240424', '20240501', '20240508'],
 'Quinta': ['20240425', '20240502', '20240509'],
 'Sexta': ['20240426', '20240503', '20240510'],
 'Sabado': ['20240427', '20240504', '20240511'],
 'Domingo': ['20240428', '20240505']
}



def process_bus_lines(bus_lines, dia, conn, grid, ends_df, cursor):

    cursor = conn.cursor()


    for linha_id in bus_lines:
        print(f'Processing line: {linha_id}', f'Processing day: {dia}')
 
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
        print('Datasoure shape: ', gdf.shape, linha_id)
        grids_stats = stat_day_per_line(gdf, grid)
        grid_filtered = grids_stats[(grids_stats['count'] > grids_stats['count'].quantile(0.5))   & (grids_stats['median_time'] > 9) & (grids_stats['median_time'] < 17) & (grids_stats['median_velocidade'] > 0)]
   
        print('Get End Points', linha_id)
        # Get end points
        line_ends = ends_df[ends_df['linha'] == linha_id]
        start_point = line_ends['start_point'].values[0]
        end_point = line_ends['end_point'].values[0]
       
        start_point_df = gpd.sjoin(grid, gpd.GeoDataFrame([start_point], columns=['geometry'], geometry='geometry'), predicate='contains')
        start_point_df['centroid'] = start_point
        start_point_df = start_point_df.drop(columns='index_right')

        end_point_df = gpd.sjoin(grid, gpd.GeoDataFrame([end_point], columns=['geometry'], geometry='geometry'), predicate='contains')
        end_point_df['centroid'] = end_point
        end_point_df = end_point_df.drop(columns='index_right')

        grid_filtered = pd.concat([grid_filtered, start_point_df, end_point_df])

        gdf = gdf.set_geometry('geometry')
        grids = grid_filtered.set_geometry('geometry')
        joined = gpd.sjoin(grids, gdf, how='inner', predicate='contains')

        print('Processing Trajectories:', linha_id)
        end = end_point.y, end_point.x
        start = start_point.y, start_point.x



        number_bus = joined['ordem'].nunique()

        all_routes_ida = {} 
        all_routes_volta = {} 
        for ordem_id in joined['ordem'].unique()[:number_bus]:
            route_ida = trace_routes(joined, ordem_id, start, end)  
            route_volta = trace_routes(joined, ordem_id, end, start)        
            if len(route_ida[0]) != 200 and len(route_ida[0]) > 10:
                all_routes_ida[ordem_id] = {'rota': route_ida[0], 'complete': route_ida[1]}

            if len(route_volta[0]) != 200 and len(route_volta[0]) > 10:
                all_routes_volta[ordem_id] = {'rota': route_volta[0], 'complete': route_volta[1]}

        biggest_route_volta = get_best_route(all_routes_volta, grid_filtered)
        biggest_route_ida = get_best_route(all_routes_ida, grid_filtered)
        # Criar um mapa centrado na rota do ônibus
        m = folium.Map([start_point.y, start_point.x], zoom_start=14)

        # Adicionar a rota do ônibus ao mapa]
        # Add grid cells to the map
        for _, row in grid_filtered.iterrows():
            folium.GeoJson(row.geometry).add_to(m)
            # folium.Marker(location=[row['geometry'].centroid.y, row['geometry'].centroid.x],
            #                   icon=folium.DivIcon(html=f'<div style="font-size: 5pt">{row["grid_id"]}</div>')).add_to(m)
            folium.Circle(location=[row.centroid.y, row.centroid.x],
                                    radius=3,
                                    color='red',
                                    fill=True,
                                    popup=row["grid_id"],
                                    fill_color='red').add_to(m)



        folium.PolyLine(locations=[(point[0], point[1]) for point in biggest_route_ida], color='blue').add_to(m)
        folium.PolyLine(locations=[(point[0], point[1]) for point in biggest_route_volta], color='red').add_to(m)

        m.save(f"routes/routes_map_{dia}_{linha_id}.html")

        start_point_wkt = f"POINT ({start_point.x} {start_point.y})"
        end_point_wkt = f"POINT ({end_point.x} {end_point.y})"

        
        best_line_out_wkt = LineString([(point[1], point[0]) for point in biggest_route_ida]).wkt
        best_line_back_wkt = LineString([(point[1], point[0]) for point in biggest_route_volta]).wkt
        print('Inserting trajectory on database')
        
        cursor.execute("""
            INSERT INTO bus_routes (linha, week_day, start_point, end_point, route_line_out, route_line_back)
            VALUES (%s, %s, ST_GeomFromText(%s, 4326), ST_GeomFromText(%s, 4326), ST_GeomFromText(%s, 4326), ST_GeomFromText(%s, 4326))
            """, (linha_id, dia, start_point_wkt, end_point_wkt, best_line_out_wkt, best_line_back_wkt))
        conn.commit()

    


# Parâmetros de conexão com o banco de dados
conn_params = {
    'dbname': 'postgres',
    'user': 'postgres',
    'password': 'postgres',
    'host': 'localhost',
    'port': '5432'
}



print('Generating Grid')
rio_minx, rio_miny = -43.7955, -23.0824
rio_maxx, rio_maxy = -43.1039, -22.7448

grid = create_grid(bounds=(rio_minx, rio_miny, rio_maxx, rio_maxy), n_cells=1500)
grid = grid.reset_index(names='grid_id')

print('Get start and end points')

conn = psycopg2.connect(**conn_params)
query = 'select * from bus_end_points'
# Read the SQL query into a GeoDataFrame
ends_df = gpd.read_postgis(query, conn, geom_col='start_point', crs='EPSG:4326')

# Extract the 'end_point' column as a separate GeoSeries
ends_df['end_point'] = gpd.GeoSeries.from_wkb(ends_df['end_point'], crs='EPSG:4326')

 
# Dividir as linhas de ônibus entre threads
bus_lines = ['483', '864', '639', '3', '309', '774', '629', 
				  '371', '397', '100', '838', '315', '624', '388', 
				  '918', '665', '328', '497', '878', '355', '138', '606', '457', '550', 
				  '803', '917', '638', '2336', '399', '298', '867', '553', '565', '422', 
				  '756', '186012003', '292', '554', '634', '232', '415', '2803', '324', 
				  '852', '557', '759', '343', '779', '905', '108']

num_threads = 6
chunks = [bus_lines[i::num_threads] for i in range(num_threads)]
cursor = conn.cursor()

# Usar ThreadPoolExecutor para processar as linhas de ônibus em paralelo
with ThreadPoolExecutor(max_workers=num_threads) as executor:
    futures = [executor.submit(process_bus_lines, chunk, 'Domingo', conn, grid, ends_df, cursor) for chunk in chunks]

# Esperar que todas as threads terminem
for future in futures:
    future.result()

print("Processamento concluído.")

