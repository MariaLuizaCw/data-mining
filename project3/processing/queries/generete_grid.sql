create table grid_cells_linha as
WITH params AS (
    SELECT
        linha,
        ST_XMin(ST_Extent(geom::geometry)) AS xmin,
        ST_YMin(ST_Extent(geom::geometry)) AS ymin,
        ST_XMax(ST_Extent(geom::geometry)) AS xmax,
        ST_YMax(ST_Extent(geom::geometry)) AS ymax,
        0.0005 AS cell_size -- Ajustar o tamanho da célula conforme necessário
    FROM public.vehicle_tracking
    WHERE (datahora >= '20240508_0000' AND datahora < '20240509_0000') or (datahora >= '20240504_0000' AND datahora < '20240505_0000')
   	and linha in ('483', '864', '639', '3', '309', '774', '629', 
				  '371', '397', '100', '838', '315', '624', '388', 
				  '918', '665', '328', '497', '878', '355', '138', '606', '457', '550', 
				  '803', '917', '638', '2336', '399', '298', '867', '553', '565', '422', 
				  '756', '186012003', '292', '554', '634', '232', '415', '2803', '324', 
				  '852', '557', '759', '343', '779', '905', '108')
	group by linha
),
grid AS (
    SELECT
        p.linha,
        row_number() OVER (PARTITION BY p.linha) AS id,
        ST_SetSRID(ST_MakeEnvelope(
            p.xmin + i * p.cell_size,
            p.ymin + j * p.cell_size,
            p.xmin + (i + 1) * p.cell_size,
            p.ymin + (j + 1) * p.cell_size,
            4326
        ), 4326) AS geom
    FROM params p,
    generate_series(0, floor((p.xmax - p.xmin) / p.cell_size)::int) AS i,
    generate_series(0, floor((p.ymax - p.ymin) / p.cell_size)::int) AS j
)
SELECT linha, geom, id FROM grid;