create table grid_cells as 
WITH bounds AS (
    SELECT 
		linha,
		0.0005 as cell_size,
        GREATEST(MIN(ST_X(geom::geometry)), -43.7955) AS XMIN,
        GREATEST(MIN(ST_Y(geom::geometry)), -23.0824) AS miny,
        LEAST(MAX(ST_X(geom::geometry)), -43.1039) AS maxx,
        LEAST(MAX(ST_Y(geom::geometry)), -22.7448) AS maxy
    FROM public.vehicle_tracking_filtered_sample 
	GROUP BY linha
), grid AS (
    SELECT
        p.linha,
        row_number() OVER (PARTITION BY p.linha) AS id,
        ST_SetSRID(ST_MakeEnvelope(
            p.xmin + i * p.cell_size,
            p.ymin + j * p.cell_size,
            p.xmin + (i + 1) * p.cell_size,
            p.ymin + (j + 1) * p.cell_size,
            4326
        ), 4326) AS geom_grid
    FROM bounds p,
    generate_series(0, floor((p.xmax - p.xmin) / p.cell_size)::int) AS i,
    generate_series(0, floor((p.ymax - p.ymin) / p.cell_size)::int) AS j
) SELECT DISTINCT g.linha, geom_grid AS cell_geom
    FROM grid g
    INNER JOIN vehicle_tracking_filtered_sample s
    ON ST_Contains(geom_grid, s.geom::geometry) AND g.linha = s.linha
