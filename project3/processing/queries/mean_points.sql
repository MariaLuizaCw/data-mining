WITH grid_points AS (
    SELECT
        gc.id AS grid_id,
        po.linha,
        po.geom::geometry
    FROM
        grid_cells gc
    LEFT JOIN
        public.vehicle_tracking po
    ON
        ST_Contains(gc.geom, po.geom::geometry)
),
mean_points AS (
    SELECT
        gp.linha,
        gp.grid_id,
        COUNT(gp.geom) AS point_count_24h,
        COALESCE(SUM(CASE WHEN po.datahora >= '20240508_000000' AND po.datahora < '20240508_050000' THEN 1 ELSE 0 END), 0) AS point_count_00_05,
        CASE
            WHEN COUNT(gp.geom) < 10 OR SUM(CASE WHEN po.datahora >= '20240508_000000' AND po.datahora < '20240508_050000' THEN 1 ELSE 0 END) >= 100 THEN
                ST_SetSRID(ST_MakePoint(0, 0), 4326)::geography
            ELSE
                ST_SetSRID(ST_MakePoint(AVG(ST_X(gp.geom)), AVG(ST_Y(gp.geom))), 4326)::geography
        END AS media_geom
    FROM
        grid_points gp
    LEFT JOIN
        public.vehicle_tracking po
    ON
        gp.grid_id = po.grid_id AND gp.linha = po.linha
    GROUP BY
        gp.linha, gp.grid_id
)
INSERT INTO pontos_media (linha, media_geom, grid_id)
SELECT linha, media_geom, grid_id FROM mean_points;