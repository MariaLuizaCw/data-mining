drop table grid_point_counts;
create table grid_point_counts as
SELECT
    gc.linha,
    gc.id AS grid_id,
    COUNT(po.geom) AS point_count,
    'madrugada' AS horario
FROM
    grid_cells gc
LEFT JOIN
    public.vehicle_tracking po
ON
    ST_Contains(gc.geom, po.geom::geometry) AND gc.linha = po.linha
WHERE po.datahora >= '20240508_000000' AND po.datahora < '20240508_050000'
and linha = '606' or linha = '634'
UNION ALL
SELECT
    gc.linha,
    gc.id AS grid_id,
    COUNT(po.geom) AS point_count,
    'restante' AS horario
FROM
    grid_cells gc
LEFT JOIN
    public.vehicle_tracking po
ON
    ST_Contains(gc.geom, po.geom::geometry) AND gc.linha = po.linha
WHERE po.datahora >= '20240508_050000' AND po.datahora < '20240509_000000'
and linha = '606' or linha = '634'


