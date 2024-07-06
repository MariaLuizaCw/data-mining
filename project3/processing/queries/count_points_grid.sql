


drop table if exists grid_point_counts_606;
create table grid_point_counts_606 as
SELECT
    gc.linha,
    gc.id AS grid_id,
    COUNT(po.geom) AS point_count,
    'madrugada' AS horario
FROM
    grid_cells_linha gc
LEFT JOIN
    public.vehicle_tracking po
ON
    ST_Contains(gc.geom, po.geom::geometry) AND gc.linha = po.linha
WHERE po.datahora >= '20240508_000000' AND po.datahora < '20240508_050000'
and gc.linha = '606'
group by gc.linha, gc.id
UNION ALL
SELECT
    gc.linha,
    gc.id AS grid_id,
    COUNT(po.geom) AS point_count,
    'restante' AS horario
FROM
    grid_cells_linha gc
LEFT JOIN
    public.vehicle_tracking po
ON
    ST_Contains(gc.geom, po.geom::geometry) AND gc.linha = po.linha
WHERE po.datahora >= '20240508_050000' AND po.datahora < '20240509_000000'
and gc.linha = '606'
group by gc.linha, gc.id
