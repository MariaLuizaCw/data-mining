INSERT INTO grid_cells_filtered
SELECT DISTINCT  gc.linha, gc.geom, gc.id
FROM grid_cells_linha gc
INNER JOIN public.vehicle_tracking po
ON ST_Contains(gc.geom, po.geom::geometry) AND gc.linha = po.linha
where (
	(datahora >= '20240504_1400' and datahora <= '20240504_1800') or
	(datahora >= '20240505_1400' and datahora <= '20240505_1800') or
	(datahora >= '20240506_1400' and datahora <= '20240506_1800') or
	(datahora >= '20240507_1400' and datahora <= '20240507_1800') or
	(datahora >= '20240508_1400' and datahora <= '20240508_1800') or
	(datahora >= '20240509_1400' and datahora <= '20240509_1800') or
	(datahora >= '20240510_1400' and datahora <= '20240510_1800' ) 
)
