create table vehicle_tracking_filtered_sample as 
SELECT *
FROM (
    SELECT a.*,
           ROW_NUMBER() OVER (PARTITION BY linha, DATE(datahora) ORDER BY RANDOM()) AS row_num
    FROM vehicle_tracking_filtered a 
) AS subquery
WHERE row_num <= 1000;

create index idx_sample_linha on vehicle_tracking_filtered_sample(linha)

CREATE INDEX idx_sample_geom ON vehicle_tracking_filtered_sample USING GIST (geom);
	
