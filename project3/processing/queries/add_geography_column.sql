alter table vehicle_tracking add column geom GEOGRAPHY(Point, 4326)


update vehicle_tracking
set geom = ST_SetSRID(ST_MakePoint(longitude, latitude), 4326)::geography