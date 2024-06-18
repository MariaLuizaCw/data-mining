-- Table: public.vehicle_tracking

-- DROP TABLE IF EXISTS public.vehicle_tracking;

CREATE TABLE IF NOT EXISTS public.vehicle_tracking
(
    ordem text COLLATE pg_catalog."default",
    latitude double precision,
    longitude double precision,
    datahora_epoch bigint,
    datahora timestamp without time zone,
    velocidade integer,
    linha text COLLATE pg_catalog."default",
    datahoraenvio_epoch bigint,
    datahoraservidor_epoch bigint,
    datahoraservidor timestamp without time zone,
    datahoraenvio timestamp without time zone,
    geom geography(Point,4326)
) PARTITION BY RANGE (datahora);

ALTER TABLE IF EXISTS public.vehicle_tracking
    OWNER to postgres;
-- Index: linha_idx

-- DROP INDEX IF EXISTS public.linha_idx;

CREATE INDEX IF NOT EXISTS linha_idx
    ON public.vehicle_tracking USING btree
    (linha COLLATE pg_catalog."default" ASC NULLS LAST)
;

-- Partitions SQL

CREATE TABLE public.vehicle_tracking_20240424_0000 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-04-24 00:00:00') TO ('2024-04-24 01:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240424_0000
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240424_0100 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-04-24 01:00:00') TO ('2024-04-24 02:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240424_0100
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240424_0200 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-04-24 02:00:00') TO ('2024-04-24 03:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240424_0200
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240424_0300 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-04-24 03:00:00') TO ('2024-04-24 04:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240424_0300
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240424_0400 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-04-24 04:00:00') TO ('2024-04-24 05:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240424_0400
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240424_0500 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-04-24 05:00:00') TO ('2024-04-24 06:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240424_0500
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240424_0600 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-04-24 06:00:00') TO ('2024-04-24 07:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240424_0600
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240424_0700 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-04-24 07:00:00') TO ('2024-04-24 08:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240424_0700
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240424_0800 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-04-24 08:00:00') TO ('2024-04-24 09:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240424_0800
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240424_0900 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-04-24 09:00:00') TO ('2024-04-24 10:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240424_0900
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240424_1000 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-04-24 10:00:00') TO ('2024-04-24 11:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240424_1000
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240424_1100 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-04-24 11:00:00') TO ('2024-04-24 12:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240424_1100
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240424_1200 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-04-24 12:00:00') TO ('2024-04-24 13:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240424_1200
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240424_1300 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-04-24 13:00:00') TO ('2024-04-24 14:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240424_1300
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240424_1400 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-04-24 14:00:00') TO ('2024-04-24 15:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240424_1400
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240424_1500 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-04-24 15:00:00') TO ('2024-04-24 16:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240424_1500
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240424_1600 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-04-24 16:00:00') TO ('2024-04-24 17:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240424_1600
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240424_1700 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-04-24 17:00:00') TO ('2024-04-24 18:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240424_1700
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240424_1800 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-04-24 18:00:00') TO ('2024-04-24 19:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240424_1800
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240424_1900 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-04-24 19:00:00') TO ('2024-04-24 20:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240424_1900
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240424_2000 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-04-24 20:00:00') TO ('2024-04-24 21:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240424_2000
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240424_2100 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-04-24 21:00:00') TO ('2024-04-24 22:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240424_2100
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240424_2200 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-04-24 22:00:00') TO ('2024-04-24 23:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240424_2200
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240424_2300 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-04-24 23:00:00') TO ('2024-04-25 00:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240424_2300
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240425_0000 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-04-25 00:00:00') TO ('2024-04-25 01:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240425_0000
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240425_0100 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-04-25 01:00:00') TO ('2024-04-25 02:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240425_0100
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240425_0200 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-04-25 02:00:00') TO ('2024-04-25 03:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240425_0200
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240425_0300 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-04-25 03:00:00') TO ('2024-04-25 04:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240425_0300
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240425_0400 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-04-25 04:00:00') TO ('2024-04-25 05:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240425_0400
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240425_0500 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-04-25 05:00:00') TO ('2024-04-25 06:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240425_0500
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240425_0600 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-04-25 06:00:00') TO ('2024-04-25 07:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240425_0600
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240425_0700 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-04-25 07:00:00') TO ('2024-04-25 08:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240425_0700
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240425_0800 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-04-25 08:00:00') TO ('2024-04-25 09:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240425_0800
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240425_0900 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-04-25 09:00:00') TO ('2024-04-25 10:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240425_0900
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240425_1000 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-04-25 10:00:00') TO ('2024-04-25 11:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240425_1000
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240425_1100 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-04-25 11:00:00') TO ('2024-04-25 12:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240425_1100
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240425_1200 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-04-25 12:00:00') TO ('2024-04-25 13:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240425_1200
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240425_1300 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-04-25 13:00:00') TO ('2024-04-25 14:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240425_1300
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240425_1400 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-04-25 14:00:00') TO ('2024-04-25 15:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240425_1400
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240425_1500 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-04-25 15:00:00') TO ('2024-04-25 16:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240425_1500
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240425_1600 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-04-25 16:00:00') TO ('2024-04-25 17:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240425_1600
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240425_1700 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-04-25 17:00:00') TO ('2024-04-25 18:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240425_1700
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240425_1800 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-04-25 18:00:00') TO ('2024-04-25 19:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240425_1800
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240425_1900 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-04-25 19:00:00') TO ('2024-04-25 20:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240425_1900
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240425_2000 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-04-25 20:00:00') TO ('2024-04-25 21:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240425_2000
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240425_2100 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-04-25 21:00:00') TO ('2024-04-25 22:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240425_2100
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240425_2200 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-04-25 22:00:00') TO ('2024-04-25 23:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240425_2200
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240425_2300 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-04-25 23:00:00') TO ('2024-04-26 00:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240425_2300
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240426_0000 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-04-26 00:00:00') TO ('2024-04-26 01:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240426_0000
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240426_0100 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-04-26 01:00:00') TO ('2024-04-26 02:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240426_0100
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240426_0200 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-04-26 02:00:00') TO ('2024-04-26 03:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240426_0200
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240426_0300 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-04-26 03:00:00') TO ('2024-04-26 04:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240426_0300
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240426_0400 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-04-26 04:00:00') TO ('2024-04-26 05:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240426_0400
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240426_0500 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-04-26 05:00:00') TO ('2024-04-26 06:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240426_0500
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240426_0600 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-04-26 06:00:00') TO ('2024-04-26 07:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240426_0600
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240426_0700 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-04-26 07:00:00') TO ('2024-04-26 08:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240426_0700
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240426_0800 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-04-26 08:00:00') TO ('2024-04-26 09:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240426_0800
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240426_0900 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-04-26 09:00:00') TO ('2024-04-26 10:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240426_0900
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240426_1000 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-04-26 10:00:00') TO ('2024-04-26 11:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240426_1000
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240426_1100 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-04-26 11:00:00') TO ('2024-04-26 12:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240426_1100
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240426_1200 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-04-26 12:00:00') TO ('2024-04-26 13:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240426_1200
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240426_1300 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-04-26 13:00:00') TO ('2024-04-26 14:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240426_1300
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240426_1400 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-04-26 14:00:00') TO ('2024-04-26 15:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240426_1400
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240426_1500 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-04-26 15:00:00') TO ('2024-04-26 16:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240426_1500
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240426_1600 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-04-26 16:00:00') TO ('2024-04-26 17:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240426_1600
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240426_1700 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-04-26 17:00:00') TO ('2024-04-26 18:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240426_1700
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240426_1800 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-04-26 18:00:00') TO ('2024-04-26 19:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240426_1800
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240426_1900 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-04-26 19:00:00') TO ('2024-04-26 20:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240426_1900
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240426_2000 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-04-26 20:00:00') TO ('2024-04-26 21:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240426_2000
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240426_2100 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-04-26 21:00:00') TO ('2024-04-26 22:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240426_2100
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240426_2200 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-04-26 22:00:00') TO ('2024-04-26 23:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240426_2200
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240426_2300 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-04-26 23:00:00') TO ('2024-04-27 00:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240426_2300
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240427_0000 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-04-27 00:00:00') TO ('2024-04-27 01:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240427_0000
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240427_0100 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-04-27 01:00:00') TO ('2024-04-27 02:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240427_0100
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240427_0200 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-04-27 02:00:00') TO ('2024-04-27 03:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240427_0200
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240427_0300 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-04-27 03:00:00') TO ('2024-04-27 04:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240427_0300
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240427_0400 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-04-27 04:00:00') TO ('2024-04-27 05:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240427_0400
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240427_0500 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-04-27 05:00:00') TO ('2024-04-27 06:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240427_0500
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240427_0600 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-04-27 06:00:00') TO ('2024-04-27 07:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240427_0600
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240427_0700 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-04-27 07:00:00') TO ('2024-04-27 08:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240427_0700
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240427_0800 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-04-27 08:00:00') TO ('2024-04-27 09:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240427_0800
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240427_0900 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-04-27 09:00:00') TO ('2024-04-27 10:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240427_0900
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240427_1000 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-04-27 10:00:00') TO ('2024-04-27 11:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240427_1000
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240427_1100 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-04-27 11:00:00') TO ('2024-04-27 12:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240427_1100
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240427_1200 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-04-27 12:00:00') TO ('2024-04-27 13:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240427_1200
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240427_1300 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-04-27 13:00:00') TO ('2024-04-27 14:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240427_1300
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240427_1400 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-04-27 14:00:00') TO ('2024-04-27 15:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240427_1400
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240427_1500 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-04-27 15:00:00') TO ('2024-04-27 16:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240427_1500
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240427_1600 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-04-27 16:00:00') TO ('2024-04-27 17:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240427_1600
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240427_1700 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-04-27 17:00:00') TO ('2024-04-27 18:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240427_1700
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240427_1800 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-04-27 18:00:00') TO ('2024-04-27 19:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240427_1800
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240427_1900 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-04-27 19:00:00') TO ('2024-04-27 20:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240427_1900
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240427_2000 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-04-27 20:00:00') TO ('2024-04-27 21:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240427_2000
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240427_2100 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-04-27 21:00:00') TO ('2024-04-27 22:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240427_2100
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240427_2200 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-04-27 22:00:00') TO ('2024-04-27 23:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240427_2200
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240427_2300 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-04-27 23:00:00') TO ('2024-04-28 00:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240427_2300
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240428_0000 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-04-28 00:00:00') TO ('2024-04-28 01:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240428_0000
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240428_0100 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-04-28 01:00:00') TO ('2024-04-28 02:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240428_0100
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240428_0200 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-04-28 02:00:00') TO ('2024-04-28 03:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240428_0200
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240428_0300 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-04-28 03:00:00') TO ('2024-04-28 04:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240428_0300
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240428_0400 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-04-28 04:00:00') TO ('2024-04-28 05:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240428_0400
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240428_0500 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-04-28 05:00:00') TO ('2024-04-28 06:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240428_0500
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240428_0600 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-04-28 06:00:00') TO ('2024-04-28 07:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240428_0600
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240428_0700 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-04-28 07:00:00') TO ('2024-04-28 08:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240428_0700
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240428_0800 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-04-28 08:00:00') TO ('2024-04-28 09:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240428_0800
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240428_0900 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-04-28 09:00:00') TO ('2024-04-28 10:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240428_0900
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240428_1000 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-04-28 10:00:00') TO ('2024-04-28 11:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240428_1000
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240428_1100 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-04-28 11:00:00') TO ('2024-04-28 12:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240428_1100
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240428_1200 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-04-28 12:00:00') TO ('2024-04-28 13:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240428_1200
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240428_1300 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-04-28 13:00:00') TO ('2024-04-28 14:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240428_1300
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240428_1400 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-04-28 14:00:00') TO ('2024-04-28 15:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240428_1400
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240428_1500 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-04-28 15:00:00') TO ('2024-04-28 16:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240428_1500
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240428_1600 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-04-28 16:00:00') TO ('2024-04-28 17:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240428_1600
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240428_1700 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-04-28 17:00:00') TO ('2024-04-28 18:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240428_1700
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240428_1800 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-04-28 18:00:00') TO ('2024-04-28 19:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240428_1800
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240428_1900 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-04-28 19:00:00') TO ('2024-04-28 20:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240428_1900
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240428_2000 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-04-28 20:00:00') TO ('2024-04-28 21:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240428_2000
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240428_2100 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-04-28 21:00:00') TO ('2024-04-28 22:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240428_2100
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240428_2200 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-04-28 22:00:00') TO ('2024-04-28 23:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240428_2200
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240428_2300 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-04-28 23:00:00') TO ('2024-04-29 00:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240428_2300
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240429_0000 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-04-29 00:00:00') TO ('2024-04-29 01:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240429_0000
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240429_0100 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-04-29 01:00:00') TO ('2024-04-29 02:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240429_0100
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240429_0200 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-04-29 02:00:00') TO ('2024-04-29 03:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240429_0200
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240429_0300 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-04-29 03:00:00') TO ('2024-04-29 04:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240429_0300
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240429_0400 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-04-29 04:00:00') TO ('2024-04-29 05:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240429_0400
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240429_0500 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-04-29 05:00:00') TO ('2024-04-29 06:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240429_0500
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240429_0600 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-04-29 06:00:00') TO ('2024-04-29 07:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240429_0600
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240429_0700 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-04-29 07:00:00') TO ('2024-04-29 08:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240429_0700
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240429_0800 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-04-29 08:00:00') TO ('2024-04-29 09:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240429_0800
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240429_0900 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-04-29 09:00:00') TO ('2024-04-29 10:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240429_0900
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240429_1000 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-04-29 10:00:00') TO ('2024-04-29 11:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240429_1000
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240429_1100 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-04-29 11:00:00') TO ('2024-04-29 12:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240429_1100
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240429_1200 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-04-29 12:00:00') TO ('2024-04-29 13:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240429_1200
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240429_1300 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-04-29 13:00:00') TO ('2024-04-29 14:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240429_1300
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240429_1400 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-04-29 14:00:00') TO ('2024-04-29 15:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240429_1400
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240429_1500 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-04-29 15:00:00') TO ('2024-04-29 16:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240429_1500
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240429_1600 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-04-29 16:00:00') TO ('2024-04-29 17:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240429_1600
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240429_1700 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-04-29 17:00:00') TO ('2024-04-29 18:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240429_1700
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240429_1800 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-04-29 18:00:00') TO ('2024-04-29 19:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240429_1800
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240429_1900 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-04-29 19:00:00') TO ('2024-04-29 20:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240429_1900
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240429_2000 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-04-29 20:00:00') TO ('2024-04-29 21:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240429_2000
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240429_2100 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-04-29 21:00:00') TO ('2024-04-29 22:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240429_2100
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240429_2200 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-04-29 22:00:00') TO ('2024-04-29 23:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240429_2200
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240429_2300 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-04-29 23:00:00') TO ('2024-04-30 00:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240429_2300
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240430_0000 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-04-30 00:00:00') TO ('2024-04-30 01:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240430_0000
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240430_0100 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-04-30 01:00:00') TO ('2024-04-30 02:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240430_0100
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240430_0200 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-04-30 02:00:00') TO ('2024-04-30 03:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240430_0200
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240430_0300 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-04-30 03:00:00') TO ('2024-04-30 04:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240430_0300
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240430_0400 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-04-30 04:00:00') TO ('2024-04-30 05:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240430_0400
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240430_0500 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-04-30 05:00:00') TO ('2024-04-30 06:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240430_0500
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240430_0600 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-04-30 06:00:00') TO ('2024-04-30 07:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240430_0600
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240430_0700 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-04-30 07:00:00') TO ('2024-04-30 08:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240430_0700
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240430_0800 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-04-30 08:00:00') TO ('2024-04-30 09:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240430_0800
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240430_0900 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-04-30 09:00:00') TO ('2024-04-30 10:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240430_0900
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240430_1000 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-04-30 10:00:00') TO ('2024-04-30 11:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240430_1000
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240430_1100 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-04-30 11:00:00') TO ('2024-04-30 12:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240430_1100
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240430_1200 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-04-30 12:00:00') TO ('2024-04-30 13:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240430_1200
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240430_1300 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-04-30 13:00:00') TO ('2024-04-30 14:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240430_1300
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240430_1400 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-04-30 14:00:00') TO ('2024-04-30 15:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240430_1400
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240430_1500 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-04-30 15:00:00') TO ('2024-04-30 16:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240430_1500
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240430_1600 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-04-30 16:00:00') TO ('2024-04-30 17:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240430_1600
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240430_1700 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-04-30 17:00:00') TO ('2024-04-30 18:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240430_1700
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240430_1800 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-04-30 18:00:00') TO ('2024-04-30 19:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240430_1800
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240430_1900 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-04-30 19:00:00') TO ('2024-04-30 20:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240430_1900
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240430_2000 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-04-30 20:00:00') TO ('2024-04-30 21:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240430_2000
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240430_2100 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-04-30 21:00:00') TO ('2024-04-30 22:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240430_2100
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240430_2200 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-04-30 22:00:00') TO ('2024-04-30 23:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240430_2200
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240430_2300 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-04-30 23:00:00') TO ('2024-05-01 00:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240430_2300
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240501_0000 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-05-01 00:00:00') TO ('2024-05-01 01:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240501_0000
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240501_0100 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-05-01 01:00:00') TO ('2024-05-01 02:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240501_0100
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240501_0200 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-05-01 02:00:00') TO ('2024-05-01 03:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240501_0200
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240501_0300 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-05-01 03:00:00') TO ('2024-05-01 04:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240501_0300
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240501_0400 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-05-01 04:00:00') TO ('2024-05-01 05:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240501_0400
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240501_0500 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-05-01 05:00:00') TO ('2024-05-01 06:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240501_0500
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240501_0600 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-05-01 06:00:00') TO ('2024-05-01 07:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240501_0600
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240501_0700 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-05-01 07:00:00') TO ('2024-05-01 08:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240501_0700
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240501_0800 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-05-01 08:00:00') TO ('2024-05-01 09:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240501_0800
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240501_0900 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-05-01 09:00:00') TO ('2024-05-01 10:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240501_0900
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240501_1000 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-05-01 10:00:00') TO ('2024-05-01 11:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240501_1000
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240501_1100 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-05-01 11:00:00') TO ('2024-05-01 12:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240501_1100
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240501_1200 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-05-01 12:00:00') TO ('2024-05-01 13:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240501_1200
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240501_1300 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-05-01 13:00:00') TO ('2024-05-01 14:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240501_1300
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240501_1400 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-05-01 14:00:00') TO ('2024-05-01 15:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240501_1400
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240501_1500 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-05-01 15:00:00') TO ('2024-05-01 16:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240501_1500
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240501_1600 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-05-01 16:00:00') TO ('2024-05-01 17:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240501_1600
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240501_1700 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-05-01 17:00:00') TO ('2024-05-01 18:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240501_1700
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240501_1800 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-05-01 18:00:00') TO ('2024-05-01 19:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240501_1800
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240501_1900 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-05-01 19:00:00') TO ('2024-05-01 20:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240501_1900
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240501_2000 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-05-01 20:00:00') TO ('2024-05-01 21:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240501_2000
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240501_2100 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-05-01 21:00:00') TO ('2024-05-01 22:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240501_2100
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240501_2200 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-05-01 22:00:00') TO ('2024-05-01 23:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240501_2200
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240501_2300 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-05-01 23:00:00') TO ('2024-05-02 00:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240501_2300
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240502_0000 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-05-02 00:00:00') TO ('2024-05-02 01:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240502_0000
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240502_0100 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-05-02 01:00:00') TO ('2024-05-02 02:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240502_0100
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240502_0200 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-05-02 02:00:00') TO ('2024-05-02 03:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240502_0200
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240502_0300 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-05-02 03:00:00') TO ('2024-05-02 04:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240502_0300
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240502_0400 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-05-02 04:00:00') TO ('2024-05-02 05:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240502_0400
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240502_0500 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-05-02 05:00:00') TO ('2024-05-02 06:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240502_0500
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240502_0600 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-05-02 06:00:00') TO ('2024-05-02 07:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240502_0600
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240502_0700 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-05-02 07:00:00') TO ('2024-05-02 08:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240502_0700
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240502_0800 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-05-02 08:00:00') TO ('2024-05-02 09:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240502_0800
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240502_0900 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-05-02 09:00:00') TO ('2024-05-02 10:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240502_0900
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240502_1000 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-05-02 10:00:00') TO ('2024-05-02 11:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240502_1000
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240502_1100 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-05-02 11:00:00') TO ('2024-05-02 12:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240502_1100
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240502_1200 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-05-02 12:00:00') TO ('2024-05-02 13:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240502_1200
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240502_1300 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-05-02 13:00:00') TO ('2024-05-02 14:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240502_1300
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240502_1400 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-05-02 14:00:00') TO ('2024-05-02 15:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240502_1400
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240502_1500 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-05-02 15:00:00') TO ('2024-05-02 16:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240502_1500
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240502_1600 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-05-02 16:00:00') TO ('2024-05-02 17:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240502_1600
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240502_1700 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-05-02 17:00:00') TO ('2024-05-02 18:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240502_1700
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240502_1800 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-05-02 18:00:00') TO ('2024-05-02 19:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240502_1800
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240502_1900 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-05-02 19:00:00') TO ('2024-05-02 20:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240502_1900
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240502_2000 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-05-02 20:00:00') TO ('2024-05-02 21:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240502_2000
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240502_2100 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-05-02 21:00:00') TO ('2024-05-02 22:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240502_2100
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240502_2200 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-05-02 22:00:00') TO ('2024-05-02 23:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240502_2200
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240502_2300 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-05-02 23:00:00') TO ('2024-05-03 00:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240502_2300
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240503_0000 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-05-03 00:00:00') TO ('2024-05-03 01:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240503_0000
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240503_0100 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-05-03 01:00:00') TO ('2024-05-03 02:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240503_0100
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240503_0200 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-05-03 02:00:00') TO ('2024-05-03 03:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240503_0200
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240503_0300 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-05-03 03:00:00') TO ('2024-05-03 04:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240503_0300
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240503_0400 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-05-03 04:00:00') TO ('2024-05-03 05:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240503_0400
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240503_0500 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-05-03 05:00:00') TO ('2024-05-03 06:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240503_0500
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240503_0600 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-05-03 06:00:00') TO ('2024-05-03 07:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240503_0600
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240503_0700 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-05-03 07:00:00') TO ('2024-05-03 08:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240503_0700
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240503_0800 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-05-03 08:00:00') TO ('2024-05-03 09:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240503_0800
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240503_0900 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-05-03 09:00:00') TO ('2024-05-03 10:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240503_0900
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240503_1000 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-05-03 10:00:00') TO ('2024-05-03 11:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240503_1000
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240503_1100 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-05-03 11:00:00') TO ('2024-05-03 12:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240503_1100
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240503_1200 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-05-03 12:00:00') TO ('2024-05-03 13:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240503_1200
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240503_1300 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-05-03 13:00:00') TO ('2024-05-03 14:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240503_1300
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240503_1400 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-05-03 14:00:00') TO ('2024-05-03 15:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240503_1400
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240503_1500 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-05-03 15:00:00') TO ('2024-05-03 16:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240503_1500
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240503_1600 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-05-03 16:00:00') TO ('2024-05-03 17:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240503_1600
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240503_1700 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-05-03 17:00:00') TO ('2024-05-03 18:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240503_1700
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240503_1800 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-05-03 18:00:00') TO ('2024-05-03 19:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240503_1800
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240503_1900 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-05-03 19:00:00') TO ('2024-05-03 20:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240503_1900
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240503_2000 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-05-03 20:00:00') TO ('2024-05-03 21:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240503_2000
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240503_2100 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-05-03 21:00:00') TO ('2024-05-03 22:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240503_2100
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240503_2200 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-05-03 22:00:00') TO ('2024-05-03 23:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240503_2200
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240503_2300 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-05-03 23:00:00') TO ('2024-05-04 00:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240503_2300
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240504_0000 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-05-04 00:00:00') TO ('2024-05-04 01:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240504_0000
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240504_0100 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-05-04 01:00:00') TO ('2024-05-04 02:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240504_0100
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240504_0200 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-05-04 02:00:00') TO ('2024-05-04 03:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240504_0200
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240504_0300 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-05-04 03:00:00') TO ('2024-05-04 04:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240504_0300
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240504_0400 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-05-04 04:00:00') TO ('2024-05-04 05:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240504_0400
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240504_0500 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-05-04 05:00:00') TO ('2024-05-04 06:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240504_0500
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240504_0600 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-05-04 06:00:00') TO ('2024-05-04 07:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240504_0600
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240504_0700 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-05-04 07:00:00') TO ('2024-05-04 08:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240504_0700
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240504_0800 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-05-04 08:00:00') TO ('2024-05-04 09:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240504_0800
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240504_0900 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-05-04 09:00:00') TO ('2024-05-04 10:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240504_0900
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240504_1000 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-05-04 10:00:00') TO ('2024-05-04 11:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240504_1000
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240504_1100 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-05-04 11:00:00') TO ('2024-05-04 12:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240504_1100
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240504_1200 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-05-04 12:00:00') TO ('2024-05-04 13:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240504_1200
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240504_1300 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-05-04 13:00:00') TO ('2024-05-04 14:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240504_1300
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240504_1400 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-05-04 14:00:00') TO ('2024-05-04 15:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240504_1400
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240504_1500 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-05-04 15:00:00') TO ('2024-05-04 16:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240504_1500
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240504_1600 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-05-04 16:00:00') TO ('2024-05-04 17:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240504_1600
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240504_1700 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-05-04 17:00:00') TO ('2024-05-04 18:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240504_1700
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240504_1800 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-05-04 18:00:00') TO ('2024-05-04 19:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240504_1800
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240504_1900 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-05-04 19:00:00') TO ('2024-05-04 20:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240504_1900
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240504_2000 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-05-04 20:00:00') TO ('2024-05-04 21:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240504_2000
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240504_2100 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-05-04 21:00:00') TO ('2024-05-04 22:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240504_2100
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240504_2200 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-05-04 22:00:00') TO ('2024-05-04 23:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240504_2200
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240504_2300 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-05-04 23:00:00') TO ('2024-05-05 00:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240504_2300
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240505_0000 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-05-05 00:00:00') TO ('2024-05-05 01:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240505_0000
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240505_0100 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-05-05 01:00:00') TO ('2024-05-05 02:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240505_0100
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240505_0200 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-05-05 02:00:00') TO ('2024-05-05 03:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240505_0200
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240505_0300 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-05-05 03:00:00') TO ('2024-05-05 04:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240505_0300
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240505_0400 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-05-05 04:00:00') TO ('2024-05-05 05:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240505_0400
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240505_0500 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-05-05 05:00:00') TO ('2024-05-05 06:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240505_0500
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240505_0600 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-05-05 06:00:00') TO ('2024-05-05 07:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240505_0600
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240505_0700 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-05-05 07:00:00') TO ('2024-05-05 08:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240505_0700
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240505_0800 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-05-05 08:00:00') TO ('2024-05-05 09:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240505_0800
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240505_0900 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-05-05 09:00:00') TO ('2024-05-05 10:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240505_0900
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240505_1000 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-05-05 10:00:00') TO ('2024-05-05 11:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240505_1000
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240505_1100 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-05-05 11:00:00') TO ('2024-05-05 12:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240505_1100
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240505_1200 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-05-05 12:00:00') TO ('2024-05-05 13:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240505_1200
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240505_1300 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-05-05 13:00:00') TO ('2024-05-05 14:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240505_1300
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240505_1400 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-05-05 14:00:00') TO ('2024-05-05 15:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240505_1400
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240505_1500 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-05-05 15:00:00') TO ('2024-05-05 16:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240505_1500
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240505_1600 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-05-05 16:00:00') TO ('2024-05-05 17:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240505_1600
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240505_1700 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-05-05 17:00:00') TO ('2024-05-05 18:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240505_1700
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240505_1800 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-05-05 18:00:00') TO ('2024-05-05 19:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240505_1800
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240505_1900 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-05-05 19:00:00') TO ('2024-05-05 20:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240505_1900
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240505_2000 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-05-05 20:00:00') TO ('2024-05-05 21:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240505_2000
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240505_2100 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-05-05 21:00:00') TO ('2024-05-05 22:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240505_2100
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240505_2200 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-05-05 22:00:00') TO ('2024-05-05 23:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240505_2200
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240505_2300 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-05-05 23:00:00') TO ('2024-05-06 00:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240505_2300
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240506_0000 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-05-06 00:00:00') TO ('2024-05-06 01:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240506_0000
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240506_0100 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-05-06 01:00:00') TO ('2024-05-06 02:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240506_0100
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240506_0200 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-05-06 02:00:00') TO ('2024-05-06 03:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240506_0200
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240506_0300 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-05-06 03:00:00') TO ('2024-05-06 04:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240506_0300
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240506_0400 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-05-06 04:00:00') TO ('2024-05-06 05:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240506_0400
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240506_0500 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-05-06 05:00:00') TO ('2024-05-06 06:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240506_0500
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240506_0600 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-05-06 06:00:00') TO ('2024-05-06 07:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240506_0600
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240506_0700 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-05-06 07:00:00') TO ('2024-05-06 08:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240506_0700
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240506_0800 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-05-06 08:00:00') TO ('2024-05-06 09:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240506_0800
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240506_0900 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-05-06 09:00:00') TO ('2024-05-06 10:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240506_0900
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240506_1000 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-05-06 10:00:00') TO ('2024-05-06 11:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240506_1000
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240506_1100 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-05-06 11:00:00') TO ('2024-05-06 12:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240506_1100
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240506_1200 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-05-06 12:00:00') TO ('2024-05-06 13:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240506_1200
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240506_1300 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-05-06 13:00:00') TO ('2024-05-06 14:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240506_1300
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240506_1400 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-05-06 14:00:00') TO ('2024-05-06 15:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240506_1400
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240506_1500 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-05-06 15:00:00') TO ('2024-05-06 16:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240506_1500
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240506_1600 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-05-06 16:00:00') TO ('2024-05-06 17:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240506_1600
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240506_1700 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-05-06 17:00:00') TO ('2024-05-06 18:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240506_1700
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240506_1800 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-05-06 18:00:00') TO ('2024-05-06 19:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240506_1800
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240506_1900 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-05-06 19:00:00') TO ('2024-05-06 20:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240506_1900
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240506_2000 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-05-06 20:00:00') TO ('2024-05-06 21:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240506_2000
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240506_2100 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-05-06 21:00:00') TO ('2024-05-06 22:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240506_2100
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240506_2200 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-05-06 22:00:00') TO ('2024-05-06 23:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240506_2200
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240506_2300 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-05-06 23:00:00') TO ('2024-05-07 00:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240506_2300
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240507_0000 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-05-07 00:00:00') TO ('2024-05-07 01:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240507_0000
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240507_0100 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-05-07 01:00:00') TO ('2024-05-07 02:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240507_0100
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240507_0200 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-05-07 02:00:00') TO ('2024-05-07 03:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240507_0200
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240507_0300 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-05-07 03:00:00') TO ('2024-05-07 04:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240507_0300
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240507_0400 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-05-07 04:00:00') TO ('2024-05-07 05:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240507_0400
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240507_0500 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-05-07 05:00:00') TO ('2024-05-07 06:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240507_0500
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240507_0600 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-05-07 06:00:00') TO ('2024-05-07 07:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240507_0600
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240507_0700 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-05-07 07:00:00') TO ('2024-05-07 08:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240507_0700
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240507_0800 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-05-07 08:00:00') TO ('2024-05-07 09:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240507_0800
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240507_0900 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-05-07 09:00:00') TO ('2024-05-07 10:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240507_0900
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240507_1000 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-05-07 10:00:00') TO ('2024-05-07 11:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240507_1000
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240507_1100 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-05-07 11:00:00') TO ('2024-05-07 12:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240507_1100
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240507_1200 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-05-07 12:00:00') TO ('2024-05-07 13:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240507_1200
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240507_1300 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-05-07 13:00:00') TO ('2024-05-07 14:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240507_1300
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240507_1400 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-05-07 14:00:00') TO ('2024-05-07 15:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240507_1400
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240507_1500 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-05-07 15:00:00') TO ('2024-05-07 16:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240507_1500
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240507_1600 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-05-07 16:00:00') TO ('2024-05-07 17:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240507_1600
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240507_1700 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-05-07 17:00:00') TO ('2024-05-07 18:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240507_1700
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240507_1800 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-05-07 18:00:00') TO ('2024-05-07 19:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240507_1800
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240507_1900 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-05-07 19:00:00') TO ('2024-05-07 20:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240507_1900
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240507_2000 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-05-07 20:00:00') TO ('2024-05-07 21:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240507_2000
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240507_2100 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-05-07 21:00:00') TO ('2024-05-07 22:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240507_2100
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240507_2200 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-05-07 22:00:00') TO ('2024-05-07 23:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240507_2200
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240507_2300 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-05-07 23:00:00') TO ('2024-05-08 00:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240507_2300
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240508_0000 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-05-08 00:00:00') TO ('2024-05-08 01:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240508_0000
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240508_0100 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-05-08 01:00:00') TO ('2024-05-08 02:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240508_0100
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240508_0200 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-05-08 02:00:00') TO ('2024-05-08 03:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240508_0200
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240508_0300 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-05-08 03:00:00') TO ('2024-05-08 04:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240508_0300
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240508_0400 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-05-08 04:00:00') TO ('2024-05-08 05:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240508_0400
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240508_0500 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-05-08 05:00:00') TO ('2024-05-08 06:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240508_0500
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240508_0600 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-05-08 06:00:00') TO ('2024-05-08 07:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240508_0600
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240508_0700 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-05-08 07:00:00') TO ('2024-05-08 08:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240508_0700
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240508_0800 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-05-08 08:00:00') TO ('2024-05-08 09:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240508_0800
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240508_0900 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-05-08 09:00:00') TO ('2024-05-08 10:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240508_0900
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240508_1000 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-05-08 10:00:00') TO ('2024-05-08 11:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240508_1000
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240508_1100 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-05-08 11:00:00') TO ('2024-05-08 12:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240508_1100
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240508_1200 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-05-08 12:00:00') TO ('2024-05-08 13:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240508_1200
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240508_1300 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-05-08 13:00:00') TO ('2024-05-08 14:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240508_1300
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240508_1400 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-05-08 14:00:00') TO ('2024-05-08 15:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240508_1400
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240508_1500 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-05-08 15:00:00') TO ('2024-05-08 16:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240508_1500
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240508_1600 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-05-08 16:00:00') TO ('2024-05-08 17:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240508_1600
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240508_1700 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-05-08 17:00:00') TO ('2024-05-08 18:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240508_1700
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240508_1800 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-05-08 18:00:00') TO ('2024-05-08 19:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240508_1800
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240508_1900 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-05-08 19:00:00') TO ('2024-05-08 20:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240508_1900
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240508_2000 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-05-08 20:00:00') TO ('2024-05-08 21:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240508_2000
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240508_2100 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-05-08 21:00:00') TO ('2024-05-08 22:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240508_2100
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240508_2200 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-05-08 22:00:00') TO ('2024-05-08 23:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240508_2200
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240508_2300 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-05-08 23:00:00') TO ('2024-05-09 00:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240508_2300
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240509_0000 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-05-09 00:00:00') TO ('2024-05-09 01:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240509_0000
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240509_0100 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-05-09 01:00:00') TO ('2024-05-09 02:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240509_0100
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240509_0200 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-05-09 02:00:00') TO ('2024-05-09 03:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240509_0200
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240509_0300 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-05-09 03:00:00') TO ('2024-05-09 04:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240509_0300
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240509_0400 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-05-09 04:00:00') TO ('2024-05-09 05:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240509_0400
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240509_0500 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-05-09 05:00:00') TO ('2024-05-09 06:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240509_0500
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240509_0600 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-05-09 06:00:00') TO ('2024-05-09 07:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240509_0600
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240509_0700 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-05-09 07:00:00') TO ('2024-05-09 08:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240509_0700
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240509_0800 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-05-09 08:00:00') TO ('2024-05-09 09:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240509_0800
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240509_0900 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-05-09 09:00:00') TO ('2024-05-09 10:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240509_0900
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240509_1000 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-05-09 10:00:00') TO ('2024-05-09 11:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240509_1000
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240509_1100 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-05-09 11:00:00') TO ('2024-05-09 12:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240509_1100
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240509_1200 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-05-09 12:00:00') TO ('2024-05-09 13:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240509_1200
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240509_1300 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-05-09 13:00:00') TO ('2024-05-09 14:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240509_1300
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240509_1400 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-05-09 14:00:00') TO ('2024-05-09 15:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240509_1400
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240509_1500 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-05-09 15:00:00') TO ('2024-05-09 16:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240509_1500
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240509_1600 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-05-09 16:00:00') TO ('2024-05-09 17:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240509_1600
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240509_1700 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-05-09 17:00:00') TO ('2024-05-09 18:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240509_1700
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240509_1800 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-05-09 18:00:00') TO ('2024-05-09 19:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240509_1800
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240509_1900 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-05-09 19:00:00') TO ('2024-05-09 20:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240509_1900
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240509_2000 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-05-09 20:00:00') TO ('2024-05-09 21:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240509_2000
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240509_2100 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-05-09 21:00:00') TO ('2024-05-09 22:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240509_2100
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240509_2200 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-05-09 22:00:00') TO ('2024-05-09 23:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240509_2200
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240509_2300 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-05-09 23:00:00') TO ('2024-05-10 00:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240509_2300
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240510_0000 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-05-10 00:00:00') TO ('2024-05-10 01:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240510_0000
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240510_0100 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-05-10 01:00:00') TO ('2024-05-10 02:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240510_0100
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240510_0200 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-05-10 02:00:00') TO ('2024-05-10 03:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240510_0200
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240510_0300 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-05-10 03:00:00') TO ('2024-05-10 04:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240510_0300
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240510_0400 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-05-10 04:00:00') TO ('2024-05-10 05:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240510_0400
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240510_0500 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-05-10 05:00:00') TO ('2024-05-10 06:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240510_0500
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240510_0600 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-05-10 06:00:00') TO ('2024-05-10 07:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240510_0600
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240510_0700 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-05-10 07:00:00') TO ('2024-05-10 08:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240510_0700
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240510_0800 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-05-10 08:00:00') TO ('2024-05-10 09:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240510_0800
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240510_0900 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-05-10 09:00:00') TO ('2024-05-10 10:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240510_0900
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240510_1000 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-05-10 10:00:00') TO ('2024-05-10 11:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240510_1000
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240510_1100 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-05-10 11:00:00') TO ('2024-05-10 12:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240510_1100
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240510_1200 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-05-10 12:00:00') TO ('2024-05-10 13:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240510_1200
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240510_1300 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-05-10 13:00:00') TO ('2024-05-10 14:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240510_1300
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240510_1400 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-05-10 14:00:00') TO ('2024-05-10 15:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240510_1400
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240510_1500 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-05-10 15:00:00') TO ('2024-05-10 16:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240510_1500
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240510_1600 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-05-10 16:00:00') TO ('2024-05-10 17:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240510_1600
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240510_1700 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-05-10 17:00:00') TO ('2024-05-10 18:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240510_1700
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240510_1800 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-05-10 18:00:00') TO ('2024-05-10 19:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240510_1800
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240510_1900 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-05-10 19:00:00') TO ('2024-05-10 20:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240510_1900
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240510_2000 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-05-10 20:00:00') TO ('2024-05-10 21:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240510_2000
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240510_2100 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-05-10 21:00:00') TO ('2024-05-10 22:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240510_2100
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240510_2200 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-05-10 22:00:00') TO ('2024-05-10 23:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240510_2200
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240510_2300 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-05-10 23:00:00') TO ('2024-05-11 00:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240510_2300
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240511_0000 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-05-11 00:00:00') TO ('2024-05-11 01:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240511_0000
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240511_0100 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-05-11 01:00:00') TO ('2024-05-11 02:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240511_0100
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240511_0200 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-05-11 02:00:00') TO ('2024-05-11 03:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240511_0200
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240511_0300 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-05-11 03:00:00') TO ('2024-05-11 04:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240511_0300
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240511_0400 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-05-11 04:00:00') TO ('2024-05-11 05:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240511_0400
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240511_0500 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-05-11 05:00:00') TO ('2024-05-11 06:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240511_0500
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240511_0600 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-05-11 06:00:00') TO ('2024-05-11 07:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240511_0600
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240511_0700 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-05-11 07:00:00') TO ('2024-05-11 08:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240511_0700
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240511_0800 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-05-11 08:00:00') TO ('2024-05-11 09:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240511_0800
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240511_0900 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-05-11 09:00:00') TO ('2024-05-11 10:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240511_0900
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240511_1000 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-05-11 10:00:00') TO ('2024-05-11 11:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240511_1000
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240511_1100 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-05-11 11:00:00') TO ('2024-05-11 12:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240511_1100
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240511_1200 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-05-11 12:00:00') TO ('2024-05-11 13:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240511_1200
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240511_1300 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-05-11 13:00:00') TO ('2024-05-11 14:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240511_1300
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240511_1400 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-05-11 14:00:00') TO ('2024-05-11 15:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240511_1400
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240511_1500 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-05-11 15:00:00') TO ('2024-05-11 16:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240511_1500
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240511_1600 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-05-11 16:00:00') TO ('2024-05-11 17:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240511_1600
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240511_1700 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-05-11 17:00:00') TO ('2024-05-11 18:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240511_1700
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240511_1800 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-05-11 18:00:00') TO ('2024-05-11 19:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240511_1800
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240511_1900 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-05-11 19:00:00') TO ('2024-05-11 20:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240511_1900
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240511_2000 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-05-11 20:00:00') TO ('2024-05-11 21:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240511_2000
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240511_2100 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-05-11 21:00:00') TO ('2024-05-11 22:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240511_2100
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240511_2200 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-05-11 22:00:00') TO ('2024-05-11 23:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240511_2200
    OWNER to postgres;
CREATE TABLE public.vehicle_tracking_20240511_2300 PARTITION OF public.vehicle_tracking
    FOR VALUES FROM ('2024-05-11 23:00:00') TO ('2024-05-12 00:00:00')
TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.vehicle_tracking_20240511_2300
    OWNER to postgres;