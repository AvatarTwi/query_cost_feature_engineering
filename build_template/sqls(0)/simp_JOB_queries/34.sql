EXPLAIN (ANALYZE,VERBOSE,COSTS,BUFFERS,TIMING,SUMMARY,FORMAT JSON)  SELECT *  FROM title  JOIN cast_info  ON title.id = cast_info.movie_id  WHERE   title.id >= 1542848 