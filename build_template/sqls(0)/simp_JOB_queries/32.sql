EXPLAIN (ANALYZE,VERBOSE,COSTS,BUFFERS,TIMING,SUMMARY,FORMAT JSON)  SELECT *  FROM title  JOIN cast_info  ON title.id = cast_info.movie_id  WHERE  title.kind_id >= 7 