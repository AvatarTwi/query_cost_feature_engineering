EXPLAIN (ANALYZE,VERBOSE,COSTS,BUFFERS,TIMING,SUMMARY,FORMAT JSON)  SELECT *  FROM title  JOIN movie_info  ON title.id = movie_info.movie_id  WHERE  title.kind_id >= 1 