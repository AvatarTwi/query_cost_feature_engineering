EXPLAIN (ANALYZE,VERBOSE,COSTS,BUFFERS,TIMING,SUMMARY,FORMAT JSON)  SELECT *  FROM title  JOIN movie_info_idx  ON title.id = movie_info_idx.movie_id  WHERE   title.season_nr >= 416 