EXPLAIN (ANALYZE,VERBOSE,COSTS,BUFFERS,TIMING,SUMMARY,FORMAT JSON)  SELECT *  FROM title  JOIN movie_keyword  ON title.id = movie_keyword.movie_id  WHERE  title.episode_nr >= 32562 