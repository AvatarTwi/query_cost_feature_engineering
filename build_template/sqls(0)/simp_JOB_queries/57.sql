EXPLAIN (ANALYZE,VERBOSE,COSTS,BUFFERS,TIMING,SUMMARY,FORMAT JSON)  SELECT *  FROM title  JOIN movie_companies  ON title.id = movie_companies.movie_id  WHERE   title.id >= 1898746 