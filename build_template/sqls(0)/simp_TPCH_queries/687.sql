EXPLAIN (ANALYZE,VERBOSE,COSTS,BUFFERS,TIMING,SUMMARY,FORMAT JSON)  SELECT *  FROM part  JOIN partsupp  ON part.p_partkey = partsupp.ps_partkey  WHERE   part.p_partkey >= 16357 