EXPLAIN (ANALYZE,VERBOSE,COSTS,BUFFERS,TIMING,SUMMARY,FORMAT JSON)  SELECT *  FROM part  WHERE   part.p_partkey >= 19462  ORDER BY p_partkey 