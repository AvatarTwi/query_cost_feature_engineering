EXPLAIN (ANALYZE,VERBOSE,COSTS,BUFFERS,TIMING,SUMMARY,FORMAT JSON)  SELECT *  FROM part  WHERE   part.p_partkey >= 8721  ORDER BY p_partkey 