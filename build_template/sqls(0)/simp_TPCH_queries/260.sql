EXPLAIN (ANALYZE,VERBOSE,COSTS,BUFFERS,TIMING,SUMMARY,FORMAT JSON)  SELECT COUNT(*)  FROM part  WHERE  part.p_partkey >= 626 