EXPLAIN (ANALYZE,VERBOSE,COSTS,BUFFERS,TIMING,SUMMARY,FORMAT JSON)  SELECT *  FROM part  WHERE  part.p_retailprice >= 1778  ORDER BY p_type 