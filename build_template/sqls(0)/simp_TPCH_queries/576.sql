EXPLAIN (ANALYZE,VERBOSE,COSTS,BUFFERS,TIMING,SUMMARY,FORMAT JSON)  SELECT *  FROM part  JOIN lineitem  ON part.p_partkey = lineitem.l_partkey  WHERE  part.p_size >= 20 