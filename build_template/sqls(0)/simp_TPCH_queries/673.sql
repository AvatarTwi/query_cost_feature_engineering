EXPLAIN (ANALYZE,VERBOSE,COSTS,BUFFERS,TIMING,SUMMARY,FORMAT JSON)  SELECT *  FROM lineitem  JOIN part  ON lineitem.l_partkey = part.p_partkey  WHERE  lineitem.l_partkey >= 6049 