EXPLAIN (ANALYZE,VERBOSE,COSTS,BUFFERS,TIMING,SUMMARY,FORMAT JSON)  SELECT COUNT(*)  FROM lineitem  WHERE  lineitem.l_suppkey >= 689 