EXPLAIN (ANALYZE,VERBOSE,COSTS,BUFFERS,TIMING,SUMMARY,FORMAT JSON)  SELECT *  FROM partsupp  JOIN lineitem  ON partsupp.ps_partkey = lineitem.l_partkey  WHERE  partsupp.ps_suppkey >= 405 