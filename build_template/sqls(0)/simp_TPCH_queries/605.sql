EXPLAIN (ANALYZE,VERBOSE,COSTS,BUFFERS,TIMING,SUMMARY,FORMAT JSON)  SELECT *  FROM lineitem  JOIN partsupp  ON lineitem.l_partkey = partsupp.ps_partkey  WHERE   lineitem.l_orderkey >= 24311 