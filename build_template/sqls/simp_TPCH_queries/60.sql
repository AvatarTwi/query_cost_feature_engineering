EXPLAIN (ANALYZE,VERBOSE,COSTS,BUFFERS,TIMING,SUMMARY,FORMAT JSON)  SELECT *  FROM partsupp  JOIN supplier  ON partsupp.ps_suppkey = supplier.s_suppkey  WHERE  partsupp.ps_partkey >= 15675 