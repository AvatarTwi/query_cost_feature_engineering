EXPLAIN (ANALYZE,VERBOSE,COSTS,BUFFERS,TIMING,SUMMARY,FORMAT JSON)  SELECT *  FROM nation  JOIN supplier  ON nation.n_nationkey = supplier.s_nationkey  WHERE  nation.n_nationkey >= 14 