EXPLAIN (ANALYZE,VERBOSE,COSTS,BUFFERS,TIMING,SUMMARY,FORMAT JSON)  SELECT *  FROM supplier  JOIN nation  ON supplier.s_nationkey = nation.n_nationkey  WHERE  supplier.s_suppkey >= 861 