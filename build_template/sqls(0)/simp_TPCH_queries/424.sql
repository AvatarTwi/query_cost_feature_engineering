EXPLAIN (ANALYZE,VERBOSE,COSTS,BUFFERS,TIMING,SUMMARY,FORMAT JSON)  SELECT *  FROM nation  JOIN customer  ON nation.n_nationkey = customer.c_nationkey  WHERE  nation.n_regionkey >= 3 