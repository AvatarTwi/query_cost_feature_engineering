EXPLAIN (ANALYZE,VERBOSE,COSTS,BUFFERS,TIMING,SUMMARY,FORMAT JSON)  SELECT *  FROM nation  WHERE  nation.n_nationkey >= 13  ORDER BY n_name 