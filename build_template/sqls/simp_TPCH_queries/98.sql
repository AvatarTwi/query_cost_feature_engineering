EXPLAIN (ANALYZE,VERBOSE,COSTS,BUFFERS,TIMING,SUMMARY,FORMAT JSON)  SELECT *  FROM nation  WHERE   nation.n_nationkey >= 10  ORDER BY n_name 