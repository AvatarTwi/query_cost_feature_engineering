EXPLAIN (ANALYZE,VERBOSE,COSTS,BUFFERS,TIMING,SUMMARY,FORMAT JSON)  SELECT *  FROM supplier  WHERE  supplier.s_suppkey >= 139  ORDER BY s_suppkey 