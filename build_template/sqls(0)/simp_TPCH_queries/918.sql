EXPLAIN (ANALYZE,VERBOSE,COSTS,BUFFERS,TIMING,SUMMARY,FORMAT JSON)  SELECT *  FROM supplier  WHERE   supplier.s_acctbal >= 6429  ORDER BY s_suppkey 