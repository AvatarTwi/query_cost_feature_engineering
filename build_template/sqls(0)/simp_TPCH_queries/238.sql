EXPLAIN (ANALYZE,VERBOSE,COSTS,BUFFERS,TIMING,SUMMARY,FORMAT JSON)  SELECT COUNT(*)  FROM supplier  WHERE   supplier.s_acctbal >= 2849 