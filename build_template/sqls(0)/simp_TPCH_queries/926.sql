EXPLAIN (ANALYZE,VERBOSE,COSTS,BUFFERS,TIMING,SUMMARY,FORMAT JSON)  SELECT *  FROM supplier  WHERE  supplier.s_acctbal >= 2089  ORDER BY s_acctbal 