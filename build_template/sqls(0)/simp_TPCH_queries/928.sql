EXPLAIN (ANALYZE,VERBOSE,COSTS,BUFFERS,TIMING,SUMMARY,FORMAT JSON)  SELECT *  FROM supplier  WHERE   supplier.s_suppkey >= 811  ORDER BY s_acctbal 