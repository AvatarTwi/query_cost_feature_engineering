EXPLAIN (ANALYZE,VERBOSE,COSTS,BUFFERS,TIMING,SUMMARY,FORMAT JSON)  SELECT *  FROM supplier  JOIN partsupp  ON supplier.s_suppkey = partsupp.ps_suppkey  WHERE  supplier.s_acctbal >= 1872 