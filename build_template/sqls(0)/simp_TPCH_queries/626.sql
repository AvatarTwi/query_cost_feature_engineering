EXPLAIN (ANALYZE,VERBOSE,COSTS,BUFFERS,TIMING,SUMMARY,FORMAT JSON)  SELECT *  FROM supplier  JOIN lineitem  ON supplier.s_suppkey = lineitem.l_suppkey  WHERE   supplier.s_acctbal >= 3659 