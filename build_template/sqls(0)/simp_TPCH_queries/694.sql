EXPLAIN (ANALYZE,VERBOSE,COSTS,BUFFERS,TIMING,SUMMARY,FORMAT JSON)  SELECT *  FROM orders  JOIN lineitem  ON orders.o_orderkey = lineitem.l_orderkey  WHERE   orders.o_orderkey >= 327809 