EXPLAIN (ANALYZE,VERBOSE,COSTS,BUFFERS,TIMING,SUMMARY,FORMAT JSON)  SELECT *  FROM orders  WHERE  orders.o_orderkey >= 21960  ORDER BY o_orderpriority 