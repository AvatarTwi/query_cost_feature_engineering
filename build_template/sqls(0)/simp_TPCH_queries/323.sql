EXPLAIN (ANALYZE,VERBOSE,COSTS,BUFFERS,TIMING,SUMMARY,FORMAT JSON)  SELECT COUNT(*)  FROM orders  WHERE  orders.o_custkey >= 12651 