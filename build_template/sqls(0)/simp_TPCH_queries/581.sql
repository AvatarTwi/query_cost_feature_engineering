EXPLAIN (ANALYZE,VERBOSE,COSTS,BUFFERS,TIMING,SUMMARY,FORMAT JSON)  SELECT *  FROM orders  JOIN customer  ON orders.o_custkey = customer.c_custkey  WHERE  orders.o_orderkey >= 305083 