EXPLAIN (ANALYZE,VERBOSE,COSTS,BUFFERS,TIMING,SUMMARY,FORMAT JSON)  SELECT *  FROM customer  JOIN orders  ON customer.c_custkey = orders.o_custkey  WHERE  customer.c_custkey >= 3994 