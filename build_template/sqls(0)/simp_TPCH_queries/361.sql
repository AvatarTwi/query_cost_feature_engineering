EXPLAIN (ANALYZE,VERBOSE,COSTS,BUFFERS,TIMING,SUMMARY,FORMAT JSON)  SELECT *  FROM lineitem  JOIN orders  ON lineitem.l_orderkey = orders.o_orderkey  WHERE   lineitem.l_linenumber >= 1 