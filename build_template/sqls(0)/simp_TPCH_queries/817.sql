EXPLAIN (ANALYZE,VERBOSE,COSTS,BUFFERS,TIMING,SUMMARY,FORMAT JSON)  SELECT *  FROM lineitem  WHERE  lineitem.l_discount >= 0  ORDER BY l_shipdate 