EXPLAIN (ANALYZE,VERBOSE,COSTS,BUFFERS,TIMING,SUMMARY,FORMAT JSON)  SELECT *  FROM lineitem  WHERE  lineitem.l_quantity >= 6  ORDER BY l_shipinstruct 