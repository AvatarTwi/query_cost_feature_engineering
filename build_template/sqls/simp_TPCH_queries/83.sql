EXPLAIN (ANALYZE,VERBOSE,COSTS,BUFFERS,TIMING,SUMMARY,FORMAT JSON)  SELECT *  FROM lineitem  WHERE   lineitem.l_linenumber >= 2  ORDER BY l_quantity 