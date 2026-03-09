📌 A) Data Retrieval Queries — Single Table				
                
#	Category	Query Type	Description	Example Pattern
1	Basic SELECT	Select all rows	Retrieve all records	SELECT * FROM table;
2	Basic SELECT	Select specific columns	Retrieve selected fields	SELECT col1, col2 FROM table;
3	Filtering	WHERE equality	Exact match filtering	WHERE col = 'value'
4	Filtering	Comparison operators	>, <, >=, <=	WHERE amount > 1000
5	Filtering	Logical conditions	AND, OR, NOT	WHERE a=1 AND b=2
6	Filtering	IN	Match multiple values	WHERE state IN ('AP','TS')
7	Filtering	BETWEEN	Range filtering	WHERE date BETWEEN d1 AND d2
8	Filtering	LIKE / ILIKE	Pattern matching	WHERE name LIKE 'S%'
9	Filtering	IS NULL	Null checking	WHERE email IS NULL
10	Sorting	ORDER BY	Sorting results	ORDER BY created_at DESC
11	Pagination	LIMIT	Restrict rows	LIMIT 100
12	Pagination	OFFSET	Skip rows	LIMIT 100 OFFSET 200
13	Aggregation	COUNT	Count rows	SELECT COUNT(*)
14	Aggregation	SUM	Total values	SELECT SUM(amount)
15	Aggregation	AVG	Average values	SELECT AVG(balance)
16	Aggregation	MIN / MAX	Min/Max value	SELECT MAX(balance)
17	Aggregation	GROUP BY	Aggregate per group	GROUP BY state
18	Aggregation	HAVING	Filter aggregated data	HAVING COUNT(*) > 10
19	Distinct	DISTINCT	Unique values	SELECT DISTINCT state
20	Window	ROW_NUMBER()	Sequential numbering	ROW_NUMBER() OVER (...)
21	Window	RANK()	Ranking values	RANK() OVER (...)
22	Window	Running total	Cumulative sum	SUM(col) OVER (...)
23	Window	Partitioning	Partitioned analytics	OVER (PARTITION BY col)
24	Subquery	Scalar subquery	Compare with derived value	WHERE col > (SELECT AVG(col)...)
25	Subquery	EXISTS	Existence check	WHERE EXISTS (SELECT 1...)
26	Subquery	IN subquery	Match via subquery	WHERE id IN (SELECT id...)
27	Conditional	CASE	Conditional logic	CASE WHEN col>100 THEN 'High'
28	Expression	COALESCE	Replace NULL	COALESCE(col,'N/A')
29	JSON (PG)	JSON extraction	Retrieve JSON fields	json_col->>'key'
30	Array (PG)	Array filtering	Filter array contents	WHERE tags @> ARRAY['vip']
                
                
📌 B) Data Retrieval Queries — Multiple Tables				
                
#	Category	Query Type	Description	Example Pattern
1	Join	INNER JOIN	Matching rows in both tables	A JOIN B ON A.id=B.id
2	Join	LEFT JOIN	All left + matched right	A LEFT JOIN B ON ...
3	Join	RIGHT JOIN	All right + matched left	A RIGHT JOIN B ON ...
4	Join	FULL OUTER JOIN	All rows from both	A FULL JOIN B ON ...
5	Join	CROSS JOIN	Cartesian product	A CROSS JOIN B
6	Join	SELF JOIN	Table joined to itself	A a1 JOIN A a2 ON ...
7	Join	Multi-table JOIN	Join 3+ tables	A JOIN B JOIN C
8	Filtering	WHERE on joined table	Filter using related table	WHERE B.status='ACTIVE'
9	Filtering	EXISTS with join	Semi-join pattern	WHERE EXISTS (SELECT 1 FROM B WHERE B.id=A.id)
10	Filtering	NOT EXISTS	Anti-join pattern	WHERE NOT EXISTS (...)
11	Aggregation	COUNT related rows	Count children per parent	COUNT(B.id)
12	Aggregation	SUM across join	Aggregate related data	SUM(T.amount)
13	Aggregation	GROUP BY joined column	Aggregate across entities	GROUP BY B.type
14	Aggregation	HAVING across join	Filter grouped join data	HAVING COUNT(B.id)>5
15	Window	Ranking per group	Rank inside joined data	RANK() OVER (PARTITION BY A.id)
16	Window	Running total per group	Cumulative analytics	SUM(...) OVER (PARTITION BY A.id)
17	Subquery	Correlated subquery	Subquery referencing outer table	WHERE A.amount > (SELECT AVG(...) FROM B WHERE B.id=A.id)
18	CTE	WITH clause	Temporary result set	WITH cte AS (...) SELECT ...
19	Recursive CTE	Hierarchy queries	Tree traversal	WITH RECURSIVE ...
20	Set Ops	UNION	Combine result sets	SELECT ... FROM A UNION SELECT ... FROM B
21	Set Ops	UNION ALL	Combine with duplicates	UNION ALL
22	Set Ops	INTERSECT	Common rows	INTERSECT
23	Set Ops	EXCEPT	Difference	EXCEPT
24	LATERAL JOIN	Derived subquery per row	JOIN LATERAL (...)	
25	JSON Join	Join via JSON key	Join using JSON field	
