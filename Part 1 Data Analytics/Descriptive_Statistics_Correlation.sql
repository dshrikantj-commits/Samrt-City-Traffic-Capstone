SELECT YearValue,
    AVG(traffic_volume) AS Mean,
    MIN(traffic_volume) AS Lowest,
    MAX(traffic_volume) AS Highest
	
FROM
	Metro_Interstate_Traffic_Volume
GROUP BY
	YearValue
