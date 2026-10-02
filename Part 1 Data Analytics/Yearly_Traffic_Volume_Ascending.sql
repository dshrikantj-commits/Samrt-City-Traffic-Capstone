
SELECT sum (traffic_volume) AS Yearly_Traffic,YearValue
FROM
	Metro_Interstate_Traffic_Volume
GROUP BY
	YearValue
ORDER BY
	Yearly_Traffic ASC;