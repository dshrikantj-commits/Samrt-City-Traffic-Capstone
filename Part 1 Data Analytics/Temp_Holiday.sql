
SELECT *
FROM Metro_Interstate_Traffic_Volume 
WHERE (holiday = "Labor Day" OR holiday = "New Years Day") AND (YearValue = 2015 OR YearValue = 2016 OR YearValue = 2017);

