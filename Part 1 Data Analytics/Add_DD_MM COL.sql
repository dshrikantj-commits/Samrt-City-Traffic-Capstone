
SELECT strftime('%m', date_time) AS Month_Value,
       strftime('%d', date_time) AS Date_Value
FROM Metro_Interstate_Traffic_Volume;
ALTER TABLE Metro_Interstate_Traffic_Volume
ADD COLUMN Date_Value INT;

