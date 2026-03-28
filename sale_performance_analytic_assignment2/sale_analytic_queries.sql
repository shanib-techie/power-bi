create database  sales_project;
use sales_project;


select * from products;


-- total 4 result ayy h total revenue total deal avg deal amount and last wale me yeh h ki kitni kaamiyaab hue h outoff all
SELECT 
    SUM(amount) AS total_revenue,
    COUNT(*) AS total_deals,
    min(amount) as less_amount_deal,
    max(amount) as maximun_amount_deal,
    AVG(amount) AS avg_deal_size,
    SUM(CASE WHEN stage = 'Closed Won' THEN 1 ELSE 0 END)*100 / COUNT(*) AS win_rate
FROM cleaned_sales_data;





SELECT 
    r.region,
    COUNT(*) AS total_deals,
    SUM(s.amount) AS total_revenue
FROM cleaned_sales_data s
JOIN sales_reps r ON s.sales_rep_id = r.sales_rep_id
GROUP BY r.region
ORDER BY total_revenue DESC;



SELECT 
    r.sales_rep_name,
    COUNT(*) AS deals_closed,
    SUM(s.amount) AS revenue
FROM cleaned_sales_data s
JOIN sales_reps r ON s.sales_rep_id = r.sales_rep_id
WHERE s.stage = 'Closed Won'
GROUP BY r.sales_rep_name
ORDER BY revenue DESC
LIMIT 10;


SELECT 
    stage,
    COUNT(*) AS total_count
FROM cleaned_sales_data
GROUP BY stage;


SELECT 
    p.product_name,
    SUM(s.amount) AS revenue
FROM cleaned_sales_data s
JOIN products p ON s.product_id = p.product_id
GROUP BY p.product_name
ORDER BY revenue DESC;