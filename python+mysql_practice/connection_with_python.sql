CREATE DATABASE project_db;

USE project_db;

CREATE TABLE students (
    id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(50),
    age INT,
    marks INT
);
select * from students