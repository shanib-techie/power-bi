import mysql.connector

# 1. MySQL connection
conn = mysql.connector.connect(
    host="127.0.0.1",
    user="root",
    password = "SH@nib123",
    database="project_db"
)

cursor = conn.cursor()
# no. of entries

# 2. User input
name = input("Enter name: ")
age = int(input("Enter age: "))
marks = int(input("Enter marks: "))

# 3. Insert query
query = "INSERT INTO students (name, age, marks) VALUES (%s, %s, %s)"
values = (name, age, marks)

cursor.execute(query, values)

# 4. Save changes
conn.commit()

print("Data inserted successfully!")

# 5. Close connection
cursor.close()
conn.close()
