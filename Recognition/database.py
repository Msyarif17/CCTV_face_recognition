import mysql.connector
import sys

mydb = mysql.connector.connect(
  host="localhost",
  user="root",
  password=""
)
sql = sys.argv[1]
mycursor = mydb.cursor()

mycursor.execute("{0}%".format(sql))