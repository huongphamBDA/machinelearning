#!/bin/bash

echo "Make sure to give mariadb time to come up before connecting to it"
sleep 10

mysql -h mariadb -u root -pfinalproject -e "USE baseball" 2> /dev/null
if [ $? -eq 1 ]; then
  echo "Create and load data to database baseball ..."
  mysql -h mariadb -u root -pfinalproject -e "CREATE DATABASE baseball"
  mysql -h mariadb -u root -pfinalproject baseball < /app/baseball.sql
fi

echo "Calling sql script to create final project data ..."
mysql -h mariadb -u root -pfinalproject baseball < /app/final_project.sql

echo "Run final project for feature engineering and ML models in Python ..."
python3 final_project_pham.py

echo "Done!"
