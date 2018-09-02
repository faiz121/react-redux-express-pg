import psycopg2
import os
import numpy as np
from config import db_config
import sys
try:
    conn = psycopg2.connect(db_config)
    print("connected!")
except:
    print("I am unable to connect to the database")

cur = conn.cursor()

def add_to_db(features, label, source):
    cur.execute("""INSERT INTO images (features, label, source) VALUES (%s, %s, %s);""", (features, label, source))
    conn.commit()
    # cur.close();
    # conn.close();
    print("features, labels, source: ", features, label, source)
    return source
