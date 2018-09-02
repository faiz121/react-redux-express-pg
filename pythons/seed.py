#!/usr/bin/python3.5
#
# Small script to show PostgreSQL and Pyscopg together
#

import psycopg2
import os
import numpy as np
from config import db_config
import sys
filename = sys.argv[1]

try:
    conn = psycopg2.connect(db_config)
    print("connected!")
except:
    print("I am unable to connect to the database")

cur = conn.cursor()

def seed(filepath):
    data_file = open(filepath, 'r')
    image_list = data_file.readlines()
    data_file.close

    for i in range(0, len(image_list)):
        print("iteration: ", i)
        feature_sets_and_label = image_list[i].split(',')

        # Create raw_features (array of 784) with pixel values between 0 and 255
        raw_features = feature_sets_and_label[1:]

        # Create normalized_features (array of 784) 0 and 1
        normalized_features = np.asfarray(raw_features) / 255.0 * 1

        # Create one_hot_label (array of 10) with the index at the labels value as a 1
        # [0, 0, 1, 0, 0, 0, 0, 0, 0, 0] represents a 2
        raw_label = feature_sets_and_label[0]
        label = np.zeros(10)
        label[int(raw_label)] = 1

        # Source is mnist data set
        source = "mnist"
        serialized_features = str(normalized_features.tolist())
        serialized_label = str(label.tolist())

        # try:
        cur.execute("""INSERT INTO images (features, label, source) VALUES (%s, %s, %s);""", (serialized_features, serialized_label, source))
        conn.commit()
        # except psycopg2.DatabaseError, e:
        #     if conn:
        #         conn.rollback()

        #     print('Error %s', e)
        #     sys.exit(1)

seed(filename)
cur.close();
conn.close();

# python3 ./seed.py data/mnist_test_10.csv
