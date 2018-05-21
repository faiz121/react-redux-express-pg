
# coding: utf-8

# In[40]:


import numpy as np
import psycopg2
from config import db_config


# In[41]:


# try:
#     conn = psycopg2.connect(db_config)
#     print("connected!")
# except:
#     print("I am unable to connect to the database")
#
# cur = conn.cursor()


# In[42]:


def convert_string_to_np_array(stringified_array):
    return np.fromstring(stringified_array[1:-1], dtype=float, sep=', ')


# In[50]:


def create_features_and_labels(query):
    try:
        conn = psycopg2.connect(db_config)
        print("connected!")
    except:
        print("I am unable to connect to the database")

    cur = conn.cursor()
    cur.execute(query)
    rows = cur.fetchall()

    features = np.array([])
    labels = np.array([])

    for row in rows:
        sample_features = convert_string_to_np_array(row[1])
        sample_label = convert_string_to_np_array(row[2])

        features = np.append(features, sample_features)
        features = np.reshape(features, (-1, 784))

        labels = np.append(labels, sample_label)
        labels = np.reshape(labels, (-1, 10))

    cur.close();
    conn.close();

    return features, labels


# features, labels = create_features_and_labels("SELECT * from images")
