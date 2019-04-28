#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import numpy
import pandas
import sys

from sklearn import linear_model

FEATURES = [
    'Powierzchnia w m2',
    'Liczba pokoi',
    'Liczba pięter w budynku',
    'Piętro',
    'Rok budowy',
]


def preprocess(data):
    data = data.replace({'parter': 0, 'poddasze': 0}, regex=True)
    data = data.applymap(numpy.nan_to_num)
    return data

input_filename = sys.argv[1]
output_filename = sys.argv[2]
trainset_filename = 'train/train.tsv'

data = pandas.read_csv(trainset_filename, header=0, sep='\t')
columns = data.columns[1:]
data = data[FEATURES + ['cena']]
data = preprocess(data)
y = pandas.DataFrame(data['cena'])
x = pandas.DataFrame(data[FEATURES])
model = linear_model.LinearRegression()
model.fit(x, y)

data = pandas.read_csv(input_filename, header=None, sep='\t', names=columns)
x = pandas.DataFrame(data[FEATURES])
x = preprocess(x)
y = model.predict(x)

pandas.DataFrame(y).to_csv(output_filename, index=None, header=None, sep='\t')
