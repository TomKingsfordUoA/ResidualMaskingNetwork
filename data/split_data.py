import csv
import typing

import pandas as pd
from sklearn.model_selection import train_test_split


if __name__ == '__main__':
    n_downsample: typing.Optional[int] = None

    raw = pd.read_csv('fer2013_extracted/fer2013.csv')
    train_raw = raw[raw['Usage'] == 'Training'].drop(columns='Usage')
    test = raw[raw['Usage'] == 'PublicTest'].drop(columns='Usage')

    if n_downsample:
        train_raw = train_raw.sample(n=n_downsample)
        test = test.sample(n=n_downsample)

    train, val = train_test_split(train_raw, test_size=0.2, random_state=42)
    csv_params = {
        'index': False,
        'header': True,
        'quoting': csv.QUOTE_NONNUMERIC,
    }
    train.to_csv('train.csv', **csv_params)
    val.to_csv('val.csv', **csv_params)
    test.to_csv('test.csv', **csv_params)
