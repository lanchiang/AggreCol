# Created by lan at 2021/1/2
import os
import re
from enum import Enum

import yaml

from definitions import ROOT_DIR

empty_cell_values = ['', '-', 'n/a', 'null', '.', '..', '...', 'x', 'X', '#', 'NaN', 'NA']

hard_empty_cell_values = ['']


class AggregationOperator(Enum):
    SUM = 'Sum'
    SUBTRACT = 'Subtract'
    AVERAGE = 'Average'
    RELATIVE_CHANGE = 'RelativeChange'
    DIVISION = 'Division'

    @staticmethod
    def all():
        return list(map(lambda c: c.value, AggregationOperator))


class AggregationDirection(Enum):
    ROW_WISE = 'RowWise'
    COLUMN_WISE = 'ColumnWise'


def is_empty_cell(value: str) -> bool:
    is_empty_cell_vector = [ecv == value.lower() for ecv in empty_cell_values]
    return any(is_empty_cell_vector)


def extract_dataset_name(path) -> str:
    """
    Extract dataset name from the compressed dataset file path. Note that to make it work, dataset compressed file must be named as follows: *.jl.gz

    :param path: path of the compressed dataset file
    :return: extracted dataset name
    """
    m = re.search(r'.*/(.+).jl.gz', path)
    ds_name = None
    if isinstance(m, re.Match):
        try:
            ds_name = m.group(1)
        except IndexError:
            print('Extracting dataset name failed')
            exit(1)
    return ds_name


def is_aggregation_equal(groundtruth, prediction, file_values) -> bool:
    """
    Check if two aggregations are equal to each other. Two aggregations are equal, if:
    1) the aggregators are the same
    2) the aggregatees are the same, or the difference set from prediction to groundtruth (prediction - groundtruth) contains only empty cells.

    :param groundtruth: the groundtruth aggregation, a 2-er tuple of (aggregator_index, aggregatee_indices)
    :param prediction: the prediction aggregation, a 2-er tuple of (aggregator_index, aggregatee_indices)
    :param file_values: values of the file, used to check intersections.
    :return: true if two aggregations are equal, false otherwise
    """
    if groundtruth[0] != prediction[0]:
        return False
    if groundtruth[2] != prediction[2]:
        return False

    groundtruth_aggregatee_indices = sorted(groundtruth[1])
    prediction_aggregatee_indices = sorted(prediction[1])
    if groundtruth_aggregatee_indices == prediction_aggregatee_indices:
        return True

    groundtruth_aggregatee_indices = [tuple(e) for e in groundtruth_aggregatee_indices]
    prediction_aggregatee_indices = [tuple(e) for e in prediction_aggregatee_indices]

    def diff(l1, l2):
        return list(set(l1) - set(l2))

    diff_set = diff(prediction_aggregatee_indices, groundtruth_aggregatee_indices)

    is_equal = True
    for d in diff_set:
        if file_values[d[0]][d[1]] not in hard_empty_cell_values:
            is_equal = False
            break
    return is_equal