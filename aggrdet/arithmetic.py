# Created by lan at 2021/5/6
import decimal
import math
from abc import ABC, abstractmethod
from decimal import Decimal
from typing import List

from helpers import hard_empty_cell_values


class ArithmeticCalculator(ABC):

    @classmethod
    @abstractmethod
    def is_aggregation(cls, aggregatee_values: List[Decimal], actual, error_level, digit_places):
        pass


class SumCalculator(ArithmeticCalculator):

    @classmethod
    def is_aggregation(cls, aggregatee_values, actual, error_level, digit_places):
        # expected_sum += aggregatee if not aggregatee.is_nan() else Decimal(0.0)
        expected = sum([v if not v.is_nan() else Decimal(0) for v in aggregatee_values])
        is_equal = False
        if actual == 0:
            real_error_level = abs(expected - actual)
        else:
            real_error_level = abs((expected - actual) / actual)
        if real_error_level <= error_level:
            is_equal = True
        return is_equal, real_error_level


class AverageCalculator(ArithmeticCalculator):

    @classmethod
    def is_aggregation(cls, aggregatee_values, actual, error_level, digit_places):
        # if len(aggregatee_values) == 236:
        #     print('STOP')
        expected_sum = sum([v if not v.is_nan() else Decimal(0) for v in aggregatee_values])
        expected = round(expected_sum / Decimal(len(aggregatee_values)), digit_places)
        nan_count = len([v for v in aggregatee_values if v.is_nan()])

        aggregatee_length_excl_nan = Decimal(len(aggregatee_values) - nan_count)
        expected_excl_nan = round(expected_sum / aggregatee_length_excl_nan, digit_places) if aggregatee_length_excl_nan > 0 else Decimal(math.inf)
        is_equal = False
        if actual == 0:
            real_error_level = min(abs(expected - actual), abs(expected_excl_nan - actual))
        else:
            real_error_level = min(abs((expected - actual) / actual), abs((expected_excl_nan - actual) / actual))
        if real_error_level <= error_level:
            # Todo: currently, if every aggregatee cell has the same value, treat it not a valid aggregation
            if len(set([elem for elem in aggregatee_values])) != 1:
                is_equal = True
        return is_equal, real_error_level


class SubtractionCalculator(ArithmeticCalculator):

    @classmethod
    def is_aggregation(cls, aggregatee_values: List[Decimal], actual, error_level, digit_places):
        if len(aggregatee_values) != 2:
            raise RuntimeError('Number of aggregatee elements (%s) improper for this aggregation type.' % len(aggregatee_values))
        is_equal = False
        if actual == 0 and all([aee_cell in hard_empty_cell_values or aee_cell == 0 for aee_cell in aggregatee_values]):
            # do not allow all zero subtraction
            real_error_level = math.inf
        else:
            expected_value = aggregatee_values[0] - aggregatee_values[1]

            if actual == Decimal(0):
                real_error_level = abs(actual - expected_value)
            else:
                real_error_level = abs((actual - expected_value) / actual)

            if real_error_level <= error_level:
                is_equal = True
        return is_equal, real_error_level


class DivisionCalculator(ArithmeticCalculator):

    @classmethod
    def is_aggregation(cls, aggregatee_values: List[Decimal], actual, error_level, digit_places):
        if len(aggregatee_values) != 2:
            raise RuntimeError('Number of aggregatee elements (%s) improper for this aggregation type.' % len(aggregatee_values))
        is_equal = False
        if all([aee_cell in hard_empty_cell_values or aee_cell == 0 for aee_cell in aggregatee_values]):
            # do not allow all zero subtraction
            real_error_level = math.inf
        else:
            try:
                expected_value = aggregatee_values[0] / aggregatee_values[1]
                # expected_value = round(aggregatee_values[0] / aggregatee_values[1], digit_places)
            except (decimal.DivisionByZero, decimal.InvalidOperation):
                real_error_level = math.inf
            else:
                if actual == Decimal(0):
                    real_error_level = min(abs(actual - expected_value), abs(actual / 100 - expected_value))
                else:
                    real_error_level = min(abs((actual - expected_value) / actual),
                                           abs((actual / 100 - expected_value) / (actual / 100)))

                if real_error_level <= error_level:
                    is_equal = True
        return is_equal, real_error_level


class RelativeChangeCalculator(ArithmeticCalculator):

    @classmethod
    def is_aggregation(cls, aggregatee_values: List[Decimal], actual, error_level, digit_places):
        if len(aggregatee_values) != 2:
            raise RuntimeError('Number of aggregatee elements (%s) improper for this aggregation type.' % len(aggregatee_values))
        is_equal = False
        if all([aee_cell in hard_empty_cell_values or aee_cell == 0 for aee_cell in aggregatee_values]):
            # do not allow all zero subtraction
            real_error_level = math.inf
        try:
            expected_value = (aggregatee_values[1] - aggregatee_values[0]) / aggregatee_values[0]
            # expected_value = round((aggregatee_values[1] - aggregatee_values[0]) / aggregatee_values[0], digit_places)
        except (decimal.DivisionByZero, decimal.InvalidOperation):
            real_error_level = math.inf
        else:
            if actual == Decimal(0):
                real_error_level = min(abs(actual - expected_value), abs(actual / 100 - expected_value))
            else:
                # real_error_level = min(abs((actual - expected_value) / actual),
                #                        abs((actual / 100 - expected_value) / (actual / 100)))
                if 1 < actual <= 100:
                    real_error_level = abs((actual / 100 - expected_value) / (actual / 100))
                else:
                    real_error_level = min(abs((actual - expected_value) / actual),
                                           abs((actual / 100 - expected_value) / (actual / 100)))

                if real_error_level <= error_level:
                    is_equal = True
        return is_equal, real_error_level
