# Created by lan at 2021/5/2
import decimal
import itertools
import math
from copy import copy
from decimal import Decimal

from cacheout import FIFOCache

from aggrecol.strategy import AdjacentListAggregationDetection, SlidingWindowAggregationDetection
from arithmetic import SumCalculator, AverageCalculator, SubtractionCalculator, DivisionCalculator, RelativeChangeCalculator
from elements import CellIndex, Direction, AggregationRelation, Cell
from helpers import AggregationOperator, hard_empty_cell_values, empty_cell_values
from number import str2decimal
from tree import AggregationRelationForest


class SumDetection(AdjacentListAggregationDetection):

    def __init__(self, file, parameters) -> None:
        super().__init__(file, parameters)

        self.operator = AggregationOperator.SUM.value
        self.task_name = self.__class__.task_name

    def detect_proximity_aggregation_relations(self, forest: AggregationRelationForest, error_bound: float):
        roots = forest.get_roots()
        aggregation_candidates = []
        for index, root in enumerate(roots):
            try:
                aggregator_value = Decimal(root.value)
            except decimal.InvalidOperation:
                continue
            if aggregator_value.is_nan():
                continue

            adjusted_error_bound = error_bound

            # forward
            aggregatee_cells = []
            aggregatee_decimal_values = []
            expected_sum = Decimal(0.0)
            is_equal = False
            real_error_level = math.inf
            for i in range(index + 1, len(roots)):
                # if this cell is empty, allows to continue
                try:
                    aggregatee = Decimal(roots[i].value)
                except decimal.InvalidOperation:
                    continue
                else:
                    expected_sum += aggregatee if not aggregatee.is_nan() else Decimal(0.0)
                    aggregatee_cells.append(roots[i])
                    aggregatee_decimal_values.append(aggregatee)
                    is_equal, real_error_level = SumCalculator.is_aggregation(aggregatee_decimal_values, aggregator_value, adjusted_error_bound,
                                                                              self.DIGIT_PLACES)
                    if is_equal:
                        break
            if is_equal and len(aggregatee_cells) >= 2:
                if not (aggregator_value == 0 and all([aee_cell.value in hard_empty_cell_values or aee_cell.value == 0 for aee_cell in aggregatee_cells])):
                    ar = AggregationRelation(copy(root), tuple([copy(aee_cell) for aee_cell in aggregatee_cells]), self.operator, Direction.FORWARD)
                    aggregation_candidates.append((ar, real_error_level))

            # backward
            aggregatee_cells = []
            aggregatee_decimal_values = []
            expected_sum = Decimal(0.0)
            is_equal = False
            real_error_level = math.inf
            for i in reversed(range(index)):
                # if this cell is empty, allows to continue
                try:
                    aggregatee = Decimal(roots[i].value)
                except decimal.InvalidOperation:
                    continue
                else:
                    expected_sum += aggregatee if not aggregatee.is_nan() else Decimal(0.0)
                    aggregatee_cells.append(roots[i])
                    aggregatee_decimal_values.append(aggregatee)
                    is_equal, real_error_level = SumCalculator.is_aggregation(aggregatee_decimal_values, aggregator_value, adjusted_error_bound,
                                                                              self.DIGIT_PLACES)
                    if is_equal:
                        break
            if is_equal and len(aggregatee_cells) >= 2:
                if not (aggregator_value == 0 and all([aee_cell.value in hard_empty_cell_values or aee_cell.value == 0 for aee_cell in aggregatee_cells])):
                    ar = AggregationRelation(copy(root), tuple([copy(aee_cell) for aee_cell in aggregatee_cells]), self.operator, Direction.BACKWARD)
                    aggregation_candidates.append((ar, real_error_level))
        return aggregation_candidates

    def mend_adjacent_aggregations(self, ar_cands_by_line, file_content, error_bound, axis):
        non_empty_ar_cands_by_line = {k: v for k, v in ar_cands_by_line.items() if bool(v[0])}
        if axis == 0:
            # row wise
            valid_row_indices = ar_cands_by_line.keys()
            ar_cands_by_column_index_direction = {}
            for ar_cands in non_empty_ar_cands_by_line.values():
                for ar_cand in ar_cands[0]:
                    aggregator = ar_cand[0].aggregator
                    column_index_direction = (aggregator.cell_index.column_index, ar_cand[0].direction)
                    if column_index_direction not in ar_cands_by_column_index_direction:
                        ar_cands_by_column_index_direction[column_index_direction] = []
                    ar_cands_by_column_index_direction[column_index_direction].append(ar_cand)
            for key, aggregations in ar_cands_by_column_index_direction.items():
                sorted_aggregations = sorted(aggregations, key=lambda x: len(x[0].aggregatees), reverse=True)
                for aggregation in sorted_aggregations:
                    aggregatees = aggregation[0].aggregatees
                    aggregatee_column_indices = [aggregatee.cell_index.column_index for aggregatee in aggregatees]
                    for row_index in valid_row_indices:
                        if row_index == aggregation[0].aggregator.cell_index.row_index:
                            continue
                        numberized_aggee_values = [self.to_number(file_content[row_index][ci], AggregationOperator.SUM.value) for ci in aggregatee_column_indices]
                        numberized_aggee_values = [value if value is not None else Decimal(0) for value in numberized_aggee_values]
                        if file_content[row_index][key[0]] in empty_cell_values:
                            continue
                        possible_aggor_value = self.to_number(file_content[row_index][key[0]], AggregationOperator.SUM.value)
                        if possible_aggor_value is None:
                            continue
                        if possible_aggor_value.is_nan() or any([e.is_nan() for e in numberized_aggee_values]):
                            continue
                        real_error_level = self.is_equal(possible_aggor_value, numberized_aggee_values, error_bound)
                        if real_error_level != math.inf:
                            mended_aggregation = AggregationRelation(Cell(CellIndex(row_index, key[0]), str(possible_aggor_value)),
                                                                     tuple([Cell(CellIndex(row_index, ci), file_content[row_index][ci]) for ci in
                                                                            aggregatee_column_indices]),
                                                                     AggregationOperator.SUM.value, aggregation[0].direction)
                            mended_collection = [elem[0] for elem in ar_cands_by_line[row_index][0]]
                            if mended_aggregation not in mended_collection:
                                mended_aggregation = AggregationRelation(Cell(CellIndex(row_index, key[0]), str(possible_aggor_value)),
                                                                         tuple([Cell(CellIndex(row_index, ci), file_content[row_index][ci]) for ci in
                                                                                aggregatee_column_indices]),
                                                                         AggregationOperator.SUM.value, aggregation[0].direction)
                                ar_cands_by_line[row_index][0].append((mended_aggregation, real_error_level))
        else:
            # column wise
            valid_column_indices = ar_cands_by_line.keys()
            ar_cands_by_row_index_direction = {}
            for ar_cands in non_empty_ar_cands_by_line.values():
                for ar_cand in ar_cands[0]:
                    aggregator = ar_cand[0].aggregator
                    row_index_direction = (aggregator.cell_index.row_index, ar_cand[0].direction)
                    if row_index_direction not in ar_cands_by_row_index_direction:
                        ar_cands_by_row_index_direction[row_index_direction] = []
                    ar_cands_by_row_index_direction[row_index_direction].append(ar_cand)
            for key, aggregations in ar_cands_by_row_index_direction.items():
                sorted_aggregations = sorted(aggregations, key=lambda x: len(x[0].aggregatees), reverse=True)
                for aggregation in sorted_aggregations:
                    aggregatees = aggregation[0].aggregatees
                    aggregatee_row_indices = [aggregatee.cell_index.row_index for aggregatee in aggregatees]
                    for column_index in valid_column_indices:
                        if column_index == aggregation[0].aggregator.cell_index.column_index:
                            continue
                        numberized_aggee_values = [self.to_number(file_content[ri][column_index], AggregationOperator.SUM.value)
                                                   for ri in aggregatee_row_indices]
                        numberized_aggee_values = [value if value is not None else Decimal(0) for value in numberized_aggee_values]
                        if file_content[key[0]][column_index] in empty_cell_values:
                            continue
                        possible_aggor_value = self.to_number(file_content[key[0]][column_index], AggregationOperator.SUM.value)
                        if possible_aggor_value is None:
                            continue
                        if possible_aggor_value.is_nan() or any([e.is_nan() for e in numberized_aggee_values]):
                            continue
                        real_error_level = self.is_equal(possible_aggor_value, numberized_aggee_values, error_bound)
                        if real_error_level != math.inf:
                            mended_aggregation = AggregationRelation(Cell(CellIndex(key[0], column_index), str(possible_aggor_value)),
                                                                     tuple([Cell(CellIndex(ri, column_index), file_content[ri][column_index]) for ri in
                                                                            aggregatee_row_indices]),
                                                                     AggregationOperator.SUM.value, aggregation[0].direction)
                            mended_collection = [elem[0] for elem in ar_cands_by_line[column_index][0]]
                            if mended_aggregation not in mended_collection:
                                mended_aggregation = AggregationRelation(Cell(CellIndex(key[0], column_index), str(possible_aggor_value)),
                                                                         tuple([Cell(CellIndex(ri, column_index), file_content[ri][column_index]) for ri in
                                                                                aggregatee_row_indices]),
                                                                         AggregationOperator.SUM.value, aggregation[0].direction)
                                ar_cands_by_line[column_index][0].append((mended_aggregation, real_error_level))
        pass

    def is_equal(self, aggregator_value, aggregatees, error_bound):
        adjusted_error_bound = error_bound
        expected_sum = sum(aggregatees)
        if aggregator_value == 0:
            error_level = abs(expected_sum - aggregator_value)
        else:
            error_level = abs((expected_sum - aggregator_value) / aggregator_value)
        return error_level if error_level <= adjusted_error_bound else math.inf


class SubtractionDetection(SlidingWindowAggregationDetection):

    def __init__(self, file, parameters) -> None:
        super().__init__(file, parameters)

        self.operator = AggregationOperator.SUBTRACT.value
        self.task_name = self.__class__.task_name

    def detect_proximity_aggregation_relations(self, forest: AggregationRelationForest, error_bound: float):
        roots = forest.get_roots()
        aggregation_candidates = []
        valid_roots = [root for root in roots if str2decimal(root.value, default=None) is not None]

        subtraction_result_cache = FIFOCache(maxsize=1024)

        for index, root in enumerate(valid_roots):
            aggregator_value = Decimal(root.value)
            if aggregator_value == 0:
                continue

            # forward
            forward_proximity = [valid_roots[i] for i in range(index + 1, index + 1 + self.WINDOW_SIZE) if i < len(valid_roots)]
            for permutation in itertools.permutations(forward_proximity, 2):
                first_element = permutation[0]
                second_element = permutation[1]

                cache_key = (first_element, second_element)
                if cache_key in subtraction_result_cache:
                    aggregatee_decimal_values = subtraction_result_cache.get(cache_key)
                else:
                    first_element_decimal_value = str2decimal(first_element.value, default=None)
                    second_element_decimal_value = str2decimal(second_element.value, default=None)
                    aggregatee_values = [first_element_decimal_value, second_element_decimal_value]
                    if aggregator_value == 0 and all([aee_cell in hard_empty_cell_values or aee_cell == 0 for aee_cell in aggregatee_values]):
                        continue

                    aggregatee_decimal_values = [first_element_decimal_value, second_element_decimal_value]
                    subtraction_result_cache.add(cache_key, aggregatee_decimal_values)

                is_aggregation, real_error_level = SubtractionCalculator.is_aggregation(aggregatee_decimal_values, aggregator_value, error_bound,
                                                                                        self.DIGIT_PLACES)

                if is_aggregation:
                    ar = self.transform2sum(copy(root), tuple(copy(permutation)))
                    aggregation_candidates.append((ar, real_error_level))

            # backward
            backward_proximity = [valid_roots[i] for i in range(index - self.WINDOW_SIZE, index) if i >= 0]
            for permutation in itertools.permutations(backward_proximity, 2):
                first_element = permutation[0]
                second_element = permutation[1]
                cache_key = (first_element, second_element)
                if cache_key in subtraction_result_cache:
                    aggregatee_decimal_values = subtraction_result_cache.get(cache_key)
                else:
                    first_element_decimal_value = str2decimal(first_element.value, default=None)
                    second_element_decimal_value = str2decimal(second_element.value, default=None)
                    aggregatee_values = [first_element_decimal_value, second_element_decimal_value]
                    if aggregator_value == 0 and all([aee_cell in hard_empty_cell_values or aee_cell == 0 for aee_cell in aggregatee_values]):
                        continue

                    aggregatee_decimal_values = [first_element_decimal_value, second_element_decimal_value]
                    subtraction_result_cache.add(cache_key, aggregatee_decimal_values)

                is_aggregation, real_error_level = SubtractionCalculator.is_aggregation(aggregatee_decimal_values, aggregator_value, error_bound,
                                                                                        self.DIGIT_PLACES)

                if is_aggregation:
                    ar = self.transform2sum(copy(root), tuple(copy(permutation)))
                    aggregation_candidates.append((ar, real_error_level))
        return aggregation_candidates

    @staticmethod
    def transform2sum(aggregator, aggregatee):
        sum_aggregator = Cell(aggregatee[0].cell_index, aggregatee[0].value)
        sum_aggregatee = tuple([
            Cell(aggregator.cell_index, aggregator.value),
            Cell(aggregatee[1].cell_index, aggregatee[1].value)
        ])
        return AggregationRelation(sum_aggregator, sum_aggregatee, AggregationOperator.SUM.value, Direction.DIRECTIONLESS)

    def mend_adjacent_aggregations(self, ar_cands_by_line, file_content, error_bound, axis):
        pass

    def is_equal(self, aggregator_value, aggregatees, error_bound):
        pass


class AverageDetection(AdjacentListAggregationDetection):

    def __init__(self, file, parameters) -> None:
        super().__init__(file, parameters)

        self.operator = AggregationOperator.AVERAGE.value
        self.task_name = self.__class__.task_name

    def detect_proximity_aggregation_relations(self, forest: AggregationRelationForest, error_bound: float):
        roots = forest.get_roots()
        aggregation_candidates = []
        for index, root in enumerate(roots):
            # if index == 245:
            #     print('STOP')
            try:
                aggregator_value = Decimal(root.value)
            except decimal.InvalidOperation:
                continue
            if aggregator_value.is_nan():
                continue
            if aggregator_value == 0:
                continue

            # forward
            aggregatee_cells = []
            aggregatee_decimal_values = []
            is_equal = False
            real_error_level = math.inf
            for i in range(index + 1, len(roots)):
                try:
                    aggregatee = Decimal(roots[i].value)
                except decimal.InvalidOperation:
                    continue
                else:
                    aggregatee_cells.append(roots[i])
                    aggregatee_decimal_values.append(aggregatee)
                    is_equal, real_error_level = AverageCalculator.is_aggregation(aggregatee_decimal_values, aggregator_value, error_bound, self.DIGIT_PLACES)
                    if is_equal:
                        break
            if is_equal and len(aggregatee_cells) >= 2:
                if not (aggregator_value == 0 and all([aee_cell.value in hard_empty_cell_values or aee_cell.value == 0 for aee_cell in aggregatee_cells])):
                    ar = AggregationRelation(copy(root), tuple([copy(aee_cell) for aee_cell in aggregatee_cells]), self.operator, Direction.FORWARD)
                    aggregation_candidates.append((ar, real_error_level))

            # backward
            aggregatee_cells = []
            aggregatee_decimal_values = []
            is_equal = False
            real_error_level = math.inf
            for i in reversed(range(index)):
                try:
                    aggregatee = Decimal(roots[i].value)
                except decimal.InvalidOperation:
                    continue
                else:
                    aggregatee_cells.append(roots[i])
                    aggregatee_decimal_values.append(aggregatee)
                    is_equal, real_error_level = AverageCalculator.is_aggregation(aggregatee_decimal_values, aggregator_value, error_bound, self.DIGIT_PLACES)
                    if is_equal:
                        break
            if is_equal and len(aggregatee_cells) >= 2:
                if not (aggregator_value == 0 and all([aee_cell.value in hard_empty_cell_values or aee_cell.value == 0 for aee_cell in aggregatee_cells])):
                    ar = AggregationRelation(copy(root), tuple([copy(aee_cell) for aee_cell in aggregatee_cells]), self.operator, Direction.BACKWARD)
                    aggregation_candidates.append((ar, real_error_level))

        return aggregation_candidates

    def mend_adjacent_aggregations(self, ar_cands_by_line, file_content, error_bound, axis):
        non_empty_ar_cands_by_line = {k: v for k, v in ar_cands_by_line.items() if bool(v[0])}
        if axis == 0:
            # row wise
            valid_row_indices = ar_cands_by_line.keys()
            ar_cands_by_column_index_direction = {}
            for ar_cands in non_empty_ar_cands_by_line.values():
                for ar_cand in ar_cands[0]:
                    aggregator = ar_cand[0].aggregator
                    column_index_direction = (aggregator.cell_index.column_index, ar_cand[0].direction)
                    if column_index_direction not in ar_cands_by_column_index_direction:
                        ar_cands_by_column_index_direction[column_index_direction] = []
                    ar_cands_by_column_index_direction[column_index_direction].append(ar_cand)
            for key, aggregations in ar_cands_by_column_index_direction.items():
                sorted_aggregations = sorted(aggregations, key=lambda x: len(x[0].aggregatees), reverse=True)
                for aggregation in sorted_aggregations:
                    aggregatees = aggregation[0].aggregatees
                    aggregatee_column_indices = [aggregatee.cell_index.column_index for aggregatee in aggregatees]
                    for row_index in valid_row_indices:
                        if row_index == aggregation[0].aggregator.cell_index.row_index:
                            continue
                        possible_aggee_values = [file_content[row_index][ci] for ci in aggregatee_column_indices]
                        numberized_aggee_values = [self.to_number(elem, AggregationOperator.AVERAGE.value) for elem in possible_aggee_values]
                        numberized_aggee_values = [value for value in numberized_aggee_values if value is not None]
                        if not bool(numberized_aggee_values) or len(numberized_aggee_values) < 2 or len(set([elem for elem in numberized_aggee_values])) == 1:
                            continue

                        possible_aggor_value = self.to_number(file_content[row_index][key[0]], AggregationOperator.AVERAGE.value)
                        if not possible_aggor_value:
                            continue
                        if possible_aggor_value.is_nan() or any([e.is_nan() for e in numberized_aggee_values]):
                            continue
                        real_error_level = self.is_equal(possible_aggor_value, numberized_aggee_values, error_bound)
                        if real_error_level != math.inf:
                            mended_aggregation = AggregationRelation(Cell(CellIndex(row_index, key[0]), str(possible_aggor_value)),
                                                                     tuple([Cell(CellIndex(row_index, ci), file_content[row_index][ci]) for ci in
                                                                            aggregatee_column_indices]),
                                                                     AggregationOperator.AVERAGE.value, aggregation[0].direction)
                            mended_collection = [elem[0] for elem in ar_cands_by_line[row_index][0]]
                            if mended_aggregation not in mended_collection:
                                mended_aggregation = AggregationRelation(Cell(CellIndex(row_index, key[0]), str(possible_aggor_value)),
                                                                         tuple([Cell(CellIndex(row_index, ci), file_content[row_index][ci]) for ci in
                                                                                aggregatee_column_indices]),
                                                                         AggregationOperator.AVERAGE.value, aggregation[0].direction)
                                ar_cands_by_line[row_index][0].append((mended_aggregation, real_error_level))
                        pass
        else:
            # column wise
            valid_column_indices = ar_cands_by_line.keys()
            ar_cands_by_row_index_direction = {}
            for ar_cands in non_empty_ar_cands_by_line.values():
                for ar_cand in ar_cands[0]:
                    aggregator = ar_cand[0].aggregator
                    row_index_direction = (aggregator.cell_index.row_index, ar_cand[0].direction)
                    if row_index_direction not in ar_cands_by_row_index_direction:
                        ar_cands_by_row_index_direction[row_index_direction] = []
                    ar_cands_by_row_index_direction[row_index_direction].append(ar_cand)
            for key, aggregations in ar_cands_by_row_index_direction.items():
                sorted_aggregations = sorted(aggregations, key=lambda x: len(x[0].aggregatees), reverse=True)
                for aggregation in sorted_aggregations:
                    aggregatees = aggregation[0].aggregatees
                    aggregatee_row_indices = [aggregatee.cell_index.row_index for aggregatee in aggregatees]
                    for column_index in valid_column_indices:
                        if column_index == aggregation[0].aggregator.cell_index.column_index:
                            continue
                        possible_aggee_values = [file_content[ri][column_index] for ri in aggregatee_row_indices]
                        numberized_aggee_values = [self.to_number(elem, AggregationOperator.AVERAGE.value) for elem in possible_aggee_values]
                        numberized_aggee_values = [value for value in numberized_aggee_values if value is not None]
                        if not bool(numberized_aggee_values) or len(numberized_aggee_values) < 2 or len(set([elem for elem in numberized_aggee_values])) == 1:
                            continue

                        possible_aggor_value = self.to_number(file_content[key[0]][column_index], AggregationOperator.AVERAGE.value)
                        if not possible_aggor_value:
                            continue
                        if possible_aggor_value.is_nan() or any([e.is_nan() for e in numberized_aggee_values]):
                            continue

                        real_error_level = self.is_equal(possible_aggor_value, numberized_aggee_values, error_bound)
                        if real_error_level != math.inf:
                            mended_aggregation = AggregationRelation(Cell(CellIndex(key[0], column_index), str(possible_aggor_value)),
                                                                     tuple([Cell(CellIndex(ri, column_index), file_content[ri][column_index]) for ri in
                                                                            aggregatee_row_indices]),
                                                                     AggregationOperator.AVERAGE.value, aggregation[0].direction)
                            mended_collection = [elem[0] for elem in ar_cands_by_line[column_index][0]]
                            if mended_aggregation not in mended_collection:
                                mended_aggregation = AggregationRelation(Cell(CellIndex(key[0], column_index), str(possible_aggor_value)),
                                                                         tuple([Cell(CellIndex(ri, column_index), file_content[ri][column_index]) for ri in
                                                                                aggregatee_row_indices]),
                                                                         AggregationOperator.AVERAGE.value, aggregation[0].direction)
                                ar_cands_by_line[column_index][0].append((mended_aggregation, real_error_level))
                        pass
                    pass
        pass

    def is_equal(self, aggregator_value, aggregatees, error_bound):
        expected_sum = sum(aggregatees)
        expected_average = round(expected_sum / len(aggregatees), self.DIGIT_PLACES)
        nan_count = len([v for v in aggregatees if v.is_nan()])

        aggregatee_length_excl_nan = Decimal(len(aggregatees) - nan_count)
        expected_excl_nan = round(expected_sum / aggregatee_length_excl_nan, self.DIGIT_PLACES) if aggregatee_length_excl_nan > 0 else Decimal(math.inf)

        if aggregator_value == 0:
            error_level = min(abs(expected_average - aggregator_value), abs(expected_excl_nan - aggregator_value))
        else:
            error_level = min(abs((expected_average - aggregator_value) / aggregator_value), abs((expected_excl_nan - aggregator_value) / aggregator_value))
        return error_level if error_level <= error_bound else math.inf


class DivisionDetection(SlidingWindowAggregationDetection):

    def __init__(self, file, parameters) -> None:
        super().__init__(file, parameters)

        self.operator = AggregationOperator.DIVISION.value
        self.task_name = self.__class__.task_name

    def detect_proximity_aggregation_relations(self, forest: AggregationRelationForest, error_bound: float):
        roots = forest.get_roots()
        aggregation_candidates = []
        valid_roots = [root for root in roots if str2decimal(root.value, default=None) is not None]

        division_result_cache = FIFOCache(maxsize=1024)

        for index, root in enumerate(valid_roots):
            aggregator_value = Decimal(root.value)
            # if root.cell_index == CellIndex(7, 5):
            #     print('STOP')
            if aggregator_value == 0:
                continue

            # forward
            forward_proximity = [valid_roots[i] for i in range(index + 1, index + 1 + self.WINDOW_SIZE) if i < len(valid_roots)]
            for permutation in itertools.permutations(forward_proximity, 2):
                first_element = permutation[0]
                second_element = permutation[1]
                cache_key = (first_element, second_element)
                if cache_key in division_result_cache:
                    aggregatee_decimal_values = division_result_cache.get(cache_key)
                else:
                    first_element_decimal_value = str2decimal(first_element.value, default=None)
                    second_element_decimal_value = str2decimal(second_element.value, default=None)
                    if first_element_decimal_value == 0 or second_element_decimal_value == 0:
                        continue

                    aggregatee_decimal_values = [first_element_decimal_value, second_element_decimal_value]
                    division_result_cache.add(cache_key, aggregatee_decimal_values)

                is_aggregation, real_error_level = DivisionCalculator.is_aggregation(aggregatee_decimal_values, aggregator_value, error_bound,
                                                                                     self.DIGIT_PLACES)

                if is_aggregation:
                    ar = AggregationRelation(copy(root), tuple(copy(permutation)), self.operator, Direction.DIRECTIONLESS)
                    aggregation_candidates.append((ar, real_error_level))

            # backward
            backward_proximity = [valid_roots[i] for i in range(index - self.WINDOW_SIZE, index) if i >= 0]
            for permutation in itertools.permutations(backward_proximity, 2):
                first_element = permutation[0]
                second_element = permutation[1]
                cache_key = (first_element, second_element)
                if cache_key in division_result_cache:
                    aggregatee_decimal_values = division_result_cache.get(cache_key)
                else:
                    first_element_decimal_value = str2decimal(first_element.value, default=None)
                    second_element_decimal_value = str2decimal(second_element.value, default=None)
                    if first_element_decimal_value == 0 or second_element_decimal_value == 0:
                        continue

                    aggregatee_decimal_values = [first_element_decimal_value, second_element_decimal_value]
                    division_result_cache.add(cache_key, aggregatee_decimal_values)

                is_aggregation, real_error_level = DivisionCalculator.is_aggregation(aggregatee_decimal_values, aggregator_value, error_bound,
                                                                                     self.DIGIT_PLACES)

                if is_aggregation:
                    ar = AggregationRelation(copy(root), tuple(copy(permutation)), self.operator, Direction.DIRECTIONLESS)
                    aggregation_candidates.append((ar, real_error_level))

        return aggregation_candidates

    def mend_adjacent_aggregations(self, ar_cands_by_line, file_content, error_bound, axis):
        pass

    def is_equal(self, aggregator_value, aggregatees, error_bound):
        expected_value = round(aggregatees[0] / aggregatees[1], ndigits=self.DIGIT_PLACES)
        if aggregator_value == 0:
            error_level = abs(expected_value - aggregator_value)
        else:
            error_level = abs((expected_value - aggregator_value) / aggregator_value)
        return error_level if error_level <= error_bound else math.inf


class RelativeChangeDetection(SlidingWindowAggregationDetection):

    def __init__(self, file, parameters) -> None:
        super().__init__(file, parameters)

        self.operator = AggregationOperator.RELATIVE_CHANGE.value
        self.task_name = self.__class__.task_name

    def detect_proximity_aggregation_relations(self, forest: AggregationRelationForest, error_bound: float):
        roots = forest.get_roots()
        aggregation_candidates = []
        valid_roots = [root for root in roots if str2decimal(root.value, default=None) is not None]

        relative_change_result_cache = FIFOCache(maxsize=1024)

        for index, root in enumerate(valid_roots):
            aggregator_value = Decimal(root.value)
            if aggregator_value == 0:
                continue
            abs_aggregator_value = abs(aggregator_value)
            if abs_aggregator_value > 100:
                continue

            # forward
            forward_proximity = [valid_roots[i] for i in range(index + 1, index + 1 + self.WINDOW_SIZE) if i < len(valid_roots)]
            for permutation in itertools.permutations(forward_proximity, 2):
                first_element = permutation[0]
                second_element = permutation[1]
                cache_key = (first_element, second_element)
                if cache_key in relative_change_result_cache:
                    aggregatee_decimal_values = relative_change_result_cache.get(cache_key)
                else:
                    first_element_decimal_value = str2decimal(first_element.value, default=None)
                    second_element_decimal_value = str2decimal(second_element.value, default=None)
                    if first_element_decimal_value == 0 or second_element_decimal_value == 0:
                        continue

                    aggregatee_decimal_values = [first_element_decimal_value, second_element_decimal_value]
                    relative_change_result_cache.add(cache_key, aggregatee_decimal_values)

                is_aggregation, real_error_level = RelativeChangeCalculator.is_aggregation(aggregatee_decimal_values, aggregator_value, error_bound,
                                                                                           self.DIGIT_PLACES)

                if is_aggregation:
                    ar = AggregationRelation(copy(root), tuple(copy(permutation)), self.operator, Direction.DIRECTIONLESS)
                    aggregation_candidates.append((ar, real_error_level))

            # backward
            backward_proximity = [valid_roots[i] for i in range(index - self.WINDOW_SIZE, index) if i >= 0]
            for permutation in itertools.permutations(backward_proximity, 2):
                first_element = permutation[0]
                second_element = permutation[1]
                cache_key = (first_element, second_element)
                if cache_key in relative_change_result_cache:
                    aggregatee_decimal_values = relative_change_result_cache.get(cache_key)
                else:
                    first_element_decimal_value = str2decimal(first_element.value, default=None)
                    second_element_decimal_value = str2decimal(second_element.value, default=None)
                    if first_element_decimal_value == 0 or second_element_decimal_value == 0:
                        continue

                    aggregatee_decimal_values = [first_element_decimal_value, second_element_decimal_value]
                    relative_change_result_cache.add(cache_key, aggregatee_decimal_values)

                is_aggregation, real_error_level = RelativeChangeCalculator.is_aggregation(aggregatee_decimal_values, aggregator_value, error_bound,
                                                                                           self.DIGIT_PLACES)

                if is_aggregation:
                    ar = AggregationRelation(copy(root), tuple(copy(permutation)), self.operator, Direction.DIRECTIONLESS)
                    aggregation_candidates.append((ar, real_error_level))

        return aggregation_candidates

    def mend_adjacent_aggregations(self, ar_cands_by_line, file_content, error_bound, axis):
        pass

    def is_equal(self, aggregator_value, aggregatees, error_bound):
        expected_value = abs((aggregatees[1] - aggregatees[0]) / aggregatees[0])
        if aggregator_value == 0:
            error_level = abs(expected_value - aggregator_value)
        else:
            error_level = abs((expected_value - aggregator_value) / aggregator_value)
        return error_level if error_level <= error_bound else math.inf
