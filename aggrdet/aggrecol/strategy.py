# Created by lan at 2021/4/21
import itertools
import time
from abc import ABC
from copy import deepcopy

import numpy as np

from aggrecol.phase import IndividualAggregationDetection
from aggrecol.pruning import prune_conflict_ar_cands
from elements import Cell, CellIndex
from helpers import AggregationDirection, AggregationOperator
from tree import AggregationRelationForest


class AdjacentListAggregationDetection(IndividualAggregationDetection, ABC):

    def detect_row_wise_aggregations(self):
        start_time = time.time()
        _file = deepcopy(self.file)
        _file.aggregation_detection_result = {}

        error_level = self.error_level_setting[self.operator]
        collected_results = []
        for number_format, formatted_file_values in _file.valid_number_formats.items():

            # Todo: just for fair timeout comparison
            if number_format != _file.number_format:
                continue

            numeric_line_indices = _file.numeric_line_indices.get(number_format, None)

            file_cells = np.full_like(formatted_file_values, fill_value=formatted_file_values, dtype=object)
            for index, value in np.ndenumerate(formatted_file_values):
                file_cells[index] = Cell(CellIndex(index[0], index[1]), value)

            forests_by_rows = [AggregationRelationForest(row_cells) for row_cells in file_cells]
            forest_by_row_index = {}
            for index, forest in enumerate(forests_by_rows):
                forest_by_row_index[index] = forest
            collected_results_by_row = {}
            while True:
                ar_cands_by_row = {index: (self.detect_proximity_aggregation_relations(forest, error_level), forest) for index, forest in
                                   forest_by_row_index.items()}

                self.mend_adjacent_aggregations(ar_cands_by_row, formatted_file_values, error_level, axis=0)

                # get all non empty ar_cands
                ar_cands_by_row = list(filter(lambda x: bool(x[0]), ar_cands_by_row.values()))
                if not ar_cands_by_row:
                    break

                forest_indexed_by_ar_cand = {}
                for ar_cands, forest in ar_cands_by_row:
                    for ar_cand in ar_cands:
                        forest_indexed_by_ar_cand[ar_cand[0]] = forest

                ar_cands_by_row, forests_by_rows = list(zip(*ar_cands_by_row))

                ar_cands_by_column_index = prune_conflict_ar_cands(ar_cands_by_row, numeric_line_indices, self.coverage, axis=0)

                ar_cands_by_column_index = {k: v for k, v in ar_cands_by_column_index.items() if
                                            len(numeric_line_indices[1][str(k[0])]) > 0 and
                                            len(v) / len(numeric_line_indices[1][str(k[0])]) >= self.coverage}

                if not bool(ar_cands_by_column_index):
                    break

                for _, ar_cands in ar_cands_by_column_index.items():
                    for i in range(len(ar_cands)):
                        ar_cands[i] = (ar_cands[i], forest_indexed_by_ar_cand[ar_cands[i]])

                for ar_column_indices, ar_cands_w_forest in ar_cands_by_column_index.items():
                    [forest.consume_relation(ar_cand) for ar_cand, forest in ar_cands_w_forest]

                for signature in ar_cands_by_column_index.keys():
                    [forest.remove_consumed_signature(signature, axis=0) for forest in forest_by_row_index.values()]

                if self.operator == AggregationOperator.AVERAGE.value:
                    break

            for _, forest in forest_by_row_index.items():
                results_dict = forest.results_to_str(self.operator, AggregationDirection.ROW_WISE.value)
                collected_results_by_row[forest] = results_dict

            collected_results = list(itertools.chain(*[results_dict for _, results_dict in collected_results_by_row.items()]))

            _file.aggregation_detection_result[number_format] = collected_results
        end_time = time.time()
        exec_time = end_time - start_time
        _file.exec_time['RowWiseDetection'] = exec_time
        return _file
        # return collected_results, exec_time

    def detect_column_wise_aggregations(self):
        start_time = time.time()
        _file = deepcopy(self.file)
        _file.aggregation_detection_result = {}

        error_level = self.error_level_setting[self.operator]
        for number_format, formatted_file_values in _file.valid_number_formats.items():
            # Todo: just for fair timeout comparison
            if number_format != _file.number_format:
                continue

            numeric_line_indices = _file.numeric_line_indices.get(number_format, None)

            file_cells = np.full_like(formatted_file_values, fill_value=formatted_file_values, dtype=object)
            for index, value in np.ndenumerate(formatted_file_values):
                file_cells[index] = Cell(CellIndex(index[0], index[1]), value)

            forests_by_columns = [AggregationRelationForest(file_cells[:, i]) for i in range(file_cells.shape[1])]
            forest_by_column_index = {}
            for index, forest in enumerate(forests_by_columns):
                forest_by_column_index[index] = forest
            collected_results_by_column = {}
            while True:
                ar_cands_by_column = {index: (self.detect_proximity_aggregation_relations(forest, error_level), forest) for index, forest in
                                      forest_by_column_index.items()}

                self.mend_adjacent_aggregations(ar_cands_by_column, formatted_file_values, error_level, axis=1)

                # get all non empty ar_cands
                ar_cands_by_column = list(filter(lambda x: bool(x[0]), ar_cands_by_column.values()))
                if not ar_cands_by_column:
                    break

                forest_indexed_by_ar_cand = {}
                for ar_cands, forest in ar_cands_by_column:
                    for ar_cand in ar_cands:
                        forest_indexed_by_ar_cand[ar_cand[0]] = forest

                ar_cands_by_column, forests_by_columns = list(zip(*ar_cands_by_column))

                ar_cands_by_row_index = prune_conflict_ar_cands(ar_cands_by_column, numeric_line_indices, self.coverage, axis=1)

                ar_cands_by_row_index = {k: v for k, v in ar_cands_by_row_index.items() if
                                         len(numeric_line_indices[0][str(k[0])]) > 0 and
                                         len(v) / len(numeric_line_indices[0][str(k[0])]) >= self.coverage}

                if not bool(ar_cands_by_row_index):
                    break

                for _, ar_cands in ar_cands_by_row_index.items():
                    for i in range(len(ar_cands)):
                        ar_cands[i] = (ar_cands[i], forest_indexed_by_ar_cand[ar_cands[i]])

                # extended_ar_cands_w_forest = []
                for ar_row_indices, ar_cands_w_forest in ar_cands_by_row_index.items():
                    [forest.consume_relation(ar_cand) for ar_cand, forest in ar_cands_w_forest]

                for signature in ar_cands_by_row_index.keys():
                    # [forest.remove_consumed_aggregator(ar_cand) for ar_cand, forest in ar_cands_w_forest]
                    [forest.remove_consumed_signature(signature, axis=1) for forest in forest_by_column_index.values()]

                if self.operator == AggregationOperator.AVERAGE.value:
                    break

            for _, forest in forest_by_column_index.items():
                results_dict = forest.results_to_str(self.operator, AggregationDirection.COLUMN_WISE.value)
                collected_results_by_column[forest] = results_dict

            collected_results = list(itertools.chain(*[results_dict for _, results_dict in collected_results_by_column.items()]))

            _file.aggregation_detection_result[number_format] = collected_results
        end_time = time.time()
        exec_time = end_time - start_time
        _file.exec_time['ColumnWiseDetection'] = exec_time
        return _file


class SlidingWindowAggregationDetection(IndividualAggregationDetection, ABC):
    WINDOW_SIZE = 10

    def detect_row_wise_aggregations(self):
        start_time = time.time()
        _file = deepcopy(self.file)
        _file.aggregation_detection_result = {}
        error_level = self.error_level_setting[self.operator]
        for number_format, formatted_file_values in _file.valid_number_formats.items():

            # Todo: just for fair timeout comparison
            if number_format != _file.number_format:
                continue

            numeric_line_indices = _file.numeric_line_indices.get(number_format, None)

            file_cells = np.full_like(formatted_file_values, fill_value=formatted_file_values, dtype=object)
            for index, value in np.ndenumerate(formatted_file_values):
                file_cells[index] = Cell(CellIndex(index[0], index[1]), value)

            forests_by_rows = [AggregationRelationForest(row_cells) for row_cells in file_cells]
            forest_by_row_index = {}
            for index, forest in enumerate(forests_by_rows):
                forest_by_row_index[index] = forest
            ar_cands_by_row = [(self.detect_proximity_aggregation_relations(forest, error_level), forest) for forest in
                               forests_by_rows]
            # get all non empty ar_cands
            ar_cands_by_row = list(filter(lambda x: bool(x[0]), ar_cands_by_row))
            if not ar_cands_by_row:
                collected_results = []
            else:
                self.mend_adjacent_aggregations(ar_cands_by_row, formatted_file_values, error_level, axis=0)

                forest_indexed_by_ar_cand = {}
                for ar_cands, forest in ar_cands_by_row:
                    for ar_cand in ar_cands:
                        forest_indexed_by_ar_cand[ar_cand[0]] = forest

                ar_cands_by_row, forests_by_rows = list(zip(*ar_cands_by_row))

                ar_cands_by_column_index = prune_conflict_ar_cands(ar_cands_by_row, numeric_line_indices, self.coverage, axis=0)

                ar_cands_by_column_index = {k: v for k, v in ar_cands_by_column_index.items() if
                                            len(numeric_line_indices[1][str(k[0])]) > 0 and
                                            len(v) / len(numeric_line_indices[1][str(k[0])]) >= self.coverage}

                collected_results = []
                if bool(ar_cands_by_column_index):
                    for _, ar_cands in ar_cands_by_column_index.items():
                        for ar_cand in ar_cands:
                            aggregator = str(ar_cand.aggregator.cell_index)
                            aggregatees = [str(aggregatee.cell_index) for aggregatee in ar_cand.aggregatees]
                            operator = ar_cand.operator
                            collected_results.append((aggregator, aggregatees, operator, AggregationDirection.ROW_WISE.value))

            _file.aggregation_detection_result[number_format] = collected_results
        end_time = time.time()
        exec_time = end_time - start_time
        _file.exec_time['RowWiseDetection'] = exec_time
        return _file

    def detect_column_wise_aggregations(self):
        start_time = time.time()
        _file = deepcopy(self.file)
        _file.aggregation_detection_result = {}

        error_level = self.error_level_setting[self.operator]

        for number_format, formatted_file_values in _file.valid_number_formats.items():
            # Todo: just for fair timeout comparison
            if number_format != _file.number_format:
                continue

            numeric_line_indices = _file.numeric_line_indices[number_format]

            file_cells = np.full_like(formatted_file_values, fill_value=formatted_file_values, dtype=object)
            for index, value in np.ndenumerate(formatted_file_values):
                file_cells[index] = Cell(CellIndex(index[0], index[1]), value)

            forests_by_columns = [AggregationRelationForest(file_cells[:, i]) for i in range(file_cells.shape[1])]
            forest_by_column_index = {}
            for index, forest in enumerate(forests_by_columns):
                forest_by_column_index[index] = forest

            ar_cands_by_column = [(self.detect_proximity_aggregation_relations(forest, error_level), forest) for forest in
                                  forests_by_columns]
            # get all non empty ar_cands
            ar_cands_by_column = list(filter(lambda x: bool(x[0]), ar_cands_by_column))
            if not ar_cands_by_column:
                collected_results = []
            else:
                self.mend_adjacent_aggregations(ar_cands_by_column, formatted_file_values, error_level, axis=1)

                forest_indexed_by_ar_cand = {}
                for ar_cands, forest in ar_cands_by_column:
                    for ar_cand in ar_cands:
                        forest_indexed_by_ar_cand[ar_cand[0]] = forest

                ar_cands_by_column, forests_by_columns = list(zip(*ar_cands_by_column))

                ar_cands_by_row_index = prune_conflict_ar_cands(ar_cands_by_column, numeric_line_indices, self.coverage, axis=1)

                ar_cands_by_row_index = {k: v for k, v in ar_cands_by_row_index.items() if
                                         len(numeric_line_indices[0][str(k[0])]) > 0 and
                                         len(v) / len(numeric_line_indices[0][str(k[0])]) >= self.coverage}

                collected_results = []
                if bool(ar_cands_by_row_index):
                    for _, ar_cands in ar_cands_by_row_index.items():
                        for ar_cand in ar_cands:
                            aggregator = str(ar_cand.aggregator.cell_index)
                            aggregatees = [str(aggregatee.cell_index) for aggregatee in ar_cand.aggregatees]
                            operator = ar_cand.operator
                            collected_results.append((aggregator, aggregatees, operator, AggregationDirection.COLUMN_WISE.value))

            _file.aggregation_detection_result[number_format] = collected_results
        end_time = time.time()
        exec_time = end_time - start_time
        _file.exec_time['ColumnWiseDetection'] = exec_time
        return _file