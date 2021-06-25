# Created by lan at 2021/5/4
import ast
import itertools
import math
from abc import ABC, abstractmethod
from copy import copy

import numpy as np

from aggrecol.pruning import filter_conflict_ar_cands
from aggrecol.individual import SumDetection, AverageDetection
from data import normalize_file
from elements import File
from helpers import AggregationDirection, AggregationOperator
from number import str2decimal


def filter(detected_result_across_solutions):
    non_conflict_ar_cands_index = copy(detected_result_across_solutions)
    for ar_index in detected_result_across_solutions.keys():
        if ar_index not in non_conflict_ar_cands_index:
            continue
        non_conflict_ar_cands_index[ar_index] = detected_result_across_solutions[ar_index]
        non_conflicts = filter_conflict_ar_cands(ar_index, non_conflict_ar_cands_index.keys())
        non_conflict_ar_cands_index = {}
        for non_conflict_indx in non_conflicts:
            non_conflict_ar_cands_index[non_conflict_indx] = detected_result_across_solutions[non_conflict_indx]
    return non_conflict_ar_cands_index


class SupplementalAggregationDetection(ABC):

    def __init__(self) -> None:
        super().__init__()
        self.original_file = None
        self.operator = None

    @abstractmethod
    def _ignore_existing(self, column_reduced_files, row_reduced_files):
        """
        Detecting aggregations on the file where the detected ones are omitted.
        :return:
        """
        pass

    def _construct_files(self, detected_aggregations):
        # row wise
        row_wise_aggregations = [detected_aggregation for detected_aggregation in detected_aggregations if
                                 detected_aggregation[3] == AggregationDirection.ROW_WISE.value]
        sum_aggregator_indices = list(set([ast.literal_eval(aggregation[0])[1] for aggregation in row_wise_aggregations if
                                           aggregation[2] == AggregationOperator.SUM.value]))
        average_aggregator_indices = [ast.literal_eval(aggregation[0])[1] for aggregation in row_wise_aggregations if
                                      aggregation[2] == AggregationOperator.AVERAGE.value]
        relative_aggregator_indices = [ast.literal_eval(aggregation[0])[1] for aggregation in row_wise_aggregations if
                                       aggregation[2] == AggregationOperator.RELATIVE_CHANGE.value]
        division_aggregatee_indices = list(
            set(itertools.chain(*[(ast.literal_eval(aggregation[0])[1], ast.literal_eval(aggregation[1][1])[1]) for aggregation in row_wise_aggregations
                                  if aggregation[2] == AggregationOperator.DIVISION.value])))
        column_indices = [i for i in range(self.original_file.num_cols) if
                          i not in sum_aggregator_indices and
                          i not in average_aggregator_indices and
                          i not in relative_aggregator_indices and
                          i not in division_aggregatee_indices]
        column_reduced_constructed_files = []
        single_added_column_indices = division_aggregatee_indices + [None]
        reduced_column_indices_wo_division_candidates = [sorted(column_indices)]
        if bool(sum_aggregator_indices):
            reduced_column_indices_wo_division_candidates.append(sorted(column_indices + sum_aggregator_indices))
        for candidate in reduced_column_indices_wo_division_candidates:
            reduced_column_indices_candidate = copy(candidate)
            for i in single_added_column_indices:
                if i is not None:
                    reduced_column_indices_candidate = reduced_column_indices_candidate + [i]
                reduced_column_indices_candidate = sorted(list(set(reduced_column_indices_candidate)))
                constructed_file_dict = {'file_name': self.original_file.file_name, 'table_id': self.original_file.sheet_name,
                                         'num_rows': self.original_file.num_rows, 'num_cols': len(reduced_column_indices_candidate)}
                original_file_narray = np.array(self.original_file.file_values)
                constructed_file_dict['table_array'] = original_file_narray[:, reduced_column_indices_candidate].tolist()
                constructed_file_dict['parameters'] = self.original_file.parameters
                constructed_file_dict['number_format'] = self.original_file.number_format
                transformed_values_by_number_format = {}
                numeric_line_indices = {}
                file_cell_data_types = {}
                nf = self.original_file.number_format
                transformed_values_by_number_format[nf], numeric_line_indices[nf], file_cell_data_types[nf] = \
                    normalize_file(constructed_file_dict['table_array'], nf)
                constructed_file_dict['valid_number_formats'] = transformed_values_by_number_format
                constructed_file_dict['numeric_line_indices'] = numeric_line_indices
                constructed_file_dict['reduced_column_indices'] = reduced_column_indices_candidate
                constructed_file_dict['exec_time'] = {}
                column_reduced_constructed_files.append(constructed_file_dict)

        # column wise
        column_wise_aggregations = [detected_aggregation for detected_aggregation in detected_aggregations if
                                    detected_aggregation[3] == AggregationDirection.COLUMN_WISE.value]
        sum_aggregator_indices = list(set([ast.literal_eval(aggregation[0])[0] for aggregation in column_wise_aggregations if
                                           aggregation[2] == AggregationOperator.SUM.value]))
        average_aggregator_indices = [ast.literal_eval(aggregation[0])[0] for aggregation in column_wise_aggregations if
                                      aggregation[2] == AggregationOperator.AVERAGE.value]
        relative_aggregator_indices = [ast.literal_eval(aggregation[0])[0] for aggregation in column_wise_aggregations if
                                       aggregation[2] == AggregationOperator.RELATIVE_CHANGE.value]
        division_aggregatee_indices = list(
            set(itertools.chain(*[(ast.literal_eval(aggregation[0])[0], ast.literal_eval(aggregation[1][1])[0]) for aggregation in column_wise_aggregations
                                  if aggregation[2] == AggregationOperator.DIVISION.value])))
        row_indices = [i for i in range(self.original_file.num_rows) if
                       i not in sum_aggregator_indices and
                       i not in average_aggregator_indices and
                       i not in relative_aggregator_indices and
                       i not in division_aggregatee_indices]
        row_reduced_constructed_files = []
        single_added_row_indices = division_aggregatee_indices + [None]
        reduced_row_indices_wo_division_candidates = [sorted(row_indices)]
        if bool(sum_aggregator_indices):
            reduced_row_indices_wo_division_candidates.append(sorted(row_indices + sum_aggregator_indices))
        for candidate in reduced_row_indices_wo_division_candidates:
            reduced_row_indices_candidate = copy(candidate)
            for i in single_added_row_indices:
                if i is not None:
                    reduced_row_indices_candidate = reduced_row_indices_candidate + [i]
                reduced_row_indices_candidate = sorted(list(set(reduced_row_indices_candidate)))
                constructed_file_dict = {'file_name': self.original_file.file_name, 'table_id': self.original_file.sheet_name,
                                         'num_rows': len(reduced_row_indices_candidate), 'num_cols': self.original_file.num_cols,
                                         'table_array': [self.original_file.file_values[row_index] for row_index in range(self.original_file.num_rows)
                                                         if row_index in reduced_row_indices_candidate], 'parameters': self.original_file.parameters,
                                         'number_format': self.original_file.number_format}
                transformed_values_by_number_format = {}
                numeric_line_indices = {}
                file_cell_data_types = {}
                nf = self.original_file.number_format
                transformed_values_by_number_format[nf], numeric_line_indices[nf], file_cell_data_types[nf] = \
                    normalize_file(constructed_file_dict['table_array'], nf)
                constructed_file_dict['valid_number_formats'] = transformed_values_by_number_format
                constructed_file_dict['numeric_line_indices'] = numeric_line_indices
                constructed_file_dict['reduced_row_indices'] = reduced_row_indices_candidate
                constructed_file_dict['exec_time'] = {}
                row_reduced_constructed_files.append(constructed_file_dict)


        return column_reduced_constructed_files, row_reduced_constructed_files

    def _construct_decimal_file_array(self):
        file_values = np.array(self.original_file.valid_number_formats[self.original_file.number_format])
        decimal_file_values = np.full_like(file_values, fill_value=math.nan, dtype=float)
        for index, value in np.ndenumerate(file_values):
            decimal_value = str2decimal(value, default=None)
            if decimal_value is not None:
                decimal_file_values[index] = decimal_value
        return decimal_file_values

    def map_result_indices(self):
        pass

    def run(self, _file):
        self.original_file = _file
        if self.original_file.number_format not in self.original_file.aggregation_detection_result:
            return None
        detected_aggregations = self.original_file.aggregation_detection_result[self.original_file.number_format]
        column_reduced_constructed_files, row_reduced_constructed_files = self._construct_files(detected_aggregations)
        detect_new = self._ignore_existing(column_reduced_constructed_files, row_reduced_constructed_files)

        return detect_new


class SupplementalSumDetection(SupplementalAggregationDetection):

    def __init__(self):
        super(SupplementalSumDetection, self).__init__()
        self.operator = AggregationOperator.SUM.value
        self.task_name = self.__class__.__name__

    def _ignore_existing(self, column_reduced_files, row_reduced_files):
        detect_new = False

        column_reduced_solutions_output = []
        for column_reduced_file_dict in column_reduced_files:
            file = File(column_reduced_file_dict)
            sum_detector = SumDetection(file, parameters=file.parameters)
            row_wise_aggregations, _ = sum_detector.run()
            column_reduced_solutions_output.append(row_wise_aggregations)

        detected_result_across_solutions = {}
        for solution in column_reduced_solutions_output:
            for supplemental_detected_result in solution.aggregation_detection_result[self.original_file.number_format]:
                aggregator_index_reduced = ast.literal_eval(supplemental_detected_result[0])
                mapped_aggregator_index = str(
                    (aggregator_index_reduced[0], solution.reduced_column_indices[aggregator_index_reduced[1]]))
                aggregatee_indices_reduced = [ast.literal_eval(e) for e in supplemental_detected_result[1]]
                mapped_aggregatee_indices = [str((e[0], solution.reduced_column_indices[e[1]])) for e in aggregatee_indices_reduced]
                mapped_result = [mapped_aggregator_index, mapped_aggregatee_indices, supplemental_detected_result[2], supplemental_detected_result[3]]
                signature = (ast.literal_eval(mapped_aggregator_index)[1], tuple(sorted([ast.literal_eval(e)[1] for e in mapped_aggregatee_indices])))
                if signature not in detected_result_across_solutions:
                    detected_result_across_solutions[signature] = []
                if mapped_result not in detected_result_across_solutions[signature]:
                    detected_result_across_solutions[signature].append(mapped_result)
                    detect_new = True
        detected_result_across_solutions = {k: v for k, v in
                                            sorted(detected_result_across_solutions.items(), key=lambda item: (len(item[0][1]), len(item[1])), reverse=True)}

        filtered_results = filter(detected_result_across_solutions)
        results = itertools.chain(*[r for r in filtered_results.values()])
        for result in results:
            if result not in self.original_file.aggregation_detection_result[self.original_file.number_format]:
                self.original_file.aggregation_detection_result[self.original_file.number_format].append(result)

        row_reduced_solutions_output = []
        for row_reduced_file_dict in row_reduced_files:
            file = File(row_reduced_file_dict)
            sum_detector = SumDetection(file, parameters=file.parameters)
            _, column_wise_aggregations = sum_detector.run()
            row_reduced_solutions_output.append(column_wise_aggregations)

        # if best_row_reduced_solution:
        detected_result_across_solutions = {}
        for solution in row_reduced_solutions_output:
            for supplemental_detected_result in solution.aggregation_detection_result[self.original_file.number_format]:
                aggregator_index_reduced = ast.literal_eval(supplemental_detected_result[0])
                mapped_aggregator_index = str(
                    (solution.reduced_row_indices[aggregator_index_reduced[0]], aggregator_index_reduced[1]))
                aggregatee_indices_reduced = [ast.literal_eval(e) for e in supplemental_detected_result[1]]
                mapped_aggregatee_indices = [str((solution.reduced_row_indices[e[0]], e[1])) for e in aggregatee_indices_reduced]
                mapped_result = [mapped_aggregator_index, mapped_aggregatee_indices, supplemental_detected_result[2], supplemental_detected_result[3]]
                signature = (ast.literal_eval(mapped_aggregator_index)[0], tuple(sorted([ast.literal_eval(e)[0] for e in mapped_aggregatee_indices])))
                if signature not in detected_result_across_solutions:
                    detected_result_across_solutions[signature] = []
                if mapped_result not in detected_result_across_solutions[signature]:
                    detected_result_across_solutions[signature].append(mapped_result)
                    detect_new = True
        detected_result_across_solutions = {k: v for k, v in
                                            sorted(detected_result_across_solutions.items(), key=lambda item: (len(item[0][1]), len(item[1])), reverse=True)}

        filtered_results = filter(detected_result_across_solutions)
        results = itertools.chain(*[r for r in filtered_results.values()])
        for result in results:
            if result not in self.original_file.aggregation_detection_result[self.original_file.number_format]:
                self.original_file.aggregation_detection_result[self.original_file.number_format].append(result)

        return detect_new


class SupplementalAverageDetection(SupplementalAggregationDetection):

    def _ignore_existing(self, column_reduced_files, row_reduced_files):
        detect_new = False

        column_reduced_solutions_output = []
        for column_reduced_file_dict in column_reduced_files:
            file = File(column_reduced_file_dict)
            detector = AverageDetection(file, parameters=file.parameters)
            row_wise_aggregations, _ = detector.run()
            column_reduced_solutions_output.append(row_wise_aggregations)

        detected_result_across_solutions = {}
        for solution in column_reduced_solutions_output:
            for supplemental_detected_result in solution.aggregation_detection_result[self.original_file.number_format]:
                aggregator_index_reduced = ast.literal_eval(supplemental_detected_result[0])
                mapped_aggregator_index = str(
                    (aggregator_index_reduced[0], solution.reduced_column_indices[aggregator_index_reduced[1]]))
                aggregatee_indices_reduced = [ast.literal_eval(e) for e in supplemental_detected_result[1]]
                mapped_aggregatee_indices = [str((e[0], solution.reduced_column_indices[e[1]])) for e in aggregatee_indices_reduced]
                mapped_result = [mapped_aggregator_index, mapped_aggregatee_indices, supplemental_detected_result[2], supplemental_detected_result[3]]
                signature = (ast.literal_eval(mapped_aggregator_index)[1], tuple(sorted([ast.literal_eval(e)[1] for e in mapped_aggregatee_indices])))
                if signature not in detected_result_across_solutions:
                    detected_result_across_solutions[signature] = []
                if mapped_result not in detected_result_across_solutions[signature]:
                    detected_result_across_solutions[signature].append(mapped_result)
                    detect_new = True
        detected_result_across_solutions = {k: v for k, v in
                                            sorted(detected_result_across_solutions.items(), key=lambda item: (len(item[0][1]), len(item[1])), reverse=True)}

        filtered_results = filter(detected_result_across_solutions)
        results = itertools.chain(*[r for r in filtered_results.values()])
        for result in results:
            if result not in self.original_file.aggregation_detection_result[self.original_file.number_format]:
                self.original_file.aggregation_detection_result[self.original_file.number_format].append(result)

        row_reduced_solutions_output = []
        for row_reduced_file_dict in row_reduced_files:
            file = File(row_reduced_file_dict)
            detector = AverageDetection(file, parameters=file.parameters)
            _, column_wise_aggregations = detector.run()
            row_reduced_solutions_output.append(column_wise_aggregations)

        detected_result_across_solutions = {}
        for solution in row_reduced_solutions_output:
            for supplemental_detected_result in solution.aggregation_detection_result[self.original_file.number_format]:
                aggregator_index_reduced = ast.literal_eval(supplemental_detected_result[0])
                mapped_aggregator_index = str(
                    (solution.reduced_row_indices[aggregator_index_reduced[0]], aggregator_index_reduced[1]))
                aggregatee_indices_reduced = [ast.literal_eval(e) for e in supplemental_detected_result[1]]
                mapped_aggregatee_indices = [str((solution.reduced_row_indices[e[0]], e[1])) for e in aggregatee_indices_reduced]
                mapped_result = [mapped_aggregator_index, mapped_aggregatee_indices, supplemental_detected_result[2], supplemental_detected_result[3]]
                signature = (ast.literal_eval(mapped_aggregator_index)[0], tuple(sorted([ast.literal_eval(e)[0] for e in mapped_aggregatee_indices])))
                if signature not in detected_result_across_solutions:
                    detected_result_across_solutions[signature] = []
                if mapped_result not in detected_result_across_solutions[signature]:
                    detected_result_across_solutions[signature].append(mapped_result)
                    detect_new = True
        detected_result_across_solutions = {k: v for k, v in sorted(detected_result_across_solutions.items(), key=lambda item: (len(item[0][1]), len(item[1])), reverse=True)}

        filtered_results = filter(detected_result_across_solutions)
        results = itertools.chain(*[r for r in filtered_results.values()])
        for result in results:
            if result not in self.original_file.aggregation_detection_result[self.original_file.number_format]:
                self.original_file.aggregation_detection_result[self.original_file.number_format].append(result)

        return detect_new
