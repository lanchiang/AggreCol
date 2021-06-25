# Created by lan at 2021/2/2
import ast
import json
import math
import os
import time
from copy import copy

import luigi
from tqdm import tqdm

from aggrecol.task.CollectiveAggregationDetectionTask import CollectiveAggregationDetectionTask
from aggrecol.task.SupplementalAggregationDetectionTask import SupplementalAggregationDetectionTask
from elements import AggregationRelation, CellIndex, Cell
from helpers import AggregationOperator


class AggreCol(luigi.Task):
    dataset_path = luigi.Parameter()
    result_path = luigi.Parameter('./results/')
    error_level_dict = luigi.DictParameter(default={'Sum': 0, 'Average': 0, 'Division': 0, 'RelativeChange': 0})
    target_aggregation_type = luigi.Parameter(default='All')
    coverage = luigi.FloatParameter(default=0.7)
    use_extend_strategy = luigi.BoolParameter(default=False, parsing=luigi.BoolParameter.EXPLICIT_PARSING)
    timeout = luigi.FloatParameter(default=300)
    stage = luigi.FloatParameter(default=3)

    debug = luigi.BoolParameter(default=False, parsing=luigi.BoolParameter.EXPLICIT_PARSING)

    def output(self):
        return luigi.LocalTarget(os.path.join(self.result_path, 'aggrecol-results.jl'))

    def requires(self):
        if self.stage == 1:
            raise NotImplementedError
        elif self.stage == 2:
            return CollectiveAggregationDetectionTask(dataset_path=self.dataset_path, result_path=self.result_path,
                                                      error_level_dict=self.error_level_dict,
                                                      target_aggregation_type=self.target_aggregation_type,
                                                      coverage=self.coverage,
                                                      use_extend_strategy=self.use_extend_strategy,
                                                      timeout=self.timeout, debug=self.debug)
        elif self.stage == 3:
            return SupplementalAggregationDetectionTask(dataset_path=self.dataset_path, result_path=self.result_path,
                                                        error_level_dict=self.error_level_dict,
                                                        target_aggregation_type=self.target_aggregation_type,
                                                        use_extend_strategy=self.use_extend_strategy,
                                                        coverage=self.coverage,
                                                        timeout=self.timeout, debug=self.debug)
        else:
            raise RuntimeError

    def run(self):
        with self.input().open('r') as file_reader:
            gathered_detection_results = [json.loads(line) for line in file_reader]

        result_dict = []
        for result in tqdm(gathered_detection_results, desc='Process results'):
            start_time = time.time()
            file_output_dict = copy(result)
            result_by_number_format = result['aggregation_detection_result']
            nf_cands = set(result_by_number_format.keys())
            nf_cands = sorted(list(nf_cands))
            if any([exec_time == math.nan for exec_time in result['exec_time'].values()]):
                pass
            if not bool(nf_cands):
                pass
            else:
                results = []
                for number_format in nf_cands:
                    row_wise_aggrs = result_by_number_format[number_format]
                    row_ar = set()
                    for r in row_wise_aggrs:
                        aggregator = ast.literal_eval(r[0])
                        aggregator = Cell(CellIndex(aggregator[0], aggregator[1]), None)
                        aggregatees = []
                        for e in r[1]:
                            aggregatees.append(ast.literal_eval(e))
                        try:
                            aggregatees = [Cell(CellIndex(e[0], e[1]), None) for e in aggregatees]
                        except TypeError:
                            print(result['file_name'])
                        if r[2] == AggregationOperator.SUM.value or r[2] == AggregationOperator.AVERAGE.value:
                            aggregatees.sort()
                        row_ar.add(AggregationRelation(aggregator, tuple(aggregatees), r[2], None))
                    det_aggrs = row_ar
                    results.append((number_format, det_aggrs))
                results.sort(key=lambda x: len(x[1]), reverse=True)
                number_format = results[0][0]
                file_output_dict['detected_number_format'] = number_format
                det_aggrs = []
                for det_aggr in results[0][1]:
                    if isinstance(det_aggr, AggregationRelation):
                        aees = [(aee.cell_index.row_index, aee.cell_index.column_index) for aee in det_aggr.aggregatees]
                        aor = (det_aggr.aggregator.cell_index.row_index, det_aggr.aggregator.cell_index.column_index)
                        det_aggrs.append({'aggregator_index': aor, 'aggregatee_indices': aees, 'operator': det_aggr.operator})
                file_output_dict['detected_aggregations'] = det_aggrs
                file_output_dict.pop('aggregation_detection_result', None)
                try:
                    file_output_dict['number_formatted_values'] = file_output_dict['valid_number_formats'][number_format]
                except KeyError:
                    print()
                file_output_dict.pop('valid_number_formats', None)

            end_time = time.time()
            exec_time = end_time - start_time
            file_output_dict['exec_time'][self.__class__.__name__] = exec_time
            file_output_dict['parameters']['algorithm'] = self.__class__.__name__

            result_dict.append(file_output_dict)

        with self.output().open('w') as file_writer:
            for file_output_dict in result_dict:
                file_writer.write(json.dumps(file_output_dict) + '\n')


if __name__ == '__main__':
    luigi.run()
