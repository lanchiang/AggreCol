# Created by lan at 2021/4/28
import json
import os
from copy import deepcopy
from queue import Queue

import luigi
from luigi.mock import MockTarget
from pebble import ProcessPool
from concurrent.futures import TimeoutError

from aggrecol.supplemental import SupplementalSumDetection, SupplementalAverageDetection, SupplementalAggregationDetection
from aggrecol.task.CollectiveAggregationDetectionTask import CollectiveAggregationDetectionTask, combine_aggregation_results
from aggrecol.approach import AggregationDetection
from elements import File
from helpers import AggregationOperator


class SupplementalAggregationDetectionTask(AggregationDetection):
    """
    After aggregation results of different operators are fused, apply some extra rules on these results to further retrieve true positive aggregations
    (supplemental aggregations) that cannot be found in the first stage.
    """
    dataset_path = luigi.Parameter()
    result_path = luigi.Parameter('./debug/')
    error_level_dict = luigi.DictParameter(default={'Sum': 0, 'Average': 0, 'Division': 0, 'RelativeChange': 0})
    target_aggregation_type = luigi.Parameter(default='All')
    coverage = luigi.FloatParameter(default=0.6)
    use_extend_strategy = luigi.BoolParameter(default=False, parsing=luigi.BoolParameter.EXPLICIT_PARSING)
    timeout = luigi.FloatParameter(default=300)
    debug = luigi.BoolParameter(default=False, parsing=luigi.BoolParameter.EXPLICIT_PARSING)

    def output(self):
        return luigi.LocalTarget(os.path.join(self.result_path, 'supplemental-aggregation-detection.jl'))

    def requires(self):
        return CollectiveAggregationDetectionTask(dataset_path=self.dataset_path, result_path=self.result_path,
                                                  error_level_dict=self.error_level_dict,
                                                  target_aggregation_type=self.target_aggregation_type,
                                                  coverage=self.coverage,
                                                  use_extend_strategy=self.use_extend_strategy,
                                                  timeout=self.timeout, debug=self.debug)

    def run(self):
        self.operator = self.target_aggregation_type
        self.task_name = self.__class__.__name__
        with self.input().open('r') as file_reader:
            fused_aggregation_results = [json.loads(line) for line in file_reader]

        # what does this task do?
        # 1) Remove other types of aggregations, and do the individual detection of this operator
        # 2) Apply individual detection of this operator on all other Sum, division results (all combinations)

        print('Conduct %s ...' % self.task_name)

        with ProcessPool(max_workers=self.cpu_count, max_tasks=1) as pool:
            returned_results = pool.map(self.detect_aggregations, fused_aggregation_results, timeout=self.timeout).result()

        supplemental_aggregation_detection_results = []
        while True:
            try:
                result = next(returned_results)
            except StopIteration:
                break
            except TimeoutError:
                pass
            else:
                supplemental_aggregation_detection_results.append(result)

        with self.output().open('w') as file_writer:
            for file_dict in supplemental_aggregation_detection_results:
                file_writer.write(json.dumps(file_dict) + '\n')

    def detect_aggregations(self, file_dict) -> (File, File):
        if self.operator not in AggregationOperator.all() and self.operator != 'All':
            return None, None
        if self.operator == 'All':
            file = File(file_dict)
            detectors_literal = [AggregationOperator.SUM.value, AggregationOperator.AVERAGE.value] * 2
            queue = Queue()
            for detector_literal in detectors_literal:
                queue.put(detector_literal)
            while not queue.empty():
                detector_literal = queue.get()
                detector = self._get_detector(detector_literal)
                if not isinstance(detector, SupplementalAggregationDetection):
                    raise RuntimeError('Illegal class %s' % str(detector.__class__))
                detector.run(file)
            combine_aggregation_results(file_dict)
        return file_dict

    def _get_detector(self, detector_literal):
        return {
            AggregationOperator.SUM.value: SupplementalSumDetection(),
            AggregationOperator.AVERAGE.value: SupplementalAverageDetection()
        }.get(detector_literal, None)

    def _fill_queue_with_detectors(self, detector_literal, queue: Queue, detectors_literal):
        other_detectors_literal = deepcopy(detectors_literal)
        if detector_literal == AggregationOperator.SUM.value:
            other_detectors_literal = [detector_literal for detector_literal in other_detectors_literal if AggregationOperator.SUM.value != detector_literal]
        elif detector_literal == AggregationOperator.AVERAGE.value:
            other_detectors_literal = [detector_literal for detector_literal in other_detectors_literal if AggregationOperator.AVERAGE.value != detector_literal]
        else:
            raise NotImplementedError

        queue_list = list(queue.queue)
        for od in other_detectors_literal:
            if od not in queue_list:
                queue.put(od)
        pass

    def setup_file_dicts(self, file_dicts, caller_name):
        pass