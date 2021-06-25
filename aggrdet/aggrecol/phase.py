# Created by lan at 2021/5/2
from abc import ABC, abstractmethod

from approaches import AggrdetApproach


class IndividualAggregationDetection(AggrdetApproach, ABC):

    def __init__(self, file, parameters) -> None:
        super().__init__(file, parameters)


    @abstractmethod
    def mend_adjacent_aggregations(self, ar_cands_by_line, file_content, error_bound, axis):
        pass

    @abstractmethod
    def is_equal(self, aggregator_value, aggregatees, error_bound):
        pass
