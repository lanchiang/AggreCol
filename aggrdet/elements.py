# Created by lan at 2021/2/2
import dataclasses
import gzip
import itertools
import json
from dataclasses import dataclass, field
from enum import Enum
from functools import total_ordering
from typing import Tuple, List


@total_ordering
class CellIndex:
    def __init__(self, row_index, column_index) -> None:
        self.row_index = row_index
        self.column_index = column_index

    def __eq__(self, o: object) -> bool:
        if not isinstance(o, CellIndex):
            return False
        return self.__key() == o.__key()

    def __lt__(self, other) -> bool:
        if not isinstance(other, CellIndex):
            return False
        return self.__key() < other.__key()

    def __hash__(self) -> int:
        return hash(self.__key())

    def __key(self):
        return self.row_index, self.column_index

    def __str__(self) -> str:
        return '(%d, %d)' % (self.row_index, self.column_index)

    def __repr__(self) -> str:
        return self.__str__()


@total_ordering
class Cell:
    def __init__(self, cell_index: CellIndex, value: str) -> None:
        self.cell_index = cell_index
        self.value = value

    def __key(self):
        return self.cell_index, self.value

    def __eq__(self, other) -> bool:
        if not isinstance(other, Cell):
            return False
        return self.__key() == other.__key()

    def __lt__(self, other) -> bool:
        if not isinstance(other, Cell):
            return False
        return self.cell_index < other.cell_index

    def __hash__(self) -> int:
        return hash(self.__key())

    def __str__(self) -> str:
        return '%s->%s' % (self.cell_index, self.value)

    def __repr__(self) -> str:
        return self.__str__()


class Direction(Enum):
    # Forward means up for a column line, or left for a row line
    FORWARD = 1
    # Backward means down for a column line, or right for a row line
    BACKWARD = 2
    UNKNOWN = 3
    DIRECTIONLESS = 4


class AggregationRelation:

    def __init__(self, aggregator: Cell, aggregatees: Tuple[Cell], operator: str, direction: Direction):
        self.aggregator = aggregator
        self.aggregatees = aggregatees
        self.operator = operator
        self.direction = direction

        # build signature
        if aggregator.cell_index.row_index == aggregatees[0].cell_index.row_index:
            signature_direction = AggregationSignatureDirection.COLUMN_SIGNATURE
            aggregator_signature = aggregator.cell_index.column_index
            aggregatee_signature = [c.cell_index.column_index for c in aggregatees]
        else:
            signature_direction = AggregationSignatureDirection.ROW_SIGNATURE
            aggregator_signature = aggregator.cell_index.row_index
            aggregatee_signature = [c.cell_index.row_index for c in aggregatees]
        self.aggregation_signature = AggregationSignature(aggregator_signature, aggregatee_signature, signature_direction)

    def __str__(self) -> str:
        return 'Aggregator: %s; Aggregatees: %s; Operator: %s; Signature: %s, Direction: %s' \
               % (self.aggregator, str(self.aggregatees), str(self.operator), str(self.aggregation_signature), str(self.direction.value))

    def __repr__(self) -> str:
        return self.__str__()

    def __eq__(self, o: object) -> bool:
        if not isinstance(o, AggregationRelation):
            return False
        return self.__key() == o.__key()

    def __hash__(self) -> int:
        return hash(self.__key())

    def __key(self):
        return self.aggregator, self.aggregatees, self.operator, self.direction


class AggregationSignatureDirection(Enum):
    ROW_SIGNATURE = 0
    COLUMN_SIGNATURE = 1


# Todo: wrap signature with a class.
class AggregationSignature:

    def __init__(self, aggregator_signature, aggregatee_signature, axis: AggregationSignatureDirection) -> None:
        self.aggregator_signature = aggregator_signature
        self.aggregatee_signature = aggregatee_signature
        self.axis = axis

    def __str__(self) -> str:
        return 'Signature: (%s, %s), direction: %s' % (self.aggregator_signature, str(self.aggregatee_signature), self.axis)

    def __repr__(self) -> str:
        return self.__str__()

    def __eq__(self, o: object) -> bool:
        if not isinstance(o, AggregationSignature):
            return False
        return self.__key() == o.__key()

    def __hash__(self) -> int:
        return hash(self.__key())

    def __key(self):
        return self.aggregator_signature, self.aggregatee_signature, self.axis


@dataclass
class File:
    file_name: str
    sheet_name: str
    num_rows: int
    num_cols: int
    number_format: str
    detected_number_format: str
    file_values: List[list] = field(default_factory=list)
    aggregation_annotations: List[dict] = field(default_factory=list)
    parameters: dict = field(default_factory=dict)
    exec_time: dict = field(default_factory=dict)
    valid_number_formats: dict = field(default_factory=dict)
    numeric_line_indices: dict = field(default_factory=dict)
    aggregation_detection_result: dict = field(default_factory=dict)
    detected_aggregations: list = field(default_factory=list)
    reduced_column_indices: list = field(default_factory=list)
    reduced_row_indices: list = field(default_factory=list)

    def __init__(self, file_dict) -> None:
        self.file_name = file_dict['file_name']
        self.sheet_name = file_dict['table_id'] if 'table_id' in file_dict else file_dict['sheet_name']
        self.num_rows = file_dict['num_rows']
        self.num_cols = file_dict['num_cols']
        self.file_values = file_dict['table_array'] if 'table_array' in file_dict else file_dict['file_values']
        self.aggregation_annotations = file_dict.get('aggregation_annotations', None)
        self.parameters = file_dict.get('parameters', None)
        self.number_format = file_dict['number_format']
        self.detected_number_format = file_dict.get('detected_number_format', None)
        # Todo: exec time should not belong to a file instance
        self.exec_time = file_dict.get('exec_time', None)
        self.valid_number_formats = file_dict.get('valid_number_formats', None)
        self.numeric_line_indices = file_dict.get('numeric_line_indices', None)
        self.aggregation_detection_result = file_dict.get('aggregation_detection_result', None)
        self.detected_aggregations = file_dict.get('detected_aggregations', None)
        self.reduced_column_indices = file_dict.get('reduced_column_indices', None)
        self.reduced_row_indices = file_dict.get('reduced_row_indices', None)


class FileJsonEncoder(json.JSONEncoder):
    def default(self, o):
        if dataclasses.is_dataclass(o):
            return dataclasses.asdict(o)
        return super().default(o)


# json.dumps(foo, cls=FileJsonEncoder)

if __name__ == '__main__':
    input_file = '../data/dataset.jl.gz'
    with gzip.open(input_file, 'r') as file_reader:
        results_dict = [json.loads(line) for line in file_reader]

    file = File(results_dict[0])
    file.aggregation_detection_results = {}
    s = file

    print(file.file_name)
    print(file.sheet_name)
