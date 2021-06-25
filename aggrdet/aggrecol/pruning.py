# Created by lan at 2021/3/12
import ast
import itertools
from copy import copy
from statistics import mean

from helpers import AggregationOperator


MAXIMUM_SHARED_AGGREGATOR_COUNT = 3
MAXIMUM_SHARED_AGGREGATEE_COUNT = MAXIMUM_SHARED_AGGREGATOR_COUNT


def prune_conflict_ar_cands(ar_cands_by_line, numeric_line_indices, line_aggregation_satisfied_ratio_threshold, axis=0):
    satisfied_cands_index = {}
    signatures_group_by_aggregatees = {}
    for ar_cands in ar_cands_by_line:
        for ar_cand, error_level in ar_cands:
            aggregator = ar_cand.aggregator
            aggregatees = ar_cand.aggregatees
            if axis == 0:
                ar_tuple = (aggregator.cell_index.column_index, tuple([aggregatee.cell_index.column_index for aggregatee in aggregatees]))
            else:
                ar_tuple = (aggregator.cell_index.row_index, tuple([aggregatee.cell_index.row_index for aggregatee in aggregatees]))
            if ar_tuple not in satisfied_cands_index:
                satisfied_cands_index[ar_tuple] = []
            satisfied_cands_index[ar_tuple].append((ar_cand, error_level))

            if ar_tuple[1] not in signatures_group_by_aggregatees:
                signatures_group_by_aggregatees[ar_tuple[1]] = set()
            signatures_group_by_aggregatees[ar_tuple[1]].add(ar_tuple)

    if axis == 0:
        satisfied_cands_index = {k: v for k, v in satisfied_cands_index.items() if
                                 len(numeric_line_indices[1][str(k[0])]) > 0 and len(v) / len(numeric_line_indices[1][str(k[0])]) >= line_aggregation_satisfied_ratio_threshold}
    else:
        satisfied_cands_index = {k: v for k, v in satisfied_cands_index.items() if
                                 len(numeric_line_indices[0][str(k[0])]) > 0 and len(v) / len(numeric_line_indices[0][str(k[0])]) >= line_aggregation_satisfied_ratio_threshold}

    satisfied_cands_group_by_aggregator_index = {}
    for ar_cand_signature, ar_cands in satisfied_cands_index.items():
        aggregator_index = ar_cand_signature[0]
        if aggregator_index not in satisfied_cands_group_by_aggregator_index:
            satisfied_cands_group_by_aggregator_index[aggregator_index] = {}
        satisfied_cands_group_by_aggregator_index[aggregator_index][ar_cand_signature] = ar_cands
    satisfied_cands_group_by_aggregator_index = {k: {sign: cases for sign, cases in sorted(v.items(), key=lambda x: len(x[1]), reverse=True)}
                                                 for k, v in satisfied_cands_group_by_aggregator_index.items()}

    satisfied_cands_group_by_aggregator_index = {k: {list(v.keys())[0]: list(v.values())[0]} for k, v in satisfied_cands_group_by_aggregator_index.items()}
    satisfied_cands_index = {_k: _v for v in satisfied_cands_group_by_aggregator_index.values() for _k, _v in v.items()}

    satisfied_cands_group_by_aggregatee_index = {}
    for ar_cand_signature, ar_cands in satisfied_cands_index.items():
        aggregatee_indices = ar_cand_signature[1]

        if aggregatee_indices not in satisfied_cands_group_by_aggregatee_index:
            satisfied_cands_group_by_aggregatee_index[aggregatee_indices] = {}
        satisfied_cands_group_by_aggregatee_index[aggregatee_indices][ar_cand_signature] = ar_cands
    satisfied_cands_group_by_aggregatee_index = {k: {sign: cases for sign, cases in sorted(v.items(), key=lambda x: len(x[1]), reverse=True)}
                                                 for k, v in satisfied_cands_group_by_aggregatee_index.items()}

    satisfied_cands_group_by_aggregatee_index = {k: {list(v.keys())[0]: list(v.values())[0]} for k, v in satisfied_cands_group_by_aggregatee_index.items()}
    satisfied_cands_index = {_k: _v for v in satisfied_cands_group_by_aggregatee_index.values() for _k, _v in v.items()}

    satisfied_cands_index = {k: v for k, v in
                             sorted(satisfied_cands_index.items(), key=lambda item: (len(item[1]), - mean([el for _, el in item[1]])),
                                    reverse=True)}
    satisfied_cands_index = {k: [e[0] for e in v] for k, v in satisfied_cands_index.items()}

    non_conflict_ar_cands_index = copy(satisfied_cands_index)
    for ar_index in satisfied_cands_index:
        if ar_index not in non_conflict_ar_cands_index:
            continue
        non_conflict_ar_cands_index[ar_index] = satisfied_cands_index[ar_index]
        non_conflicts = filter_conflict_ar_cands(ar_index, non_conflict_ar_cands_index.keys())
        non_conflict_ar_cands_index = {}
        for non_conflict_indx in non_conflicts:
            non_conflict_ar_cands_index[non_conflict_indx] = satisfied_cands_index[non_conflict_indx]
    return non_conflict_ar_cands_index


def filter_conflict_ar_cands(ar_index, list_ar_cands_index):
    # conflict rule 1: bidirectional aggregation
    survivor_cr1 = []
    for ar_cand_index in list_ar_cands_index:
        if ar_index[0] == ar_cand_index[0]:
            if (ar_cand_index[0] - ar_cand_index[1][0]) * (ar_index[0] - ar_index[1][0]) > 0:
                survivor_cr1.append(ar_cand_index)
        else:
            survivor_cr1.append(ar_cand_index)

    # conflict rule 2: complete inclusion
    survivor_cr2 = []
    for ar_cand_index in survivor_cr1:
        aggee_overlap = list(set(ar_index[1]) & set(ar_cand_index[1]))
        if ar_index[0] in ar_cand_index[1] and bool(aggee_overlap):
            continue
        if ar_cand_index[0] in ar_index[1] and bool(aggee_overlap):
            continue
        survivor_cr2.append(ar_cand_index)

    # # conflict rule 3: partial aggregatees overlap
    # survivor_cr3 = []
    # for ar_cand_index in survivor_cr2:
    #     ar_aggee_set = set(ar_index[1])
    #     ar_cand_aggee_set = set(ar_cand_index[1])
    #     aggee_overlap = list(ar_aggee_set & ar_cand_aggee_set)
    #     if (len(ar_aggee_set) == len(aggee_overlap) and len(ar_cand_aggee_set) == len(aggee_overlap)) or len(aggee_overlap) == 0:
    #         survivor_cr3.append(ar_cand_index)
    survivor_cr3 = survivor_cr2

    # conflict rule 4: aggregator-aggregatee circle
    survivor_cr4 = []
    for ar_cand_index in survivor_cr3:
        if ar_index[0] in ar_cand_index[1] and ar_cand_index[0] in ar_index[1]:
            continue
        survivor_cr4.append(ar_cand_index)

    return survivor_cr4
