#!/bin/bash

while getopts ":d:o:c:t:r:s:" opt; do
  case $opt in
  d)
    dataPath="$OPTARG"
    ;;
  o)
    operator="$OPTARG"
    ;;
  c)
    coverage="$OPTARG"
    ;;
  t)
    timeout="$OPTARG"
    ;;
  r)
    resultPath="$OPTARG"
    ;;
  s)
    stage="$OPTARG"
    ;;
  \?)
    echo "Invalid option: -$OPTARG" >&2
    exit 1
    ;;
  :)
    echo "Option -$OPTARG requires an argument." >&2
    exit 1
    ;;
  esac
done

projectRootPath="../../"

pythonPath="${projectRootPath}aggrdet"
#echo $pythonPath

errorLevel="{\"Sum\": 0.0005, \"Average\": 0.0, \"Division\": 0.0005, \"RelativeChange\": 0.05}"

env PYTHONPATH=$pythonPath luigi --module algorithm AggreCol \
        --local-scheduler \
        --log-level WARNING \
        --dataset-path $dataPath \
        --result-path $resultPath \
        --error-level "$errorLevel" \
        --coverage $coverage \
        --target-aggregation-type $operator \
        --timeout $timeout \
        --stage $stage