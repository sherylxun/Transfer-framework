# -*- coding: utf-8 -*-

import sys
import os
import dataclasses
import json
import argparse

from pathlib import Path
from transformers import HfArgumentParser, TrainingArguments

from t2ner.trainers import NERTrainer, NERArguments

from t2ner.models import ModelArguments
from t2ner.models import AutoModelForTokenClassification



def main(args):
    base_json = args.base_json
    exp_json = args.exp_json
    
    if args.exp_type == "ner":
        parser = HfArgumentParser((TrainingArguments, ModelArguments, NERArguments))
        trainer = NERTrainer(*parse_json_config(parser.dataclass_types, base_json, exp_json))
        trainer.run(AutoModelForTokenClassification)
    
    else:
        raise NotImplementedError("Implment please!")

## 将json类配置文件解析成元组形式并返回
def parse_json_config(dataclass_types, base_json, exp_json=None):
    data = json.loads(Path(base_json).read_text())
    if exp_json:
        exp_data = json.loads(Path(exp_json).read_text())
        for k, v in exp_data.items():
            data[k] = v
    outputs = []
    for dtype in dataclass_types:
        keys = {f.name for f in dataclasses.fields(dtype)}
        inputs = {k: v for k, v in data.items() if k in keys}
        obj = dtype(**inputs)
        outputs.append(obj)
    return (*outputs,)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    ## Required parameters
    parser.add_argument(
        "--exp_type",
        default=None, type=str, required=True,
        help="Type of experiment {ner, unsup_adapt, ...}."
    )
    parser.add_argument(
        "--base_json", 
        default=None, type=str, required=True,
        help="Common JSON config file."
    )
    parser.add_argument(
        "--exp_json", 
        default=None, type=str, required=True,
        help="Experiment specific JSON config file."
    )
    parser.add_argument(
        "--method",
        default=None, type=str,
        help="Sub-method in a specific experiment."
    )
    args = parser.parse_args()
    main(args)
