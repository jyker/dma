# -*- coding: utf-8 -*-
import os
import json
from loguru import logger
from .common import Dataset

__all__ = ['load_data', 'dump_map']


def load_data(respath: os.PathLike, goldpath: os.PathLike = None) -> Dataset:
    DB = Dataset()
    # load gold
    if goldpath is None:
        gold = {}
    else:
        gold = {}
        with open(goldpath, 'r') as file:
            for line in file.readlines():
                ins, tag = line.split()
                gold[ins] = tag
    # load response
    with open(respath, 'r') as file:
        for line in file:
            instance, worker, tag = line.split()
            if gold != {}:
                true_tag = gold[instance]
            else:
                true_tag = ''
            DB.add_instance(name=instance,
                            worker=worker,
                            tag=tag,
                            true_tag=true_tag)
    return DB


def dump_map(dataset: Dataset, dst_path: os.PathLike) -> None:
    data = {
        'instance':
        {inst.id: name
         for name, inst in dataset.instances.items()},
        'worker':
        {worker.id: name
         for name, worker in dataset.workers.items()},
        'tag': {tag.id: name
                for name, tag in dataset.tags.items()}
    }
    with open(dst_path, 'w') as f:
        json.dump(data, f)