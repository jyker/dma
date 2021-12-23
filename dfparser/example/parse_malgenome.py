import os
import json
import pandas as pd
from pathlib import Path
from dfparser import TagParse, TagVoc, TagAlias
from collections import Counter, defaultdict
from operator import itemgetter
from loguru import logger
from codetiming import Timer

# experiment basic config
PURPOSE = 'dfparser'
TARGET = 'malgenome'
TIME = '2021115'
HOME = Path(__file__).parent
VTPATH = HOME.joinpath(f'data/{TARGET}_vt2.json')
METAPATH = HOME.joinpath(f'data/{TARGET}_meta_sha256_family.csv')

logger.add(f"log/{PURPOSE}_{TARGET}_{TIME}.log", level="DEBUG")
timer = Timer("parse",
              text="Time spent: {:.4f} seconds",
              logger=logger.warning)

# experiment parameters config
K = list(range(1, 16))
RESPATH = f'data/{TARGET}.resp'
GOLDPATH = f'data/{TARGET}.gold'


def load_meta_groundtruth():
    df = pd.read_csv(METAPATH, index_col='sha256')
    df.fillna("", inplace=True)
    return df.to_dict(orient='index')


def generate_vt2():
    with open(VTPATH, 'r') as file:
        for line in file:
            yield json.loads(line)


def topk(tag_counter, k: int = 3):
    result = ['PLH' for i in range(k)]
    if len(tag_counter) == 0:
        return result

    for i in range(min(k, len(tag_counter))):
        result[i] = tag_counter[i]

    return result


def top_k_accuracy_score(y_true, y_predict, *, k=3):
    total = 0
    acc = 0
    for t, plist in zip(y_true, y_predict):
        total += 1
        if t in plist[:k]:
            acc += 1
    return acc / total


def top_k_accuracy_class(y_true, y_predict, *, k=3):
    true_counter = sorted(Counter(y_true).items(),
                          key=itemgetter(1),
                          reverse=True)
    acc = defaultdict(int)
    for t, plist in zip(y_true, y_predict):
        if t in plist[:k]:
            acc[t] += 1
    result = {}
    for t, count in true_counter:
        result[t] = acc[t] / count
    return result


def main():
    TP = TagParse(tagvoc=TagVoc())
    TA = TagAlias()
    gt = load_meta_groundtruth()
    resp = open(RESPATH, 'w')
    gold = open(GOLDPATH, 'w')
    y_true = {}
    y_parsed = {}
    for vt2 in generate_vt2():
        sha256 = vt2['sha256']
        # get ground truth
        ground = TA.get(gt[sha256]['family'])
        y_true[sha256] = ground
        gold.write(f"{sha256} {ground}\n")
        # parse
        result = TP.label(vt2)
        engine_family = {}
        for engine, data in result.scans.items():
            if 'dfparser' not in data:
                continue
            if 'family' not in data['dfparser']:
                continue
            family = data['dfparser']['family']
            engine = engine.lower()
            engine_family[engine] = TA.get(family)
        # top_k tags
        tags = topk([i[0] for i in result.verbose], k=max(K))
        y_parsed[sha256] = tags
        for engine, family in engine_family.items():
            if family in tags[:10]:
                resp.write(f'{sha256} {engine} {family}\n')
    resp.close()
    gold.close()
    # parse metric: P_k
    topk_acc = {}
    for i in K:
        topk_acc[i] = top_k_accuracy_score(y_true.values(),
                                           y_parsed.values(),
                                           k=i)
        logger.warning(f'top_{i} = {topk_acc[i]}')
    # parse metric: class P3
    top3_acc = top_k_accuracy_class(y_true.values(), y_parsed.values(), k=3)
    top3_df = pd.DataFrame.from_dict(top3_acc, orient='index')
    top3_df.to_excel(f'data/{PURPOSE}_{TARGET}_top3.xlsx')


if __name__ == '__main__':
    timer.start()
    main()
    timer.stop()