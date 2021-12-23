import pickle
import json
import pandas as pd
from pathlib import Path
from qvote.io import load_data
from qvote.evaluation import Evaluation
from qvote.model import DSModel, qvote
from collections import Counter
from operator import itemgetter
from loguru import logger
from codetiming import Timer

PURPOSE = 'dsmodel'
TARGET = 'malgenome'
TIME = '20211115'
logger.add(f"log/{PURPOSE}_{TARGET}_{TIME}.log", level="DEBUG")
timer = Timer(PURPOSE,
              text="Time spent: {:.4f} seconds",
              logger=logger.warning)

DATAPATH = Path(__file__).parent.joinpath('data')
RESPATH = DATAPATH.joinpath(f'{TARGET}.resp')
GOLDPATH = DATAPATH.joinpath(f'{TARGET}.gold')
PARSEDPATH = DATAPATH.joinpath(f'{TARGET}_sha256_engine_family.json')


def load_parsed():
    with open(PARSEDPATH, 'r') as f:
        return json.load(f)


def main():
    db = load_data(RESPATH, GOLDPATH)

    # infer
    epoch = 20
    ds = DSModel(converge_rate=0.0001, max_epoch=epoch, num_cpu=8)
    ds.infer(db)
    # evaluate
    eva = Evaluation(db)
    # save ds
    with open(f'data/{TARGET}_{TIME}.{PURPOSE}', 'wb') as f:
        pickle.dump(ds, f)

    return ds


def vote(parsed, ds):
    db = qvote(parsed, ds)
    eva = Evaluation(db)
    report = eva.class_report(beta=0.5)
    logger.info(report['micro avg'])
    df_report = pd.DataFrame.from_dict(report, orient='index')
    df_report.index.name = 'family'
    df_report.to_excel(f"data/{TARGET}_qvote_0.5_report.xlsx")


if __name__ == '__main__':
    timer.start()
    ds = main()
    timer.stop()
    # evaluate qvote
    parsed = load_parsed()
    vote(parsed, ds)