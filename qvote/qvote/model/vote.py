from collections import Counter
from operator import itemgetter
from typing import List
from qvote.common import (Dataset, SINGLETON, NOTAG)


def qvote(sample_engine2family, dsmodel):
    tagname2id = {v: k for k, v in dsmodel.dataset.tagid2name.items()}
    db = dsmodel.dataset
    for sha256, data in sample_engine2family.items():
        f2g = {}
        tags = []
        for engine, family in data.items():
            if family not in tagname2id:
                continue
            if engine not in db.workers:
                continue

            tagid = tagname2id[family]
            q = dsmodel.workers[db.workers[engine].id].pi[tagid, tagid]
            if q >= 0.8:
                tags.append(family)
                f2g[family] = q

        if len(tags) == 0:
            db.instances[sha256].pred_tag = 'SINGLETON'
            continue

        top = sorted(Counter(tags).items(), key=itemgetter(1, 0),
                     reverse=True)[0]
        f2g1 = sorted(f2g.items(), key=itemgetter(1, 0), reverse=True)[0]

        if top[1] >= 2:
            db.instances[sha256].pred_tag = top[0]
        elif f2g1[1] >= 0.95:
            db.instances[sha256].pred_tag = f2g1[0]
        else:
            db.instances[sha256].pred_tag = 'SINGLETON'

    return dsmodel.dataset


def nvote(tags, n: int = 2):
    if len(tags) == 0:
        return NOTAG, {}

    counts = Counter(tags)
    tag_counter = sorted(counts.items(), key=itemgetter(1, 0), reverse=True)
    max_tag = tag_counter[0]
    if max_tag[1] >= n:
        return max_tag[0], tag_counter
    else:
        return SINGLETON, tag_counter


class VoteModel:
    """
    n-voting model
    """
    def __init__(self, n=2):
        self.N = n
        self.instnum = 0  # num instance
        self.tagnum = 0  # num tags
        self.worknum = 0  # num workers

    def initialize(self, db: Dataset):
        self.instnum = db.last_instid + 1
        self.tagnum = db.last_tagid + 1
        self.worknum = db.last_workid + 1

    def infer(self, db: Dataset):
        self.initialize(db)
        for inst in db.instances.values():
            tags = [t.name for t in inst.worker2tag.values()]
            inst.pred_tag = nvote(tags, self.N)[0]
