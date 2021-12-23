from typing import Dict

__all__ = ['Tag', 'Worker', 'Instance', 'Dataset']

SINGLETON = 'SINGLETON'
NOTAG = 'NOTAG'


class Tag:
    __slots__ = ['id', 'name']

    def __init__(self, id: int, name: str):
        self.id = id
        self.name = name

    # def __hash__(self):
    #     return hash(self.name)

    def __repr__(self):
        return f"Tag({self.id}, {self.name})"


class Worker:
    __slots__ = ['id', 'name', 'inst2tag']

    def __init__(self, id: int, name: str):
        self.id = id
        self.name = name
        self.inst2tag = {}

    # def __hash__(self):
    #     return hash(self.name)

    def __repr__(self):
        return f"Worker({self.id}, {self.name})"


class Instance:
    __slots__ = ['id', 'name', 'true_tag', 'pred_tag', 'worker2tag']

    def __init__(self, id: int, name: str, true_tag: str = None):
        self.id = id
        self.name = name
        self.true_tag = true_tag
        self.pred_tag = None
        self.worker2tag = {}

    # def __hash__(self):
    #     return hash(self.name)

    def __repr__(self):
        return f"Instance({self.id}, {self.name})"


class Dataset:
    __slots__ = [
        'instances', 'workers', 'tags', 'tagid2name', 'last_tagid',
        'last_instid', 'last_workid'
    ]

    def __init__(self):
        # <name>:<obj> map
        self.instances = {}
        self.workers = {}
        self.tags = {}
        self.tagid2name = {}
        # init id
        self.last_tagid = -1
        self.last_instid = -1
        self.last_workid = -1

    def __repr__(self):
        return "Datset(instnum={}, worknum={}, tagnum={})".format(
            self.last_instid + 1, self.last_workid + 1, self.last_tagid + 1)

    def add_instance(self,
                     name: str,
                     worker: str,
                     tag: str,
                     true_tag: str = ''):
        if name in self.instances:
            ins = self.instances[name]
        else:
            self.last_instid += 1
            cur_id = self.last_instid
            ins = Instance(id=cur_id, name=name, true_tag=true_tag)
        # update worker2tag
        wk = self.add_worker(worker, tag, ins)
        ins.worker2tag[wk] = self.tags[tag]
        # add worker
        self.instances[name] = ins

    def add_worker(self, name: str, tag: str, inst: Instance) -> Worker:
        if name in self.workers:
            worker = self.workers[name]
        else:
            self.last_workid += 1
            cur_id = self.last_workid
            worker = Worker(cur_id, name)
        # add tag
        t = self.add_tag(tag)
        # update worker
        if inst not in worker.inst2tag:
            worker.inst2tag[inst] = t
        # add worker
        self.workers[name] = worker
        return worker

    def add_tag(self, name) -> Tag:
        if name in self.tags:
            t = self.tags[name]
        else:
            self.last_tagid += 1
            cur_id = self.last_tagid
            t = Tag(cur_id, name)
            self.tags[name] = t
        # update tagid2name
        self.tagid2name[t.id] = t.name
        return t