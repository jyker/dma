'''Dawid and Skene's model for multi-class annotation.

> Dawid, A. P., & Skene, A. M. (1979). Maximum likelihood estimation 
of observer error-rates using the EM algorithm. Applied statistics, 20-28.
'''
import gc
import math
import warnings
import functools
import multiprocessing as mp
import numpy as np
from loguru import logger
from pathlib import Path
from collections import Counter
from sklearn.metrics import confusion_matrix
from scipy.sparse import (csc_matrix, spdiags)
from typing import List, Dict, NewType
from qvote.common import (Dataset, Instance, Worker, SINGLETON, NOTAG)
from codetiming import Timer
from .vote import nvote

# csc_matrix is more fast in this case, just ignore warning
warnings.filterwarnings("ignore")

TMTEXT = "{name} cost: {seconds:.4f} s"

ID = NewType('ID', int)


def pool_likelihood(workers, theta, inst):
    return inst.compute_likelihood(workers, theta)


def pool_estep(workers, theta, inst):
    inst.e_step(workers, theta)
    return inst


def pool_mstep(instances, work):

    work.m_step(instances)

    return work


class DSWorker:
    __slots__ = ['worker', 'tagnum', 'pi']

    def __init__(self, worker: Worker, tagnum: int, pi):
        self.worker = worker
        self.tagnum = tagnum
        self.pi = pi

    def m_step(self, ds_instances: Dict):
        new_pi = csc_matrix((self.tagnum, self.tagnum), dtype=np.float64)
        for inst, tag in self.worker.inst2tag.items():
            new_pi[:, tag.id] = (new_pi[:, tag.id] +
                                 ds_instances[inst.id].yprobs.reshape((-1, 1)))
        # avoid 0
        self.pi = csc_matrix(new_pi / np.maximum(np.sum(new_pi, axis=0), 1e-7))


class DSInstance:
    __slots__ = ['inst', 'tagnum', 'yprobs', 'fillp']

    def __init__(self, inst: Instance, tagnum: int, yprobs):
        self.inst = inst
        self.tagnum = tagnum
        self.yprobs = yprobs
        # self.fillp = 0.1 / tagnum

    def compute_likelihood(self, ds_workers: Dict[ID, DSWorker], theta):
        prod = np.ones(self.tagnum)
        for worker, tag in self.inst.worker2tag.items():
            pi = ds_workers[worker.id].pi
            prod[tag.id] = prod[tag.id] * np.prod(pi[:, tag.id].data)

        likelihood = np.sum(prod * theta)
        return likelihood

    def e_step(self, ds_workers: List[DSWorker], theta):
        prod = np.ones(self.tagnum)
        for worker, tag in self.inst.worker2tag.items():
            pi = ds_workers[worker.id].pi
            # avoid 0
            # prod *= np.maximum(pi[:, tag.id].toarray(), self.fillp).flatten()
            prod *= pi[:, tag.id].toarray().flatten()
        self.yprobs = prod * theta
        # normalized
        self.yprobs = self.yprobs / np.sum(self.yprobs)

    def predict(self, by_random: bool = True):
        probs = self.yprobs
        if not by_random:
            return np.argmax(probs)
        else:
            return np.random.choice(np.flatnonzero(probs == probs.max()))

    def dumps(self):
        return {
            self.inst.name:
            {tagid: prob
             for tagid, prob in enumerate(self.yprobs)}
        }


class DSModel:
    '''Dawid and Skene's model for multi-class annotation.
    
    > Dawid, A. P., & Skene, A. M. (1979). Maximum likelihood estimation 
      of observer error-rates using the EM algorithm. Applied statistics, 20-28.
    '''
    def __init__(self,
                 max_epoch: int = 20,
                 converge_rate: float = 0.001,
                 num_cpu: int = 4,
                 observe: Dict = None):
        self.max_epoch = max_epoch
        self.converge_rate = converge_rate
        # Dict[ID, DSWorker]
        self.workers = {}
        # Dict[ID, DInstance]
        self.instances = {}
        self.instnum = 0
        self.tagnum = 0
        self.dataset = None
        # one-dimensional
        self.theta = []
        # mp
        self.num_cpu = num_cpu
        # load observe_pi
        self.observe = {}
        if observe is not None:
            for k, v in observe.items():
                self.observe[k] = self.load_pi(v)

    @staticmethod
    def load_pi(fpath):
        return np.load(fpath)

    def _initial_worker_pi(self, worker: Worker):
        y_vote = []
        y_worker = []
        tags = list(self.dataset.tags.keys()) + [NOTAG, SINGLETON]
        for inst in self.dataset.instances.values():
            y_vote.append(inst.pred_tag)
            if worker in inst.worker2tag:
                y_worker.append(inst.worker2tag[worker].name)
            else:
                y_worker.append(NOTAG)

        pi = confusion_matrix(y_true=y_vote,
                              y_pred=y_worker,
                              labels=tags,
                              normalize='true')[:-2, :-2]
        return csc_matrix(pi)

    def _initial_inst_yprobs(self, inst: Instance):
        tags = [t.name for t in inst.worker2tag.values()]
        pred, tag_counter = nvote(tags, n=2)
        inst.pred_tag = pred
        tagnum = self.tagnum
        tags = self.dataset.tags
        yprobs = np.zeros(tagnum, dtype=np.float64)
        if pred == SINGLETON:
            tagids = [tags[i[0]].id for i in tag_counter]
            yprobs[tagids] = 1.0 / len(tagids)
            return csc_matrix(yprobs)
        else:
            pred_id = tags[pred].id
            tagids = [tags[i[0]].id for i in tag_counter][1:]
            num_id = len(tagids)
            if num_id == 0:
                yprobs[pred_id] = 1.0
            else:
                yprobs[pred_id] = 0.8
                yprobs[tagids] = 0.2 / num_id
            return csc_matrix(yprobs)

    def _initial_theta(self):
        tags = self.dataset.tags
        instnum = float(self.instnum)
        pred_tag = Counter([
            inst.pred_tag for inst in self.dataset.instances.values()
            if inst.pred_tag != SINGLETON
        ])
        tagids = []
        p = []
        for k, v in pred_tag.items():
            tagids.append(tags[k].id)
            p.append(v / instnum)
        theta = np.full(shape=self.tagnum,
                        dtype=np.float64,
                        fill_value=(1 - sum(p)) / (instnum - len(tagids)))
        theta[tagids] = p
        return theta

    def initialize(self, db: Dataset):
        self.dataset = db
        self.instnum = db.last_instid + 1
        self.tagnum = db.last_tagid + 1
        self.worknum = db.last_workid + 1

        # chunksize
        self.chunk_work = self.worknum // self.num_cpu
        self.chunk_inst = self.instnum // self.num_cpu

        # init ds_instances and ds_work
        logger.info(f'Initial instances = {self.instnum}')
        for inst in db.instances.values():
            self.instances[inst.id] = DSInstance(
                inst=inst,
                tagnum=self.tagnum,
                yprobs=self._initial_inst_yprobs(inst))
        logger.info(f'Initial workers = {self.worknum}')
        for worker in db.workers.values():
            self.workers[worker.id] = DSWorker(
                worker=worker,
                tagnum=self.tagnum,
                pi=self._initial_worker_pi(worker))
        # init theta
        self.theta = self._initial_theta()
        logger.info(f'Total tags = {self.tagnum}')

    def loglikelihood(self):
        log_like = 0.0
        if self.num_cpu == 1:
            for inst in self.instances.values():
                l = inst.compute_likelihood(self.workers, self.theta)
                log_like += math.log(l)
            return log_like
        else:
            pool = mp.Pool(processes=self.num_cpu)
            compute = functools.partial(pool_likelihood, self.workers,
                                        self.theta)
            results = pool.imap_unordered(compute,
                                          list(self.instances.values()),
                                          chunksize=self.chunk_inst)
            pool.close()
            pool.join()
            for like in results:
                log_like += math.log(like)
            return log_like

    def e_step(self):
        if self.num_cpu == 1:
            for _, inst in self.instances.items():
                inst.e_step(self.workers, self.theta)
        else:
            pool = mp.Pool(processes=self.num_cpu)
            compute = functools.partial(pool_estep, self.workers, self.theta)
            results = pool.imap_unordered(compute,
                                          list(self.instances.values()),
                                          chunksize=self.chunk_inst)
            pool.close()
            pool.join()
            for ist in results:
                self.instances[ist.inst.id].yprobs = ist.yprobs

    def m_step(self):
        if self.num_cpu == 1:
            # update theta
            self.theta.fill(0.0)
            for _, inst in self.instances.items():
                self.theta += inst.yprobs
            self.theta = self.theta / sum(self.theta)
            for _, w in self.workers.items():
                w.m_step(self.instances)
        else:
            pool = mp.Pool(processes=self.num_cpu)
            compute = functools.partial(pool_mstep, self.instances)
            results = pool.imap_unordered(compute,
                                          list(self.workers.values()),
                                          chunksize=self.chunk_work)
            pool.close()
            pool.join()
            for wk in results:
                self.workers[wk.worker.id] = wk

    def inspect_loss(self, epoch):
        if len(self.observe) == 0:
            return

        for k, v in self.observe.items():
            worker_id = self.dataset.workers[k].id
            pi = self.workers[worker_id].pi
            l1 = np.linalg.norm((pi - v), ord=1)
            logger.warning(f'epoch={epoch}: {k} | l1={l1}')

    def em(self):
        logger.info('Start EM')
        epoch = 1
        last_likelihood = -float('inf')
        logger.info('Get initial log-likelihood')
        with Timer(name="loglike_init", text=TMTEXT, logger=logger.info):
            curr_likehihood = self.loglikelihood()
        logger.info(f"DSModel initial log-likelihood = {curr_likehihood}")
        while True:
            dec_rate = (abs(curr_likehihood - last_likelihood) /
                        abs(last_likelihood))
            if (dec_rate < self.converge_rate):
                logger.warning(
                    f"DSModel converage: dec_rate = {dec_rate} | epoch = {epoch}"
                )
                break
            if epoch > self.max_epoch:
                logger.warning(
                    f"DSModel stop: epoch = {epoch} | dec_rate = {dec_rate}")
                break
            # e_m_step
            with Timer(name="e_step", text=TMTEXT, logger=logger.info):
                self.e_step()
            with Timer(name="m_step", text=TMTEXT, logger=logger.info):
                self.m_step()

            last_likelihood = curr_likehihood
            with Timer(name="loglike", text=TMTEXT, logger=logger.info):
                curr_likehihood = self.loglikelihood()
            logger.info(
                f"DSModel: epoch = {epoch} | log-likelihood = {curr_likehihood}"
            )
            epoch += 1
            # gc
            gc.collect()

    def predict(self):
        # logger.info('DSModel start predict')
        for _, ds_inst in self.instances.items():
            inst = ds_inst.inst
            pred_tag = self.dataset.tagid2name[ds_inst.predict(by_random=True)]
            # logger.info(
            #     f"{inst.name} | true={inst.true_tag} | pred={pred_tag}")
            inst.pred_tag = pred_tag

    def infer(self, dataset: Dataset):
        with Timer(name="initialize", text=TMTEXT, logger=logger.info):
            self.initialize(dataset)
        self.em()
        self.predict()