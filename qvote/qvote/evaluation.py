# -*- coding: utf-8 -*-
import numpy as np
from collections import Counter
from operator import itemgetter
from sklearn.metrics import precision_recall_fscore_support
from .common import Dataset

__all__ = ['Evaluation']


class Evaluation:
    """
    Performance evaluation
    """
    def __init__(self, db: Dataset, singleton_tags=['SINGLETON']):
        self.db = db
        self.y_true = []
        self.y_pred = []
        self.singleton_tags = singleton_tags
        self.singleton_count = 0
        self.initialize()

    def initialize(self):
        self.singleton_count = 0
        for inst in self.db.instances.values():
            if inst.pred_tag in self.singleton_tags:
                self.singleton_count += 1
                continue
            self.y_true.append(inst.true_tag)
            self.y_pred.append(inst.pred_tag)
        self.true_counter = sorted(Counter(self.y_true).items(),
                                   key=itemgetter(1),
                                   reverse=True)

    def srate(self):
        return self.singleton_count / self.db.last_instid

    def class_report(self,
                     output_dict: bool = True,
                     digits: int = 4,
                     beta: float = 0.5):
        labels = [i[0] for i in self.true_counter]
        return classification_report(y_true=self.y_true,
                                     y_pred=self.y_pred,
                                     labels=labels,
                                     output_dict=output_dict,
                                     digits=digits,
                                     beta=beta)


def classification_report(y_true,
                          y_pred,
                          labels,
                          output_dict: bool = True,
                          digits: int = 4,
                          beta: float = 0.5):
    headers = ['precision', 'recall', f'f{beta}-score', 'support']
    average_options = ("micro", "macro", "weighted")
    # compute per-class results without averaging
    p, r, f1, s = precision_recall_fscore_support(y_true,
                                                  y_pred,
                                                  labels=labels,
                                                  average=None,
                                                  zero_division=0,
                                                  beta=beta)
    rows = zip(labels, p, r, f1, s)
    # import pdb; pdb.set_trace()

    if output_dict:
        report_dict = {label[0]: label[1:] for label in rows}
        for label, scores in report_dict.items():
            report_dict[label] = dict(zip(headers, [i.item() for i in scores]))
    else:
        longest_last_line_heading = "weighted avg"
        name_width = max(len(cn) for cn in labels)
        width = max(name_width, len(longest_last_line_heading), digits)
        head_fmt = "{:>{width}s} " + " {:>9}" * len(headers)
        report = head_fmt.format("", *headers, width=width)
        report += "\n\n"
        row_fmt = "{:>{width}s} " + " {:>9.{digits}f}" * 3 + " {:>9}\n"
        for row in rows:
            report += row_fmt.format(*row, width=width, digits=digits)
        report += "\n"
    # compute all applicable averages

    for average in average_options:
        line_heading = average + " avg"

        # compute averages with specified averaging method
        avg_p, avg_r, avg_f1, _ = precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=labels,
            average=average,
            zero_division=0,
            beta=beta)
        avg = [avg_p, avg_r, avg_f1, np.sum(s)]

        if output_dict:
            report_dict[line_heading] = dict(
                zip(headers, [i.item() for i in avg]))
        else:
            report += row_fmt.format(line_heading,
                                     *avg,
                                     width=width,
                                     digits=digits)
    # output
    if output_dict:
        for field, data in report_dict.items():
            report_dict[field] = {k: round(v, digits) for k, v in data.items()}
        return report_dict
    else:
        return report
