import itertools
from loguru import logger
from typing import Generator, List


def items2key(*kwargs):
    """Transform items to sorted key

    win32 -> win32
    (win32, bot) -> bot|win32
    (bot, win32) -> bot|win32
    """
    if len(kwargs) == 1:
        return kwargs[0]
    else:
        kwargs = sorted(kwargs)
        return '|'.join(kwargs)


def key2items(key):
    return key.split('|')


def check_support(item_support, min_support):
    pop_item = [
        item for item, sup in item_support.items() if sup < min_support
    ]
    for item in pop_item:
        del item_support[item]
    return item_support


def check_transaction(trans_gen, item_support, check_len):
    for idx, itemset in enumerate(trans_gen):
        if len(itemset) < check_len:
            continue
        if check_len == 1:
            comb = [i for i in itemset]
        else:
            comb = itertools.combinations(itemset, check_len)
        for items in comb:
            if items2key(*items) not in item_support:
                [itemset.discard(t) for t in items]
                trans_gen[idx] = itemset
    return trans_gen


def setup_item_support(trans_gen, min_support, cur_len=1, item_support={}):
    # cur_len == 1
    if cur_len == 1:
        item_support = {}
        trans_sup = []
        # calculate item_support
        for item_list in trans_gen:
            if len(item_list) == 0:
                continue
            itemset = set([i for i in item_list])
            trans_sup.append(itemset)
            for item in itemset:
                if item in item_support:
                    item_support[item] += 1
                else:
                    item_support[item] = 1
        # filter item_support
        item_support = check_support(item_support, min_support)
        # filter transaction
        trans_sup = check_transaction(trans_sup, item_support, cur_len)
        return item_support, trans_sup
    # cur_len >= 2
    else:
        for itemset in trans_gen:
            if len(itemset) < cur_len:
                continue
            for items in itertools.combinations(itemset, cur_len):
                items = items2key(*items)

                if items in item_support:
                    item_support[items] += 1
                else:
                    item_support[items] = 1
        # filter item_support
        item_support = check_support(item_support, min_support)
        # filter transaction
        trans_gen = check_transaction(trans_gen, item_support, cur_len)
        return item_support, trans_gen


def safe_divide(a, b):
    if b == 0:
        return 0.0
    else:
        return float(a) / b


def rule_learn(trans_gen: Generator,
               min_support: int = 20,
               confidence_threshold: float = 0.8,
               lift_threshold: float = 1.0,
               max_len: int = 2) -> List:
    """Associatin rule for family -> type

    Parameters
    ----------
    trans_gen : Generator
        transaction generator

        For example,
    ```
    [
        ('A', 'B', 'C', 'E', 'F', 'O'),
        ('A', 'C', 'G'),
        ('E', 'I'),
        ('A', 'C', 'D', 'E', 'G'),
        ('A', 'C', 'E', 'G', 'L'),
        ('E', 'J'),
        ('A', 'B', 'C', 'E', 'F', 'P'),
        ('A', 'C', 'D'),
        ('A', 'C', 'E', 'G', 'M'),
        ('A', 'C', 'E', 'G', 'N'),
    ]
    ```
    min_support : int, optional
        Minimum support, by default 20
    confidence_threshold: float, optional
        confidence threshold, by default 0.8
    lift_threshold: float, optional
        lift threshold, by default 1.0
    max_len : int, optional
        Maximum length of the itemsets generated, by default 2

    Returns
    -------
    List
        List of rule dictionary with following keys:
    <antecedent, consequent, antecedent_support, 
    consequent_support, support, confidence, lift>
    """
    # setup_item_support
    item_support = {}
    trans_count = 0
    for cur_len in range(1, max_len + 1):
        logger.info(f"[+]: setup_item_support -> len = {cur_len}")
        item_support, trans_gen = setup_item_support(trans_gen, min_support,
                                                     cur_len, item_support)
        if cur_len == 1:
            trans_count = len(trans_gen)

    # compute confidence | lift
    def get_rule(antecedent, consequent, union_key):
        antecedent_support = item_support[antecedent]
        consequent_support = item_support[consequent]
        union_support = item_support[union_key]
        confidence = safe_divide(union_support, antecedent_support)
        if confidence < confidence_threshold:
            return None
        lift = trans_count * safe_divide(
            union_support, antecedent_support * consequent_support)
        if lift < lift_threshold:
            return None
        return {
            "antecedent": antecedent,
            "consequent": consequent,
            "antecedent_support": antecedent_support,
            "consequent_support": consequent_support,
            "support": union_support,
            "confidence": confidence,
            "lift": lift
        }

    rules = []
    logger.info("[+]: generate rules")
    for key in item_support:
        items = key2items(key)
        if len(items) == 1:
            continue
        antecedent, consequent = items[0], items2key(*items[1:])
        wait_r = [
            get_rule(antecedent, consequent, key),
            get_rule(consequent, antecedent, key)
        ]
        for r in wait_r:
            if r is not None:
                rules.append(r)
    return rules


def alias_detect(trans_gen: Generator,
                 min_support: int = 20,
                 confidence_threshold: float = 0.8,
                 max_len: int = 2) -> List:
    """Alias detection for malware tags

    Parameters
    ----------
    trans_gen : Generator
        transaction generator

        For example,
    ```
    [
        ('A', 'B', 'C', 'E', 'F', 'O'),
        ('A', 'C', 'G'),
        ('E', 'I'),
        ('A', 'C', 'D', 'E', 'G'),
        ('A', 'C', 'E', 'G', 'L'),
        ('E', 'J'),
        ('A', 'B', 'C', 'E', 'F', 'P'),
        ('A', 'C', 'D'),
        ('A', 'C', 'E', 'G', 'M'),
        ('A', 'C', 'E', 'G', 'N'),
    ]
    ```
    min_support : int, optional
        Minimum support, by default 20
    confidence_threshold: float, optional
        confidence threshold, by default 0.8
    max_len : int, optional
        Maximum length of the itemsets generated, by default 2

    Returns
    -------
    List
        List of rule dictionary with following keys:
    <antecedent, consequent, antecedent_support, 
    consequent_support, support, antecedent_confidence, consequent_confidence>
    """
    # setup_item_support
    item_support = {}
    for cur_len in range(1, max_len + 1):
        logger.info(f"[+]: setup_item_support -> len = {cur_len}")
        item_support, trans_gen = setup_item_support(trans_gen, min_support,
                                                     cur_len, item_support)

    # compute confidence
    def get_alias(antecedent, consequent, union_key):
        antecedent_support = item_support[antecedent]
        consequent_support = item_support[consequent]
        union_support = item_support[union_key]
        ant_conf = safe_divide(union_support, antecedent_support)
        cons_conf = safe_divide(union_support, consequent_support)
        if (ant_conf > confidence_threshold
                and cons_conf > confidence_threshold):
            return {
                "antecedent": antecedent,
                "consequent": consequent,
                "antecedent_support": antecedent_support,
                "consequent_support": consequent_support,
                "support": union_support,
                "antecedent_confidence": ant_conf,
                "consequent_confidence": cons_conf
            }
        else:
            return None

    alias = []
    logger.info("[+]: generate alias")
    for key in item_support:
        items = key2items(key)
        if len(items) == 1:
            continue
        antecedent, consequent = items[0], items2key(*items[1:])
        wait = get_alias(antecedent, consequent, key)
        if wait is not None:
            alias.append(wait)
    return alias