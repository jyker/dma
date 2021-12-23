import string
import logging
import copy
import pandas as pd
from pathlib import Path
from pyparsing import (ParserElement, Word, Optional, Suppress)

log = logging.getLogger('maltag')
TAGVOC_PATH = Path(__file__).parent.joinpath('data/dma_voc.xlsx')
ENGINE_JACCARD = Path(__file__).parent.joinpath('data/engine_jaccard.csv')

# change to Suppress
ParserElement.inlineLiteralsUsing(Suppress)


class SignName:
    __slots__ = ('_sign', '_name')

    def __init__(self, sign, name):
        self._name = name
        self._sign = sign

    def __get__(self, obj, objtype=None):
        if obj._sign_or_name == 'sign':
            return self._sign
        else:
            return self._name


class Taxonomy:
    __slots__ = ('_sign_or_name', '_sign_name_map')
    # five category
    type = SignName('TYP', 'type')
    platform = SignName('PLA', 'platform')
    family = SignName('FAM', 'family')
    method = SignName('MET', 'method')
    modifier = SignName('MOD', 'modifier')
    # for debug
    uncategorized = SignName('UNC', 'uncategorized')
    outofvoc = SignName('OOV', 'outofvoc')

    def __init__(self):
        self._sign_or_name = 'sign'
        self._sign_name_map = {}
        for name in dir(self):
            if not name.startswith('_'):
                sign = getattr(self, name)
                self._sign_name_map[sign] = name
                self._sign_name_map[name] = sign

    def copy(self):
        return copy.deepcopy(self)

    @property
    def name(self):
        tcd = self.copy()
        tcd._sign_or_name = 'name'
        return tcd

    @property
    def sign(self):
        tcd = self.copy()
        tcd._sign_or_name = 'sign'
        return tcd

    def __getitem__(self, sign_or_name):
        return self._sign_name_map[sign_or_name]


ClassSign = Taxonomy().sign
ClassName = Taxonomy().name


class TagChar:
    alphas = string.ascii_lowercase
    nums = string.digits
    alphanums = alphas + nums
    whitespace = ' \t'
    semicolon = ';'
    punctuation = r'!"#$%&\'()*+,./:;<=>?@[\\]^`{|}~'
    printables = set(string.printable)

    tag = alphanums + '-_'
    sign = string.ascii_uppercase


class TagWord:
    tag = Word(TagChar.tag)
    alias = (Word(TagChar.tag) + Optional(Suppress(TagChar.semicolon)))[0, 10]


class TagIns:
    __slots__ = ['sign', 'name', 'node', 'path', 'alias', 'remark']

    def __init__(self, name, sign, node, alias, remark):
        self.name = name
        self.sign = sign
        self.node = node
        self.path = f'{sign}:{node}:{name}'
        self.alias = set(TagWord.alias.parseString(alias).asList())
        self.remark = remark

    def __hash__(self):
        return hash(self.path)

    def __str__(self):
        return self.path

    def __repr__(self):
        return "{}({})".format(type(self).__name__, self.path)


def load_tagvoc(datapath, sheet_name, load_remark):
    df = pd.read_excel(datapath, sheet_name=sheet_name, dtype=str)
    df.fillna('', inplace=True)

    tag = {}
    for _, data in df.to_dict(orient='index').items():
        if not load_remark:
            data['remark'] = ''
        data = {k: v.strip() for k, v in data.items()}
        t = TagIns(**data)
        if t.name in tag:
            log.warning(
                f'DPVAC: {t.name} -> {sheet_name} | {tag[t.name].sign}')
            continue

        tag[t.name] = t
        tag[t.path] = t
        for i in t.alias:
            if i in tag:
                log.warning(f'DPIAC: {sheet_name} -> {tag[i].name} | {i}')
                continue
            if i is not None:
                tag[i] = t
    return tag


class TagVoc:
    prud_voc = [
        ClassName.type, ClassName.platform, ClassName.family, ClassName.method,
        ClassName.modifier
    ]
    test_voc = [ClassName.type, ClassName.platform, ClassName.method]

    def __init__(self,
                 voc_path=None,
                 voc_list=None,
                 mode='prud',
                 load_remark=False):
        if voc_path is None:
            self.voc_path = TAGVOC_PATH
        else:
            self.voc_path = voc_path
        # mode
        self.mode = mode
        self.load_remark = load_remark
        self.taxonomy = {}
        # load voc
        if voc_list is None:
            if mode == 'test':
                self._load_tag(self.test_voc)
            else:
                self._load_tag(self.prud_voc)
        else:
            self._load_tag(voc_list)
        self._all_tag = self.check_duplicate()
        # voc_list
        self.voc_list = list(self.taxonomy.keys())

    def __getitem__(self, tax_name):
        return self.taxonomy[tax_name]

    def _load_tag(self, sheet_list):
        for sheet in sheet_list:
            self.taxonomy[sheet] = load_tagvoc(self.voc_path, sheet,
                                               self.load_remark)

    def get(self, tax_name, tag, default=None) -> TagIns:
        if tag in self.taxonomy[tax_name]:
            return self.taxonomy[tax_name][tag]
        else:
            return default

    def check_duplicate(self):
        all_voc = {}
        find_duplicate = False
        for voc in self.taxonomy.values():
            for key, tag in voc.items():
                t = all_voc.get(key, None)
                if t is not None and t.path != tag.path:
                    find_duplicate = True
                    log.warning(f'duplicate {t.path} -> {tag.path}')
        if not find_duplicate:
            return all_voc
        else:
            raise RuntimeError('find duplicate')

    def all(self):
        return self._all_tag