import os
import re
import string
import datetime
from loguru import logger
from collections import Counter, defaultdict
from operator import itemgetter
from dataclasses import (dataclass, field, asdict)
from typing import (List, Callable, NewType, Dict)

from .config import Config
from .tag import TagVoc, TagChar
from .alias import TagAlias, ALIAS_PATH

# maximum number of tags after tokenizing a engine label
MAX_TAG_NUM = 10

# type
Tag = NewType('Tag', str)
Label = NewType('Label', str)

# global
DIGIT_SET = set(string.digits)
DIGIT_UPPER_SET = set(string.digits + string.ascii_uppercase)


# === ParseResult start === #
class StatusCode:
    OOL = 'OOL: out-of-locator'
    OOT = 'OOT: out-of-type'
    OOP = 'OOP: out-of-platform'
    INV_AV = 'INV_AV: invalid_av'
    INV_LA = 'INV_LA: invalid_label'
    ERR_NAU = 'ERR_NAU: error_null-after-uniform'
    OK = 'OK'


class ParseMode:
    strict = 'strict'
    effort = 'best_effort'


def default_hited(t: Tag) -> bool:
    return False


@dataclass
class Locator:
    name: str = field(default='')
    hited: Callable[[Tag], bool] = field(default=default_hited)
    record: int = field(default=MAX_TAG_NUM)


class TagPosition:
    def __init__(self, tagvoc: TagVoc):
        self.tagvoc = tagvoc
        self.locators = [
            Locator(name='type', hited=self.hit_type),
            Locator(name='platform', hited=self.hit_platform),
            Locator(name='method', hited=self.hit_method),
            # add for using open vocabulary
            Locator(name='family', hited=self.hit_family)
        ]

    def reset_record(self):
        for locator in self.locators:
            locator.record = MAX_TAG_NUM

    def hit_type(self, t: Tag) -> bool:
        return t in self.tagvoc['type']

    def hit_platform(self, t: Tag) -> bool:
        return t in self.tagvoc['platform']

    def hit_family(self, t: Tag) -> bool:
        return t in self.tagvoc['family']

    def hit_method(self, t: Tag) -> bool:
        return t in self.tagvoc['method']

    def locator_hited(self, t: Tag) -> bool:
        for locator in self.locators:
            if locator.hited(t):
                return True
        return False


@dataclass
class ParseResult:
    # result field
    type: str = field(default='')
    platform: str = field(default='')
    family: str = field(default='')
    method: str = field(default='')
    # record parsing status msg
    status: str = field(default=StatusCode.OK)
    # record meta
    engine: str = field(default='')
    label: Label = field(default='')
    mode: str = field(default=ParseMode.strict)
    # verbose
    verbose: str = field(default='')

    def nuts(self):
        nut_key = ['type', 'platform', 'family', 'method']
        data = {}
        for k in nut_key:
            if getattr(self, k) != '':
                data[k] = getattr(self, k)
        return data


def tagchar_all_inset(tag: Tag, char_set: set) -> bool:
    for i in tag:
        if i not in char_set:
            return False
    return True


def tagchar_all_digit(t: Tag) -> bool:
    return tagchar_all_inset(t, DIGIT_SET)


def tagchar_all_upperdigit(t: Tag) -> bool:
    return tagchar_all_inset(t, DIGIT_UPPER_SET)


def tag_digit_ratio(tag: Tag) -> float:
    tag_len = len(tag)
    if tag_len == 0:
        return 0.0

    count = sum(c.isdigit() for c in tag)
    return count / tag_len


# === ParseResult end === #

# === vtscan start === #
INVALID_DATE = '1000-10-10 10:10:10'
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def delta_day(start, end):
    '''delta_data = end - start (day)'''
    try:
        start = datetime.datetime.strptime(start, DATE_FORMAT)
        end = datetime.datetime.strptime(end, DATE_FORMAT)
    except BaseException:
        return -1
    if start == end:
        return 0
    return (end - start).days


@dataclass
class SampleInfo():
    md5: str = field(default='')
    sha1: str = field(default='')
    sha256: str = field(default='')
    total: int = field(default=0)
    positives: int = field(default=0)
    first_seen: str = field(default=INVALID_DATE)
    scan_date: str = field(default=INVALID_DATE)
    scan_delta: int = field(default=-1)
    scans: dict = field(default_factory=dict)
    verbose: list = field(default_factory=list)
    malicious: bool = field(default=False)
    family: str = field(default='')


def parse_vtscans_v2(vtscans: Dict, as_dict=False) -> SampleInfo:
    # init data
    sample = SampleInfo()
    # check necessary
    necessary_keys = ['scans', 'md5', 'sha1', 'sha256']
    for k in necessary_keys:
        try:
            setattr(sample, k, vtscans[k])
        except KeyError:
            raise KeyError(f'{k} is necessary but not found')
    # check recommend
    recommend_keys = ['first_seen', 'scan_date']
    for k in recommend_keys:
        try:
            setattr(sample, k, vtscans[k])
        except KeyError:
            logger.warning(f'{k} is recommend but not found')
    # parse
    sample.scan_delta = delta_day(sample.first_seen, sample.scan_date)
    scans = {}
    for av, res in sample.scans.items():
        sample.total += 1
        if res['detected']:
            sample.positives += 1
            clean_label = ''.join(
                filter(lambda x: x in string.printable,
                       res['result'])).strip()
            if len(clean_label) == 0:
                continue
            scans[av] = {'result': clean_label}
    # update
    sample.scans = scans
    if as_dict:
        return asdict(sample)
    else:
        return sample


# === vtscan end === #


# === nvote start === #
def nvote(tags, n: int = 2):
    if len(tags) == 0:
        return 'NONE', {}

    counts = Counter(tags)
    tag_counter = sorted(counts.items(), key=itemgetter(1, 0), reverse=True)
    max_tag = tag_counter[0]
    if max_tag[1] >= n:
        return max_tag[0], tag_counter
    else:
        return 'SINGLETON', tag_counter


# === nvote end === #


class TagParse:
    ai_engines = [
        'apex', 'acronis', 'babable', 'crowdstrike', 'cybereason', 'cylance',
        'cynet', 'elastic', 'endgame', 'invincea', 'max', 'paloalto',
        'sangfor', 'trapmine', 'whitearmor', 'egambit', 'fileadvisor',
        'sentinelone', ' carpediem (v)'
    ]

    def __init__(self,
                 tagvoc: TagVoc,
                 cachable: bool = True,
                 mode: str = 'prud',
                 config: Config = Config(),
                 alias_path: os.PathLike = ALIAS_PATH,
                 packer_asfamily: bool = False,
                 gen_tags: List[str] = []):
        self.tagvoc = tagvoc
        self.cachable = cachable
        self.mode = mode
        self.alias_path = alias_path
        self.packer_asfamily = packer_asfamily
        self.gen_tags = gen_tags
        # generic label indicator
        self.genlabel = False
        # config
        self.CFG = config
        # init Global Tag Position
        self.GTP = TagPosition(tagvoc=tagvoc)
        # init tagalias
        self.TA = TagAlias(alias_path)

    def remove_generic_tags(self, tag_list: List[Tag],
                            engine: str) -> List[Tag]:
        return [t for t in tag_list if t not in self.CFG.get_gentag()]

    def remove_generic_prefix(self, lb: Label, engine: str) -> Label:
        # prefix
        prefix = self.CFG.get_prefix(engine)
        if len(prefix) == 0:
            return lb

        for i in prefix:
            if i in lb:
                lb = lb.split(i)[-1]
        return lb.strip()

    def remove_generic_suffix(self, lb: Label, engine: str) -> Label:
        # suffix tags
        for i in self.CFG.get_suffix(engine):
            if i in lb:
                lb = lb.split(i)[0]
        return lb

    def continuous_upperdigit_remove(self,
                                     tag_list: List[Tag],
                                     reserved_len=2) -> List[Tag]:
        if len(tag_list) <= reserved_len:
            return tag_list
        # check
        remove = []
        for i in range(reserved_len, len(tag_list)):
            t = tag_list[i]
            if not tagchar_all_upperdigit(t):
                continue
            else:
                if not self.GTP.locator_hited(tag_list[i - 1].lower()):
                    remove.append(t)
        # remove
        for i in remove:
            tag_list.remove(i)
        return tag_list

    def prune_locator_tag(self,
                          tag_list: List[Tag],
                          engine: str = 'default') -> List[Tag]:
        """prune and choose fine-grained locator tag
        
        For example:
            - trojan.pws.onlinegames -> pws.onlinegames
            - virus.w32.trojan.downloader.small -> downloader.w32.small
            - w32.trojan.pws.onlinegames -> w32.pws.onlinegames
            - virus.w32.downloader.small -> downloader.w32.small

        Parameters
        ----------
        engine : str
            engine name
        tag_list : List[str]
            tag list

        Returns
        -------
        List[str]
            tag list
        """
        res = []
        self.GTP.reset_record()
        for tag in tag_list:
            hited = False
            # hit locator
            for locator in self.GTP.locators:
                if locator.hited(tag):
                    hited = True
                    if locator.record == MAX_TAG_NUM:
                        # first hit and added
                        res.append(tag)
                        locator.record = res.index(tag)
                    else:
                        # choose last fine-grained
                        res[locator.record] = tag
                    break
            # add non-locator
            if not hited:
                res.append(tag)
        return res

    def location_first_search(self,
                              tag_list: List[Tag],
                              engine: str = 'default',
                              mode: str = ParseMode.strict,
                              result: ParseResult = None) -> ParseResult:
        if result is None:
            result = ParseResult()
        if len(tag_list) == 0:
            return result

        # === recognize locator tag === #
        self.GTP.reset_record()
        for loc, tag in enumerate(tag_list):
            for locator in self.GTP.locators:
                if locator.hited(tag):
                    locator.record = loc
                    setattr(result, locator.name, tag)
                    break

        # === locate family tag from vocabulary === #
        # if already locate family tag, return
        if result.family != '':
            result.verbose = 'family from vocabulary'
            return result

        # === generic label === #
        # generic label does not search family
        if self.genlabel:
            result.verbose = 'generic label'
            return result

        # === location first search for family tag === #
        def set_recorded(locator_list):
            for locator in locator_list:
                setattr(result, locator.name, tag_list[locator.record])

        def get_search_scope(locator_list):
            records = [i.record for i in locator_list]
            # records == 1
            if len(records) == 1:
                if records[0] == 0:
                    return [records[0] + 1]
                else:
                    return [records[0] - 1, records[0] + 1]
            # records >= 2
            mask = [
                i for i in range(min(records) - 1,
                                 max(records) + 2) if i not in records
            ]
            min_loc, max_loc = min(records), max(records)
            if len(mask) >= 3:
                # search inner
                return [
                    i for i in range(min_loc, max_loc + 1) if i not in records
                ]
            else:
                # search margin
                return [
                    i for i in range(max(min_loc - 1, 0), max_loc + 2)
                    if i not in records
                ]

        def search_family(search_scop):
            for loc in search_scop:
                try:
                    result.family = tag_list[loc]
                    break
                except IndexError:
                    continue

        locator_list = self.GTP.locators
        record_num = sum([i.record != MAX_TAG_NUM for i in locator_list])
        if record_num == 0:
            result.family = tag_list[0]
            if mode == ParseMode.strict:
                result.status = StatusCode.OOL
            return result
        elif record_num == 1:
            # set
            locator = [i for i in locator_list if i.record != MAX_TAG_NUM]
            set_recorded(locator)
            # search
            search_family(get_search_scope(locator))
            return result
        else:
            # set
            locators = [i for i in locator_list if i.record != MAX_TAG_NUM]
            set_recorded(locators)
            # search
            search_family(get_search_scope(locators))
            return result

    def tokenize(self, label: str, engine: str) -> List[Tag]:
        separator = self.CFG.get_separator(engine)
        tag_list = []
        for tag in re.split(separator, label):
            if tag and tag not in tag_list:
                tag_list.append(tag)
        # remove useless tags
        return tag_list[:MAX_TAG_NUM]

    def prune_general_tag(self, tag_list: str, engine: str) -> List[Tag]:
        data = []
        generic_tags = self.CFG.get_gentag(engine) + self.gen_tags
        for tag in tag_list:
            if len(tag) <= 3 and not self.GTP.locator_hited(tag):
                # remove short tag
                continue
            elif tag in generic_tags:
                # remove generic tag
                continue
            else:
                data.append(tag)

        if len(data) == 0:
            return data
        # tag_prefix
        tag_prefix = ['avariantof', 'modificationof', 'suspectedof']
        for i in tag_prefix:
            if i in data[0]:
                data[0] = data[0].split(i)[-1]
        if data[0] == '':
            return data[1:]
        else:
            return data

    def uniform_label(self, label: str, engine: str) -> List[str]:
        # === tokenization === #
        tag_list = self.tokenize(label, engine)
        # === remove whitespace === #
        tag_list = [t.replace(' ', '') for t in tag_list]
        # === remove continuous_upperdigit === #
        tag_list = self.continuous_upperdigit_remove(tag_list)
        # === lower label === #
        tag_list = [i.lower() for i in tag_list]
        # === prune tag === #
        tag_list = self.prune_general_tag(tag_list, engine)
        # === others === #
        return tag_list

    def filepath_like(self, label: str) -> bool:
        # \sav6\work_channel1_12\57745154
        # /sav6/work_channel1_12/57745154
        if label.count('/') > 2:
            return True
        elif label.count('\\') > 0:
            return True
        else:
            return False

    def is_valid_label(self, label):
        # === check none == #
        if label is None:
            return False
        # === check len === #
        if len(label) < 3:
            return False
        # === check printable === #
        for ch in label:
            if ch not in TagChar.printables:
                return False
        # === check filepath-like === #
        if self.filepath_like(label):
            return False
        # # === check invalid tags === #
        # label = label.lower()
        # for tag in self.invalid_tags:
        #     if tag in label:
        #         return False
        # === others === #
        return True

    def is_generic_label(self, engine, label):
        label_low = label.lower()
        for indicator in self.CFG.get_genlabel(engine):
            if indicator in label_low:
                return True
        return False

    def remove_digit_family(self,
                            result: ParseResult,
                            tod: float = 0.5) -> ParseResult:
        ignore = ['cve']
        for i in ignore:
            if i in result.family:
                return result

        digit_ratio = tag_digit_ratio(result.family)
        if digit_ratio >= tod:
            result.family = ''
        return result

    def post_result(self, result: ParseResult, engine: str) -> ParseResult:
        # === remove digit family === #
        result = self.remove_digit_family(result)
        # === remove gen family === #
        if result.family in set(self.CFG.get_genfamily(engine)):
            result.family = ''
        # === remove packer family === #
        typ_tag = self.tagvoc.get('type', result.type)
        if (not self.packer_asfamily and typ_tag is not None
                and typ_tag.node == 'packer'):
            result.family = ''

        return result

    def parse(self,
              label: str,
              engine: str = 'default',
              mode: str = ParseMode.strict) -> ParseResult:
        # init ParseRestult
        engine = engine.lower()
        result = ParseResult(engine=engine, label=label, mode=mode)
        # === ignore ai engines === #
        # if engine in self.ai_engines:
        #     result.status = StatusCode.INV_AV
        #     return result
        # === check label valid === #
        if not self.is_valid_label(label):
            result.status = StatusCode.INV_LA
            return result
        # === remove generic prefix === #
        label = self.remove_generic_prefix(label, engine)
        # === remvoe generic suffix === #
        label = self.remove_generic_suffix(label, engine)
        # === check generic label === #
        '''generic label does not contain a meaningful family tag
        for example:
            - generic.mg.e04965cfbd3584bc
            - genericrxhd-ci!606cb01f0517
            - troj_gen.r03bc0caf20
            - trojan.genericcs.s5480318
        '''
        self.genlabel = self.is_generic_label(engine, label)
        # === uniform label === #
        tag_list = self.uniform_label(label, engine)
        if len(tag_list) == 0:
            result.status = StatusCode.ERR_NAU
            return result
        # === prune tags === #
        tag_list = self.prune_locator_tag(tag_list, engine)
        # === locate family in context of type and platform === #
        result = self.location_first_search(tag_list,
                                            engine,
                                            mode=mode,
                                            result=result)
        # === post_result === #
        result = self.post_result(result, engine)
        # === others === #
        return result

    def remove_duplicate_engine(self, scans):
        label_engine = defaultdict(list)
        for engine, data in scans.items():
            label = data['result']
            '''dupicate label check
                > Emsisoft and F-Secure rules are from avclass
            '''
            # Emsisoft uses same label as
            # GData/ESET-NOD32/BitDefender/Ad-Aware/MicroWorld-eScan,
            # but suffixes ' (B)' to their label. Remove the suffix.
            if label.endswith(' (B)'):
                label = label[:-4]
            # F-Secure uses Avira's engine since Nov. 2018
            # but prefixes 'Malware.' to Avira's label. Remove the prefix.
            if label.startswith('Malware.'):
                label = label[8:]
            # add
            label_engine[label].append(engine)

        # select engines in alphabetical order
        label_engine = {k: sorted(v) for k, v in label_engine.items()}
        valid_scans = {}
        for _, v in label_engine.items():
            valid_scans[v[0]] = scans[v[0]]
        return valid_scans

    def label(self,
              vtscans: Dict,
              tof: int = 2,
              mode: str = ParseMode.strict,
              as_dict=False) -> SampleInfo:
        sample_info = parse_vtscans_v2(vtscans)
        tags = []
        scans = self.remove_duplicate_engine(sample_info.scans)
        for engine, data in scans.items():
            label = data['result']
            parsed = self.parse(label, engine, mode)
            family = self.TA.get(parsed.family)
            if family != '':
                tags.append(family)
            data['dfparser'] = parsed.nuts()
        # update
        sample_info.scans = scans
        # vote
        predict, tag_counter = nvote(tags, n=tof)
        sample_info.family = predict
        sample_info.verbose = tag_counter

        if as_dict:
            return asdict(sample_info)
        else:
            return sample_info
