from pathlib import Path

ALIAS_PATH = Path(__file__).parent.joinpath('data/default.aliases')


class TagAlias:
    def __init__(self, alias_path=ALIAS_PATH):
        if alias_path is not None:
            self.alias_map = self.read_alias(alias_path)
        else:
            self.alias_map = {}

    @staticmethod
    def read_alias(alfile):
        '''Read aliases map from given file'''
        if alfile is None:
            return {}
        almap = {}
        with open(alfile, 'r') as fd:
            for line in fd:
                if line.startswith('#') or line == '\n':
                    continue
                alias, token = line.strip().split()[0:2]
                almap[alias] = token
        return almap

    def get(self, tag):
        return self.alias_map.get(tag, tag)
