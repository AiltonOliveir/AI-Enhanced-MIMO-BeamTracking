
class Ailton_AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(Ailton_AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

    @classmethod
    def from_dict(cls, d):
        if not isinstance(d, dict):
            return d
        return cls({k: cls.from_dict(v) if isinstance(v, dict) else v for k, v in d.items()})
