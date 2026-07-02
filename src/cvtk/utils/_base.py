import json
import numpy as np


class JsonComplexEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif obj is np.nan:
            return None
        else:
            return super().default(obj)


def as_list(x):
    if isinstance(x, (str, dict)):
        return [x]
    if isinstance(x, (list, tuple)):
        return list(x)
    raise TypeError(f'The input is expected to be a str, dict, list, or tuple, but got {type(x)}.')


def save_json(data, output, indent=4, ensure_ascii=False):
    with open(output, 'w', encoding='utf-8') as fp:
        json.dump(data,
                  fp,
                  indent=indent,
                  ensure_ascii=ensure_ascii,
                  cls=JsonComplexEncoder)

