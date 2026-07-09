import random


def split_dataset(
    data: str|list[str, str]|tuple[str, str],
    output: str|None=None,
    ratios: list[float]|tuple[float]=[0.8, 0.1, 0.1],
    shuffle: bool=True,
    stratify: bool=True,
    random_seed: int|None=None
) -> list[list]:
    """Split a dataset into multiple subsets with specified ratios.

    Splits a dataset (from list or text file) into train/validation/test subsets
    with optional shuffling and stratified sampling for balanced class distribution.

    Args:
        data (str|list|tuple): Dataset to split. Can be:
            - str: Path to text file (one sample per line, optional tab-separated label)
            - list/tuple: List of samples, each element or (sample, label) tuple
        output (str|None): Base path to save split subsets as text files. Files saved as
            output.0, output.1, etc. Default None (no output file).
        ratios (list[float]|tuple[float]): Split ratios. Must sum to 1.0.
            Default [0.8, 0.1, 0.1] (train/val/test split).
        shuffle (bool): Randomly shuffle dataset before splitting. Default True.
        stratify (bool): Maintain class distribution across splits if labels present.
            Default True.
        random_seed (int|None): Seed for reproducible shuffling. Default None (random).

    Returns:
        list[list]: List of subsets. Each subset is a list of samples/records.

    Raises:
        ValueError: If ratios don't sum to 1.0 or data format invalid.

    Examples:
        >>> subsets = split_dataset('data.txt', ratios=[0.8, 0.1, 0.1])
        >>> len(subsets)
        3
        >>> subsets = split_dataset(['img1.jpg', 'img2.jpg'], output='split',
        ...                          ratios=[0.7, 0.3], shuffle=True, random_seed=42)
        >>> with open('split.0', 'r') as f:
        ...     print(f.readlines())
    """
    if abs(1.0 - sum(ratios)) > 1e-10:
        raise ValueError('The sum of `ratios` should be 1.')

    data_ = []
    label_ = []
    if isinstance(data, str):
        with open(data, 'r') as infh:
            for line in infh:
                line = line.strip()
                m = line.split('\t', 2)
                data_.append(line)
                if len(m) > 1:
                    label_.append(m[1])
    elif isinstance(data, (list, tuple)):
        for d in data:
            data_.append(d)
            if len(d) > 1:
                label_.append(d[1])
    else:
        raise ValueError('The input data should be a list or a path to a text file.')
    data = data_
    label = label_

    ratios_cumsum = [0]
    for r in ratios:
        ratios_cumsum.append(r + ratios_cumsum[-1])
    ratios_cumsum[-1] = 1.0

    # shuflle data
    if shuffle:
        if random_seed is not None:
            random.seed(random_seed)
        idx = list(range(len(data)))
        random.shuffle(idx)
        data = [data[i] for i in idx]
        if len(label) > 0:
            label = [label[i] for i in idx]
    
    # group data by label
    datadict = {}
    if stratify and len(label) > 0:
        for i, label in enumerate(label):
            if label not in datadict:
                datadict[label] = []
            datadict[label].append(data[i])
    
    # split data
    data_subsets = []
    label_subsets = []
    for i in range(len(ratios)):
        data_subsets.append([])
        label_subsets.append([])
        if len(datadict) > 0:
            for cl in datadict:
                n_samples = len(datadict[cl])
                n_splits = [int(n_samples * r) for r in ratios_cumsum]
                data_subsets[i] += datadict[cl][n_splits[i]:n_splits[i + 1]]
                label_subsets[i] += [cl] * (n_splits[i + 1] - n_splits[i])
        else:
            n_samples = len(data)
            n_splits = [int(n_samples * r) for r in ratios_cumsum]
            data_subsets[i] = data[n_splits[i]:n_splits[i + 1]]

    if output is not None:
        for i in range(len(data_subsets)):
            with open(f'{output}.{i}', 'w') as fh:
                for data_record in data_subsets[i]:
                    if isinstance(data_record, (list, tuple)):
                        data_record = '\t'.join(data_record)
                    fh.write(data_record + '\n')
    
    return data_subsets
