import importlib, os
from torch.utils.data import DataLoader

def find_dataset_using_name(args):
    """Import the module "data/[dataset_name]_dataset.py".

    In the file, the class called DatasetNameDataset() will
    be instantiated. It has to be a subclass of BaseDataset,
    and it is case-insensitive.
    """

    dataset_name = os.path.split(args.dataset.rstrip('/'))[-1]
    dataset_filename = "streamer.datasets." + dataset_name + "_dataset"
    datasetlib = importlib.import_module(dataset_filename)

    dataset_class = None
    dataset_args_class = None
    target_dataset_name = dataset_name.replace('_', '') + 'dataset'
    target_dataset_args_name = dataset_name.replace('_', '') + 'datasetarguments'
    for name, cls in datasetlib.__dict__.items():
        if name.lower() == target_dataset_name.lower():
            dataset_class = cls
        elif name.lower() == target_dataset_args_name.lower():
            dataset_args_class = cls

    if dataset_class is None:
        raise NotImplementedError("In %s.py, there should be a class name that matches %s in lowercase." % (dataset_filename, target_dataset_name))

    if dataset_args_class is None:
        raise NotImplementedError("In %s.py, there should be an argument class name that matches %s in lowercase." % (dataset_filename, target_dataset_args_name))


    dataset_args = dataset_args_class.from_args(args)
    dataset = dataset_class(dataset_args)
    args.snippet_size = dataset.snippet_size
    loader = DataLoader(dataset=dataset, batch_size=1, num_workers= args.num_workers, pin_memory=True)

    return loader
