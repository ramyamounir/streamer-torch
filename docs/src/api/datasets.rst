Datasets
########

Ego4D dataset
=============

.. code-block:: python

    from torch.utils.data import DataLoader
    from streamer.datasets.ego4d_dataset import Ego4dDatasetArguments, Ego4dDataset

    ego4d_args = Ego4dDatasetArguments(world_size=8, 
                                       rank=1, 
                                       dataset='ego4d', 
                                       frame_size=[128, 128], 
                                       percentage=25, 
                                       split='train')
    ego4d_dataset = Ego4dDataset(ego4d_args)
    loader = DataLoader(dataset=ego4d_dataset, batch_size=1, num_workers= 1, pin_memory=True)


.. autoclass:: streamer.datasets.ego4d_dataset.Ego4dDatasetArguments
    :members:

.. autoclass:: streamer.datasets.ego4d_dataset.Ego4dDataset
    :members:
    :special-members:
    :exclude-members: __init__, __weakref__


====


EPIC-KITCHENS dataset
=====================


.. code-block:: python

    from torch.utils.data import DataLoader
    from streamer.datasets.epic_dataset import EpicDatasetArguments, EpicDataset

    epic_args = EpicDatasetArguments(world_size=8, 
                                      rank=1, 
                                      dataset='ego4d', 
                                      frame_size=[128, 128], 
                                      percentage=25, 
                                      split='train')
    epic_dataset = EpicDataset(epic_args)
    loader = DataLoader(dataset=epic_dataset, batch_size=1, num_workers= 1, pin_memory=True)

.. autoclass:: streamer.datasets.epic_dataset.EpicDatasetArguments
    :members:


.. autoclass:: streamer.datasets.epic_dataset.EpicDataset
    :members:
    :special-members:
    :exclude-members: __init__, __weakref__


