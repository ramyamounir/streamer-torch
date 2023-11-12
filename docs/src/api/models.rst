Models
######

Streamer Model
==============


.. code-block:: python

    from streamer.models import CNNEncoder, CNNDecoder
    from streamer.models import StreamerModelArguments, StreamerModel

    model_args = StreamerModelArguments(max_layers=3,
                                        feature_dim=1024,
                                        evolve_every=50000,
                                        buffer_size=20,
                                        force_fixed_buffer=True,
                                        lr=0.0001,
                                        init_ckpt='',
                                        snippet_size=0.5,
                                        demarcation_mode='average',
                                        window_size=50,
                                        distance_mode='similarity',
                                        force_base_dist=False)
    model = StreamerModel(args=model_args, logger=None, encoder=CNNEncoder, decoder=CNNDecoder)

.. autoclass:: streamer.models.model.StreamerModelArguments
    :members:

.. autoclass:: streamer.models.model.StreamerModel
    :members:
    :special-members:
    :private-members:
    :exclude-members: __init__, __weakref__, init_layer, extract_rep, get_num_layers


====


Streamer Inference Model
========================

.. code-block:: python

    from streamer.models.inference_model import InferenceModel

    model = InferenceModel(checkpoint='to/checkpoint/path/')
    result = model(filename='to/video/file/path')


.. autoclass:: streamer.models.inference_model.InferenceLoader
    :members:

.. autoclass:: streamer.models.inference_model.InferenceModel
    :members:
    :special-members:
    :exclude-members: __init__, __weakref__, get_optimizer


