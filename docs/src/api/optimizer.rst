Optimizer
#########


.. code-block:: python

    from streamer.optimizers.streamer_optimizer import StreamerOptimizerArguments, StreamerOptimizer

    optim_args = StreamerOptimizerArguments(world_size=8,
                                            alpha=3,
                                            max_layers=3,
                                            optimize_every=100,
                                            average_every=100,
                                            hgn_timescale=True,
                                            hgn_reach=True,
                                            bp_up=True,
                                            bp_down=True)
    optimizer = StreamerOptimizer(optim_args)



.. autoclass:: streamer.optimizers.streamer_optimizer.StreamerOptimizerArguments
    :members:


.. autoclass:: streamer.optimizers.streamer_optimizer.StreamerOptimizer
    :members:
    :special-members:
    :exclude-members: __init__, __weakref__, equal_layers, get_current_params

