import importlib

def getOptimizer(args, model):

    model_filename  = f'streamer.optimizers.streamer_optimizer'
    modellib = importlib.import_module(model_filename)

    optimizer_class = None
    optimizer_args_class = None
    target_optimizer_name =  f'StreamerOptimizer'
    target_optimizer_args_name =  f'StreamerOptimizerArguments'
    for name, cls in modellib.__dict__.items():
        if name.lower() == target_optimizer_name.lower():
            optimizer_class = cls
        elif name.lower() == target_optimizer_args_name.lower():
            optimizer_args_class = cls

    if optimizer_class is None:
        raise NotImplementedError("In %s.py, there should be a class name that matches %s in lowercase." % (model_filename, target_optimizer_name))

    if optimizer_args_class is None:
        raise NotImplementedError("In %s.py, there should be a class name that matches %s in lowercase." % (model_filename, target_optimizer_args_name))

    optimizer_args = optimizer_args_class.from_args(args)
    optimizer = optimizer_class(model, optimizer_args)
    return optimizer

