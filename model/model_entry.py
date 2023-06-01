from torch import nn
from model.base.model import FCN


def select_model(args):
    type2model = {
        'fcn': FCN(args),
    }
    model = type2model[args.model_type]
    return model


def equip_multi_gpu(model, args):
    model = nn.DataParallel(model, device_ids=args.gpus)
    return model
