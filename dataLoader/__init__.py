from .IncidentalData import MethodistFull
from .methodistConfig import *
from .lunaConfig import *
from .LunaData import LunaDataset


def getDataset(args):
    if args.datasource == "luna":
        # Dataset = datald.luna
        config = LunaConfig()
        Dataset = Luna
    elif args.datasource == "methBenMinmax":
        config = DataBenConfigV0()
        Dataset = MethodistFull
    elif args.datasource == "methBenMinmaxNew":
        config = DataBenConfigV1()
        Dataset = MethodistFull
    elif args.datasource == "methKelvinMinmax":
        config = DataKelvinConfigV0()
        Dataset = MethodistFull
    elif args.datasource == "methodist":
        config = IncidentalConfig()
        # config.SPLIT_SEED = args.rseed
        # config.LIMIT_TRAIN = args.limit_train
        # config.MASK_LUNG = args.mask
        # config.CROP_LUNG = args.crop
        # config.AUGTYPE["flip"] = args.flip
        # config.AUGTYPE["swap"] = args.swap
        # config.AUGTYPE["scale"] = args.scale
        # config.AUGTYPE["rotate"] = args.rotate
        # config.AUGTYPE["contrast"] = args.contrast
        # config.AUGTYPE["bright"] = args.bright
        # config.AUGTYPE["sharp"] = args.sharp
        # config.AUGTYPE["splice"] = args.splice
        # config.KFOLD = args.kfold
        # config.SPLIT_ID = args.split_id
        Dataset = MethodistFull
    else:
        print("Unknonw datasource: {:s}".format(args.datasource))
    return config, Dataset