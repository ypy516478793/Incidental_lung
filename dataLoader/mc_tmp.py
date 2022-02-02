'''
    masked_files: masked -> resampled, pad = -3000
    preprocessed: normalized -> masked -> resampled, pad = 170
    modeNorm_files: masked -> resampled -> modeNorm, pad = 0

    # DATA_DIR = "/home/cougarnet.uh.edu/pyuan2/Projects/DeepLung-3D_Lung_Nodule_Detection/Methodist_incidental/data_Ben/maskCropDebug"
    # DATA_DIR = "./Methodist_incidental/data_Ben/modeNorm3"
    # DATA_DIR = "/home/cougarnet.uh.edu/pyuan2/Projects/DeepLung-3D_Lung_Nodule_Detection/Methodist_incidental/data_Ben/masked_croped_modeNorm"
    # DATA_DIR = "/home/cougarnet.uh.edu/pyuan2/Projects/DeepLung-3D_Lung_Nodule_Detection/Methodist_incidental/data_unlabeled/masked_with_crop"
    # DATA_DIR = "/home/cougarnet.uh.edu/pyuan2/Projects/Incidental_Lung/data/"
    # DATA_DIR = "/home/cougarnet.uh.edu/pyuan2/Projects/Incidental_Lung/data_king/labeled/"
    # DATA_DIR = "/home/cougarnet.uh.edu/pyuan2/Projects/Incidental_Lung/data_king/unlabeled/"
    # DATA_DIR = "/home/cougarnet.uh.edu/pyuan2/Projects/Incidental_Lung/data/raw_data/unlabeled/"
    # DATA_DIR = "/home/cougarnet.uh.edu/pyuan2/Projects/Incidental_Lung/data_mamta/processed_data/unlabeled/"
'''

class IncidentalConfig(object):

    ## Data Config
    DTYPE = "npz"
    RESAMPLE = True
    MASK = True
    PET_CT = None
    DATA_DIR = None
    POS_LABEL_FILE = None
    BLACK_LIST = None
    # DATA_DIR = "/home/cougarnet.uh.edu/pyuan2/Projects/DeepLung-3D_Lung_Nodule_Detection/Methodist_incidental/data_Ben/maskCropDebug"
    # DATA_DIR = "./Methodist_incidental/data_Ben/modeNorm3"
    # DATA_DIR = "/home/cougarnet.uh.edu/pyuan2/Projects/DeepLung-3D_Lung_Nodule_Detection/Methodist_incidental/data_Ben/masked_croped_modeNorm"
    # DATA_DIR = "/home/cougarnet.uh.edu/pyuan2/Projects/DeepLung-3D_Lung_Nodule_Detection/Methodist_incidental/data_unlabeled/masked_with_crop"
    # DATA_DIR = "/home/cougarnet.uh.edu/pyuan2/Projects/Incidental_Lung/data/"
    # DATA_DIR = "/home/cougarnet.uh.edu/pyuan2/Projects/Incidental_Lung/data_king/labeled/"
    # DATA_DIR = "/home/cougarnet.uh.edu/pyuan2/Projects/Incidental_Lung/data_king/unlabeled/"
    # DATA_DIR = "/home/cougarnet.uh.edu/pyuan2/Projects/Incidental_Lung/data/raw_data/unlabeled/"
    # DATA_DIR = "/home/cougarnet.uh.edu/pyuan2/Projects/Incidental_Lung/data_mamta/processed_data/unlabeled/"
    # POS_LABEL_FILE = None
    # POS_LABEL_FILE = "pos_labels_norm.csv"
    # POS_LABEL_FILE = "gt_labels_checklist.xlsx"
    # POS_LABEL_FILE = "Predicted_labels_checklist_Kim_TC.xlsx"
    # INFO_FILE = "CTinfo.npz"
    # BLACK_LIST = ["001030196-20121205", "005520101-20130316", "009453325-20130820", "034276428-20131212",
    #               "036568905-20150714", "038654273-20160324", "011389806-20160907", "015995871-20160929",
    #               "052393550-20161208", "033204314-20170207", "017478009-20170616", "027456904-20180209",
    #               "041293960-20170227", "000033167-20131213", "022528020-20180525", "025432105-20180730",
    #               "000361956-20180625"]
    MAX_NODULE_SIZE = 60


    ## Model Config
    ANCHORS = [10.0, 30.0, 60.0]  # ANCHORS = [5., 10., 20.]  # [ 10.0, 30.0, 60.]
    CHANNEL = 1
    CROP_SIZE = [96, 96, 96]
    STRIDE = 4
    MAX_STRIDE = 16
    NUM_NEG = 800
    TH_NEG = 0.02
    TH_POS_TRAIN = 0.5
    TH_POS_VAL = 1
    NUM_HARD = 2
    BOUND_SIZE = 12
    RESO = 1
    SIZE_LIM = 2.5  # 3 #6. #mm
    SIZE_LIM2 = 10  # 30
    SIZE_LIM3 = 20  # 40
    AUG_SCALE = True
    R_RAND_CROP = 0.3
    AUGTYPE = {"flip": False, "swap": False, "scale": False, "rotate": False,
               "contrast": False, "bright": False, "sharp": False, "splice": False}
    # AUGTYPE = {"flip": True, "swap": True, "scale": True, "rotate": True}
    CONF_TH = 4
    NMS_TH = 0.3
    DETECT_TH = 0.5
    SIDE_LEN = 144
    MARGIN = 32
    ORIGIN_SCALE = False


    ## Experiment Config
    KFOLD = None
    KFOLD_SEED = None
    SPLIT_SEED = None
    LIMIT_TRAIN = None
    SPLIT_ID = None

    def display(self):
        """Display Configuration values."""
        print("\nConfigurations:")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                print("{:30} {}".format(a, getattr(self, a)))
        print("\n")


class DataBenConfigV0(IncidentalConfig):
    """ preprocessed: normalized -> masked -> resampled, pad = 170 """
    DATA_DIR = "./Methodist_incidental/data_Ben/preprocessed"
    POS_LABEL_FILE = None
    PAD_VALUE = 170
    DTYPE = "npz"
    RESAMPLE = True
    MASK = True
    NORMALIZATION = "min-max"

class DataBenConfigV1(IncidentalConfig):
    """ preprocessed: normalized -> masked(NEW) -> resampled, pad = 170 """
    DATA_DIR = "./Methodist_incidental/data_Ben/preprocessed_data_v1"
    POS_LABEL_FILE = None
    PAD_VALUE = 170
    DTYPE = "npz"
    RESAMPLE = True
    MASK = True
    NORMALIZATION = "min-max"

class DataKelvinConfigV0(IncidentalConfig):
    """ preprocessed: masked -> resampled -> normalized (Min-max), pad = 170 """
    DATA_DIR = "./Methodist_incidental/data_Kelvin/preprocessed"
    POS_LABEL_FILE = None
    PAD_VALUE = 170
    DTYPE = "npz"
    RESAMPLE = True
    MASK = True
    NORMALIZATION = "min-max"

if __name__ == '__main__':
    config = DataBenConfigV0
    print("")