
class paths(object):
    """ This is a class with goals to call all data paths from it. It  simplifies and streamlines the code from long paths.
    It is used following this rules:
    - in the file needs to include the file : import pathconfig,
    - create object from class : paths = pathconfig.paths()
    - call path from property of class: for example path_semcor = paths.TRAIN_DATASET
    Change all path in order to set own path and used them in the code.
    I remember that for path mappings the path are the same. So use this class to call them.
    Many files were deleted so
    """

    def __init__(self):
        #possible formats of files
        self.JSONL = '.jsonl'
        self.CSV = '.csv'
        self.MODEL = '.model'

        #Resources path base
        self.BASE_RESOURCES = _BASE_RES_PATH = '..\\resources\\'

        #Train Data Set
        self.PATH_TRAIN_DATASET = _BASE_RES_PATH + 'train_dataset{}'.format(self.JSONL)
        self.PATH_TEST_DATASET = _BASE_RES_PATH + 'test_dataset_blind{}'.format(self.JSONL)
        self.PATH_PREDICTION_CVS = _BASE_RES_PATH + '1860363{}'.format(self.CSV)

        #Model saved
        self.PATH_MODEL_BINARY_CLASSIFICATION = _BASE_RES_PATH + 'model_binary{}'.format(self.MODEL)
        self.PATH_MODEL_MULTI_CLASS_CLASSIFICATION = _BASE_RES_PATH + 'model_multiclass{}'.format(self.MODEL)