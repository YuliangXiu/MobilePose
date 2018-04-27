from dataloader import *

ROOT_DIR = "../deeppose_tf/datasets/mpii"  # root dir to the dataset

DEBUG_MODE = False

def get_transform(modeltype, input_size):
    """
    :param modeltype: "resnet" / "mobilenet"
    :param input_size:
    :return:
    """
    if modeltype == "resnet":
        return Rescale((input_size, input_size))
    elif modeltype == "mobilenet":
        return Wrap((input_size, input_size))
    else:
        raise ValueError("modeltype is not wrong")


class DatasetFactory:
    def __init__(self):
        pass

    @staticmethod
    def get_train_dataset(modeltype, input_size, debug=DEBUG_MODE):
        """
        :param modeltype: "resnet" / "mobilenet"
        :return: type: PoseDataset
        Example:
        DataFactory.get_train_dataset("resnet", 224)
        In debug mode, it will return a small dataset
        """
        csv_name = "train_joints.csv"
        if debug:
            csv_name = "train_joints-500.csv"

        return PoseDataset(csv_file=os.path.join(ROOT_DIR, csv_name),
                           transform=transforms.Compose([
                               Augmentation(),
                               # Rescale((inputsize, inputsize)),  # for resnet18
                               # # Wrap((inputsize,inputsize)),# for mobilenetv2
                               get_transform(modeltype, input_size),
                               Expansion(),
                               Guass(),
                               ToTensor()
                           ]))

    @staticmethod
    def get_test_dataset(modeltype, input_size, debug=DEBUG_MODE):
        """
        :param modeltype: resnet / mobilenet
        :return: type: PoseDataset
        Example:
        DataFactory.get_test_dataset("resnet", 224)
        In debug mode, it will return a small dataset
        """
        csv_name = "test_joints.csv"
        if debug:
            csv_name = "test_joints-500.csv"
        return PoseDataset(
            csv_file=os.path.join(ROOT_DIR, csv_name),
            transform=transforms.Compose([
                # Rescale((inputsize, inputsize)),  # for resnet18
                # # Wrap((inputsize, inputsize)),# for mobilenetv2
                get_transform(modeltype, input_size),
                Expansion(),
                Guass(),
                ToTensor()
            ]))
