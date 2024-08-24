import os
from src.utils.Singleton import Singleton


class Constants(metaclass=Singleton):
    __FILE_REALPATH = os.path.dirname(os.path.realpath(__file__))
    PROJECT_ROOT = os.path.abspath(os.path.join(__FILE_REALPATH, '../../'))

    CONFIG_FILE = os.path.join(PROJECT_ROOT, 'conf/config.yaml')
