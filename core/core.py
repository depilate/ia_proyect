import tensorflow as tf
import numpy as np
from typing import List, Tuple, Union
from abc import ABC, abstractmethod

TF_ENABLE_ONEDNN_OPTS=0

print("TensorFlow version:", tf.__version__)