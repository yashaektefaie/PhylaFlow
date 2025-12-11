from ete3 import Tree
import numpy as np
import random
import math
import os

def get_possible_ids(nexus_root):
    ids = []
    for name in os.listdir(nexus_root):
        base, ext = os.path.splitext(name)
        ids.append(base)
    ids.sort()
    return ids