# from . import src
# from .src import *

from . import src

import importlib.util
import logging
import os
import re
import sys
import traceback
from collections import defaultdict, OrderedDict

def ip(user_dir):
    # module_path = getattr(args, 'user_dir', None)
    module_path = user_dir
    if module_path is not None:
        module_path = os.path.abspath(module_path)
        module_parent, module_name = os.path.split(module_path)
        if module_name not in sys.modules:
            sys.path.insert(0, module_parent)
            print(f'moddule: {module_name},,, {module_parent},, {module_path}')
            importlib.import_module(module_name)
            sys.path.pop(0)
#
# # moddule: tree_fairseq,,, /projects/nmt/runs,, /projects/nmt/runs/tree_fairseq
# ip("/projects/nmt/tree_fairseq")