# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch

import importlib


def import_file(module_name, file_path, make_importable=None):
    module = importlib.load_source(module_name, file_path)
    return module
