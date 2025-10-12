# Copyright (c) Megvii Inc. All rights reserved.
# This version disables the fast C++ evaluation fallback for compatibility.

from pycocotools.cocoeval import COCOeval as pycocoeval

class COCOeval_opt(pycocoeval):
    """
    Compatibility fallback for environments where fast C++ evaluation is unavailable.
    Inherits standard COCOeval behavior from pycocotools.
    """

    def __init__(self, cocoGt=None, cocoDt=None, iouType='bbox'):
        super().__init__(cocoGt, cocoDt, iouType)
