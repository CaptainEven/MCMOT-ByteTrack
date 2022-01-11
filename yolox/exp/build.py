# encoding=utf-8

import importlib
import os
import sys

from loguru import logger


def get_exp_by_file(exp_file):
    """
    :param exp_file:
    :return:
    """
    try:
        exp_file = os.path.abspath(exp_file)
        logger.info("Exp file path: {:s}.".format(exp_file))

        dir_path = os.path.dirname(exp_file)
        logger.info("Exp file's dir path: {:s}.".format(dir_path))
        sys.path.append(dir_path)

        module_name = os.path.basename(exp_file).split(".")[0]
        logger.info("Module name: {:s}.".format(module_name))

        current_exp = importlib.import_module(module_name)
        exp = current_exp.Exp()
    except Exception as e:
        logger.exception(e)
        raise ImportError("{} doesn't contains class named 'Exp'".format(exp_file))

    return exp


def get_exp_by_name(exp_name):
    """
    :param exp_name:
    :return:
    """
    import yolox

    yolox_path = os.path.dirname(os.path.dirname(yolox.__file__))
    filedict = {
        "yolox-s": "yolox_s.py",
        "yolox-m": "yolox_m.py",
        "yolox-l": "yolox_l.py",
        "yolox-x": "yolox_x.py",
        "yolox-tiny": "yolox_tiny_det.py",
        "yolox-nano": "nano.py",
        "yolov3": "yolov3.py",
    }
    filename = filedict[exp_name]
    exp_path = os.path.join(yolox_path, "exps", "default", filename)
    return get_exp_by_file(exp_path)


def get_exp(exp_file, exp_name):
    """
    get Exp object by file or name. If exp_file and exp_name
    are both provided, get Exp by exp_file.
    Args:
    @:param exp_file (str): file path of experiment.
    @:param exp_name (str): name of experiment. "yolo-s",
    """
    assert (exp_file is not None or exp_name is not None), \
        "plz provide exp file or exp name."

    if exp_file is not None:
        exp = get_exp_by_file(exp_file)
    else:
        exp = get_exp_by_name(exp_name)

    return exp
