#!/usr/bin/python3

"""
wrappers for simple input/output functions of json module
"""

__author__ = "Caetano Souto-Maior"
__copyright__ = ""
__credits__ = ["Caetano Souto-Maior"]
__license__ = "GNU Affero General Public License (GNU AGPLv3)"
__version__ = "1.0"
__maintainer__ = "Caetano Souto-Maior"
__email__ = "caetanosoutomaior@protonmail.com"
__status__ = "Development"

import json
import warnings

def read_json_file(filename, printFlag=False):
    try:
        with open(filename, 'r') as ofhandle:
            dump = ofhandle.read()
    except:
        warnings.warn(">>> error writing file, check json dump", UserWarning)

    diciontary = json.loads( dump )
    if printFlag:
        print(dump)

    return diciontary


def write_json_file(filename, diciontary, idnt=None, printFlag=False):

    dump = json.dumps( diciontary, indent=idnt )
    if printFlag:
        print(dump)
    try:
        with open(filename, 'w') as fhandle:
            fhandle.write( dump )
    except:
        warnings.warn(">>> error writing file, check json dump", UserWarning)

    return dump
