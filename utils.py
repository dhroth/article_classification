from datetime import datetime
import os

def log(message):
    print str(datetime.now()) + ": " + message

def noext(path):
    return os.path.split(path)[-1][:-4]
