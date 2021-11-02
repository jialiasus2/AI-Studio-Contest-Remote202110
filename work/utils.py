import os
import datetime
import json
import cv2
import numpy as np

USE_PIL = True

def post_process(result):
    '''
    result: 2*H*W
    '''
    if result.shape[0]<=3:
        res = cv2.GaussianBlur(result.transpose([1,2,0]),(5,5),1).transpose([2, 0, 1])
    else:
        res = cv2.GaussianBlur(result,(5,5),1)
    return res

def load_json(json_file):
    with open(json_file, 'r') as f:
        return json.load(f)

def get_val(arr):
    return arr.numpy()[0]

def get_datetime():
    return datetime.datetime.now().strftime('%m%d_%H%M')

class LogWriter():
    def __init__(self, log_path=None, print_out=True, log_time=True, clear_pre_content=True):
        self.log_path = log_path
        self.print_out = print_out
        self.log_time = log_time
        if log_path and clear_pre_content:
            os.system('rm -f '+log_path)

    def __call__(self, log_content):
        log_content = self.add_info(log_content)
        if self.print_out:
            self.Print(log_content)
        if self.log_path:
            self.Save(self.log_path, log_content)

    def add_info(self, log_content):
        if self.log_time:
            log_content = ('LOG[%s]:  '%get_datetime())+log_content
        return log_content

    def Print(self, log_content):
        print(log_content)

    def Save(self, log_path, log_content):
        if log_path:
            with open(log_path, 'a') as f:
                f.write(log_content+'\n')
