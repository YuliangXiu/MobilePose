import os
import tensorflow as tf

def get_graph_path(model_name):
    return {
        'res_227x227': './cpu-model/res-cpu-140_227x227.t7'
    }[model_name]

def model_wh(model_name):
    width, height = model_name.split('_')[-1].split('x')
    return int(width), int(height)
