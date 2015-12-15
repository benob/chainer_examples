import numpy
import chainer

import os

gpu_num = 0
if 'GPU' in os.environ:
    gpu_num = int(os.environ['GPU'])

class wrapper:
    @staticmethod
    def init():
        with chainer.cuda.get_device(gpu_num):
            chainer.cuda.init()

    @staticmethod
    def make_var(array, dtype=numpy.float32):
        with chainer.cuda.get_device(gpu_num):
            return chainer.Variable(chainer.cuda.to_gpu(numpy.array(array, dtype=dtype)))

    @staticmethod
    def get_data(variable):
        with chainer.cuda.get_device(gpu_num):
            return chainer.cuda.to_cpu(variable.data)

    @staticmethod
    def zeros(shape, dtype=numpy.float32):
        with chainer.cuda.get_device(gpu_num):
            return chainer.Variable(chainer.cuda.zeros(shape, dtype=dtype))

    @staticmethod
    def ones(shape, dtype=numpy.float32):
        with chainer.cuda.get_device(gpu_num):
            return chainer.Variable(chainer.cuda.ones(shape, dtype=dtype))

    @staticmethod
    def make_model(**kwargs):
        with chainer.cuda.get_device(gpu_num):
            return chainer.FunctionSet(**kwargs).to_gpu()
 
    @staticmethod
    def begin_model_access(model):
        model.to_cpu()

    @staticmethod
    def end_model_access(model):
        with chainer.cuda.get_device(gpu_num):
            model.to_gpu()

