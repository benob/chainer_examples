import os

gpu = -1
if 'GPU' in os.environ:
    gpu = int(os.environ['GPU'])

if gpu == -1:
    import util.chainer_cpu_wrapper as selected_wrapper
else:
    import util.chainer_gpu_wrapper as selected_wrapper

wrapper = selected_wrapper.wrapper
