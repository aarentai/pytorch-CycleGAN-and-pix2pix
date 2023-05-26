#!/usr/bin/env python

import numpy, PIL.Image, torch, scipy, os
from utils import *
from tqdm import tqdm

torch.set_grad_enabled(False) # make sure to not compute gradients for computational performance
torch.backends.cudnn.enabled = True # make sure to use cudnn for computational performance

# arguments_strModel = 'bsds500' # only 'bsds500' for now

if __name__ == '__main__':
    input_dir = '/home/ubuntu/hcdai/Projects/Datasets/phtd_raw'
    output_dir = '/home/ubuntu/hcdai/Projects/Datasets/phtd_hed'
    file_name_list = os.listdir(input_dir)

    for i in tqdm(range(len(file_name_list))):
        if not file_name_list[i].endswith('.png') and not file_name_list[i].endswith('.jpg'):
            continue
        img = numpy.array(PIL.Image.open(f'{input_dir}/{file_name_list[i]}'))
        img = img[...,:3]
        # assert len(img.shape)==3, f'This image has {len(img.shape)} dimensions, while only RGB image is allowed.'
        try:
            # normalize pixel value between [0,1], permute the tensor to [3, h, w]
            input = numpy.ascontiguousarray(img[:, :, ::-1].transpose(2, 0, 1).astype(numpy.float32) * (1.0 / 255.0))
            input = torch.FloatTensor(scipy.ndimage.zoom(input, (1, 320/input.shape[1], 480/input.shape[2])))
            output = get_hed(input)
            # scale back to [0,255], permute the tensor to [h, w, 3]
            PIL.Image.fromarray(((1-output.clip(0.0, 1.0).numpy().transpose(1, 2, 0)[:, :, 0]) * 255.0).astype(numpy.uint8)).save(f'{output_dir}/{file_name_list[i]}')
        except RuntimeError as e:
            print(f'{file_name_list[i]} conversion failed')
            continue