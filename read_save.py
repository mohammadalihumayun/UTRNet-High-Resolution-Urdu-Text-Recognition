import os
import pytz
import math
import argparse
from PIL import Image
from datetime import datetime
import csv
import torch
import torch.utils.data

from model import Model
from dataset import NormalizePAD
from utils import CTCLabelConverter, AttnLabelConverter, Logger


def read(opt, device, image_path):
    #print('reading',image_path)
    opt.device = device
    os.makedirs("read_outputs", exist_ok=True)
    datetime_now = str(datetime.now(pytz.timezone('Asia/Kolkata')).strftime("%Y-%m-%d_%H-%M-%S"))
    logger = Logger(f'read_outputs/{datetime_now}.txt')


    """ model configuration """
    if 'CTC' in opt.Prediction:
        converter = CTCLabelConverter(opt.character)
    else:
        converter = AttnLabelConverter(opt.character)
    opt.num_class = len(converter.character)

    if opt.rgb:
        opt.input_channel = 3
    model = Model(opt)
    model = model.to(device)

    # load model
    model.load_state_dict(torch.load(opt.saved_model, map_location=device))
    # logger.log('Loaded pretrained model from %s' % opt.saved_model)
    model.eval()
    #print('model loaded')
    
    if opt.rgb:
        img = Image.open(image_path).convert('RGB')
    else:
        img = Image.open(image_path).convert('L')
    img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    w, h = img.size
    ratio = w / float(h)
    if math.ceil(opt.imgH * ratio) > opt.imgW:
        resized_w = opt.imgW
    else:
        resized_w = math.ceil(opt.imgH * ratio)
    img = img.resize((resized_w, opt.imgH), Image.Resampling.BICUBIC)
    transform = NormalizePAD((1, opt.imgH, opt.imgW))
    img = transform(img)
    img = img.unsqueeze(0)
    img = img.to(device)

    batch_size = img.shape[0]  # Define batch_size here

    preds = model(img)
    preds_size = torch.IntTensor([preds.size(1)] * batch_size)
    
    _, preds_index = preds.max(2)
    preds_str = converter.decode(preds_index.data, preds_size.data)[0]
    #logger.log(preds_str)
    #with open(opt.output_text, mode='a') as file:
    #  file.write(f"{preds_str}\n")
    with open(opt.output_text, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([image_path, preds_str])    
    #print(preds_str)
    return {'detected_text': preds_str}#image_data



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--images_path', required=True, help='path to directory containing images to read')
    parser.add_argument('--saved_model', required=True, help="path to saved_model to evaluation")
    parser.add_argument('--output_text', required=True, help='path to output text')
    """ Data processing """
    parser.add_argument('--batch_max_length', type=int, default=100, help='maximum-label-length')
    parser.add_argument('--imgH', type=int, default=32, help='the height of the input image')
    parser.add_argument('--imgW', type=int, default=400, help='the width of the input image')
    parser.add_argument('--rgb', action='store_true', help='use rgb input')
    """ Model Architecture """
    parser.add_argument('--FeatureExtraction', type=str, default="HRNet", help='FeatureExtraction stage HRNet|VGG|RCNN|ResNet|Densenet|InceptionUnet|ResUnet|AttnUNet|UNet')
    parser.add_argument('--SequenceModeling', type=str, default="DBiLSTM", help='SequenceModeling stage LSTM|GRU|MDLSTM|BiLSTM|DBiLSTM')
    parser.add_argument('--Prediction', type=str, default="CTC", help='Prediction stage CTC|Attn')
    parser.add_argument('--num_fiducial', type=int, default=20, help='number of fiducial points of TPS-STN')
    parser.add_argument('--input_channel', type=int, default=1, help='the number of input channel of Feature extractor')
    parser.add_argument('--output_channel', type=int, default=512, help='the number of output channel of Feature extractor')
    parser.add_argument('--hidden_size', type=int, default=256, help='the size of the LSTM hidden state')
    """ GPU Selection """
    parser.add_argument('--device_id', type=str, default=None, help='cuda device ID')
    
    opt = parser.parse_args()
    if opt.FeatureExtraction == "HRNet":
        opt.output_channel = 32
    """ vocab / character number configuration """
    file = open("UrduGlyphs.txt","r",encoding="utf-8")
    content = file.readlines()
    content = ''.join([str(elem).strip('\n') for elem in content])
    opt.character = content + " "

    cuda_str = 'cuda'
    if opt.device_id is not None:
        cuda_str = f'cuda:{opt.device_id}'
    device = torch.device(cuda_str if torch.cuda.is_available() else 'cpu')
    print("Device : ", device)

    results = []  # Initialize an empty list to store results

    # Iterate over each image file in the specified directory
    for image_file in os.listdir(opt.images_path):
        if image_file.endswith((".png", ".jpg", ".jpeg")):
            image_path = os.path.join(opt.images_path, image_file)
            # print("Processing image:", image_path)
            result = read(opt, device, image_path)  # Call read function with image_path
            results.append(result)

