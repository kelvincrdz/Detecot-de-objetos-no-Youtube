"""
https://github.com/amdegroot/ssd.pytorch
Updated by: F Kelvin C Belo
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from layers import *
from data import voc, coco
import os


class SSD(nn.Module):
    """Arquitetura de caixa múltipla de disparo único
     A rede é composta por uma rede VGG básica seguida pela rede
     camadas de conversão de caixa múltipla adicionadas. Cada camada multibox se ramifica em
         1) conv2d para pontuações de configuração de classe
         2) conv2d para previsões de localização
         3) camada de caixa anterior associada para produzir delimitação padrão
            caixas específicas para o tamanho do mapa de feição da camada.
     Veja: https://arxiv.org/pdf/1512.02325.pdf para mais detalhes.

     Args:
         phase: (string) Pode ser "teste" ou "treinar"
         tamanho: tamanho da imagem de entrada
         base: camadas VGG16 para entrada, tamanho de 300 ou 500
         extras: camadas extras que alimentam as camadas multibox loc e conf
         head: "multibox head" consiste em camadas conv loc e conf
     """
    def __init__(self, phase, size, base, extras, head, num_classes):
        super(SSD, self).__init__()
        self.phase = phase
        self.num_classes = num_classes
        self.cfg = (coco, voc)[num_classes == 21]
        self.priorbox = PriorBox(self.cfg)
        self.priors = self.priorbox.forward() 
        self.size = size


        self.vgg = nn.ModuleList(base)

        self.L2Norm = L2Norm(512, 20)
        self.extras = nn.ModuleList(extras)

        self.loc = nn.ModuleList(head[0])
        self.conf = nn.ModuleList(head[1])


        if phase == 'test':
            self.softmax = nn.Softmax(dim=-1)

            self.detect = Detect()



    def forward(self, x):
       """Aplica camadas de rede e operações na(s) imagem(ns) de entrada x.

         Args:
             x: imagem de entrada ou lote de imagens. Forma: [lote, 3.300.300].

         Retornar:
             Dependendo da fase:
             teste:
                 Variável (tensor) de previsões de rótulo de classe de saída,
                 pontuação de confiança e previsões de localização correspondentes para
                 cada objeto detectado. Forma: [lote, topk, 7]

             Comboio:
                 lista de saídas concat de:
                     1: camadas de confiança, Forma: [lote*num_priors,num_classes]
                     2: camadas de localização, Forma: [lote,num_priors*4]
                     3: camadas da caixa anterior, Forma: [2,num_priors*4]
         """
        sources = list()
        loc = list()
        conf = list()


        for k in range(23):
            x = self.vgg[k](x)

        s = self.L2Norm(x)
        sources.append(s)

        for k in range(23, len(self.vgg)):
            x = self.vgg[k](x)

        sources.append(x)

        for k, v in enumerate(self.extras):
            x = F.relu(v(x), inplace=True)
            if k % 2 == 1:
                sources.append(x)

        for (x, l, c) in zip(sources, self.loc, self.conf):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())

        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
        if self.phase == "test":

            output = self.detect.apply(self.num_classes, 0, 200, 0.01, 0.45,

                loc.view(loc.size(0), -1, 4),                   # loc preds
                self.softmax(conf.view(conf.size(0), -1,
                             self.num_classes)),                # conf preds
                self.priors.type(type(x.data))                  # default boxes
            )
        else:
        # train
            output = (
                loc.view(loc.size(0), -1, 4),
                conf.view(conf.size(0), -1, self.num_classes),
                self.priors
            )
        return output

    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            self.load_state_dict(torch.load(base_file,
                                 map_location=lambda storage, loc: storage))
            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')


def vgg(cfg, i, batch_norm=False):
    layers = []
    in_channels = i
    for v in cfg:

        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]

        elif v == 'C':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
    layers += [pool5, conv6,
               nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]
    return layers


def add_extras(cfg, i, batch_norm=False):

    layers = []
    in_channels = i
    flag = False
    for k, v in enumerate(cfg):
        if in_channels != 'S':
            if v == 'S':

                layers += [nn.Conv2d(in_channels, cfg[k + 1],
                           kernel_size=(1, 3)[flag], stride=2, padding=1)]
            else:
                layers += [nn.Conv2d(in_channels, v, kernel_size=(1, 3)[flag])]
            flag = not flag
        in_channels = v
    return layers


def multibox(vgg, extra_layers, cfg, num_classes):
    loc_layers = []
    conf_layers = []
    vgg_source = [21, -2]

    for k, v in enumerate(vgg_source):

        loc_layers += [nn.Conv2d(vgg[v].out_channels,
                                 cfg[k] * 4, kernel_size=3, padding=1)]

        conf_layers += [nn.Conv2d(vgg[v].out_channels,
                        cfg[k] * num_classes, kernel_size=3, padding=1)]

    for k, v in enumerate(extra_layers[1::2], 2):

        loc_layers += [nn.Conv2d(v.out_channels, cfg[k]
                                 * 4, kernel_size=3, padding=1)]

        conf_layers += [nn.Conv2d(v.out_channels, cfg[k]
                                  * num_classes, kernel_size=3, padding=1)]
    return vgg, extra_layers, (loc_layers, conf_layers)


base = {
    '300': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
            512, 512, 512],
    '512': [],
}
extras = {
    '300': [256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256],
    '512': [],
}

mbox = {
    '300': [4, 6, 6, 6, 4, 4],  # number of boxes per feature map location
    '512': [],
}


def build_ssd(phase, size=300, num_classes=21):
    if phase != "test" and phase != "train":
        print("ERROR: Phase: " + phase + " not recognized")
        return
    if size != 300:
        print("ERROR: You specified size " + repr(size) + ". However, " +
              "currently only SSD300 (size=300) is supported!")
        return

    base_, extras_, head_ = multibox(vgg(base[str(size)], 3),
                                     add_extras(extras[str(size)], 1024),
                                     mbox[str(size)], num_classes)
    return SSD(phase, size, base_, extras_, head_, num_classes)
