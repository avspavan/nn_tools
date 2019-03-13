#!/bin/bash
python3 caffe_analyser.py ~/ncsdk2.01/weights/caffe_files/squeezenet_deploy.prototxt squeezenet_layers.csv 1,3,227,227
python3 caffe_analyser.py ./../networks_layers/yolov2-tiny-voc.prototxt tinyyolov2_layers.csv 1,3,416,416
python3 caffe_analyser.py ~/ncsdk2.01/weights/caffe_files/ResNet-50-deploy.prototxt resnet50_layers.csv 1,3,224,224
python3 caffe_analyser.py ~/ncsdk2.01/weights/caffe_files/VGG_ILSVRC_16_layers_deploy.prototxt vgg16_layers.csv 1,3,224,224
python3 caffe_analyser.py ./../networks_layers/ResNet-152-deploy.prototxt resnet152_layers.csv 1,3,224,224
python3 caffe_analyser.py ./../networks_layers/deploy_inception-v3.prototxt inceptionv3_layers.csv 1,3,299,299

