#!/bin/bash
IMAGE_NAME="zaku.sys.es.osaka-u.ac.jp:10081/multimodal-learning/transformers:miyazawa"
docker build . -t ${IMAGE_NAME}