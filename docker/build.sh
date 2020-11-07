#!/bin/bash
IMAGE_NAME="zaku.sys.es.osaka-u.ac.jp:10081/ohmori/vq-vae_1d:1.7.0-cuda11.0-cudnn8-runtime"
docker build . -t ${IMAGE_NAME}
