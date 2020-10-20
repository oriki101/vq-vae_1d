#!/bin/bash
IMAGE_NAME="zaku.sys.es.osaka-u.ac.jp:10081/OHMORI/vq-vae_1d:v0.0.0"
docker build . -t ${IMAGE_NAME}