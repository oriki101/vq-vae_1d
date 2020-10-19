IMAGE_NAME="zaku.sys.es.osaka-u.ac.jp:10081/OHMORI/vq-vae_1d:ohmori"

docker run --rm -it \
    --privileged \
    --gpus all \
    -e LOCAL_UID=$(id -u $USER) \
    -e LOCAL_GID=$(id -g $USER) \
    -e PYTHONPATH=/home/developer \
    -v $(pwd)/../:/home/developer \
    -p 8888-8899:8889 \
    --name $(id -u -n)-transformers \
    --net host \
    ${IMAGE_NAME}