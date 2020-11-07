IMAGE_NAME="zaku.sys.es.osaka-u.ac.jp:10081/ohmori/vq-vae_1d:1.7.0-cuda11.0-cudnn8-runtime"

docker run --rm -it \
    --privileged \
    --gpus all \
    --shm-size 50G \
    -e LOCAL_UID=$(id -u $USER) \
    -e LOCAL_GID=$(id -g $USER) \
    -e PYTHONPATH=/home/developer/transformers:/home/developer/transformers/experiments/minigrid/dependencies/gym-minigrid:/home/developer/transformers/experiments/minigrid/dependencies/torch-ac \
    -v $(pwd)/../../vq-vae_1d:/home/jovyan/vq-vae_1d\
    --net host \
    -e USE_VNC=true \
    -e DISPLAY=:5 \
    -e BUSID=$(nvidia-xconfig --query-gpu-info | grep BusID | sed -n 1p | cut -d' ' -f 6) \
    -e SCREEN_RESOLUTION=1280x1024x24 \
    -e VNC_PASSWORD=passpass \
    --name $(id -u -n)-transformers \
    ${IMAGE_NAME}
