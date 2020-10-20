IMAGE_NAME="zaku.sys.es.osaka-u.ac.jp:10081/OHMORI/vq-vae_1d:v0.0.0"

docker run --rm -it \
    --privileged \
    --runtime nvidia \
    --shm-size 50G \
    -e LOCAL_UID=$(id -u $USER) \
    -e LOCAL_GID=$(id -g $USER) \
    -e PYTHONPATH=/home/developer/transformers:/home/developer/transformers/experiments/minigrid/dependencies/gym-minigrid:/home/developer/transformers/experiments/minigrid/dependencies/torch-ac \
    -v $(pwd)/../:/home/developer/transformers \
    -v /home/k_miyazawa/sshfs_dir:/home/developer/data \
    --net host \
    -e USE_VNC=true \
    -e DISPLAY=:5 \
    -e BUSID=$(nvidia-xconfig --query-gpu-info | grep BusID | sed -n 1p | cut -d' ' -f 6) \
    -e SCREEN_RESOLUTION=1280x1024x24 \
    -e VNC_PASSWORD=passpass \
    --name $(id -u -n)-transformers \
    ${IMAGE_NAME}
