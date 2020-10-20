IMAGE_NAME="zaku.sys.es.osaka-u.ac.jp:10081/ohmori/vq-vae_1d:v0.0.0"

docker run --rm -it \
    --privileged \
    -e LOCAL_UID=$(id -u $USER) \
    -e LOCAL_GID=$(id -g $USER) \
    -e PYTHONPATH=/home/developer/transformers:/home/developer/transformers/experiments/minigrid/dependencies/gym-minigrid:/home/developer/transformers/experiments/minigrid/dependencies/torch-ac \
    -v $(pwd)/../:/home/developer/transformers \
    -v /home/k_miyazawa/sshfs_dir:/home/developer/data \
    -p 8888-8899:8888 \
    -p 6006-6010:6006 \
    -p 5900-5902:5900 \
    -e SCREEN_RESOLUTION=1280x1024 \
    -e VNC_PASSWORD=passpass \
    -e USE_VNC=false \
    --name $(id -u -n)-transformers \
    ${IMAGE_NAME}
