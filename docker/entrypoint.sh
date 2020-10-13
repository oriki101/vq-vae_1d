#!/bin/bash

# inside docker script

if "${USE_VNC}"; then
    export DISPLAY=${DISPLAY}

    Xvfb ${DISPLAY} -screen 0 $SCREEN_RESOLUTION &
    sleep 1

    x11vnc -display ${DISPLAY} -passwd $VNC_PASSWORD -forever &
    sleep 1

    icewm-session &
    sleep 1
fi

USER_ID=${LOCAL_UID:-9001}
GROUP_ID=${LOCAL_GID:-9001}

usermod -u $USER_ID -o -d /home/developer -m developer
groupmod -g $GROUP_ID developer

exec /usr/sbin/gosu developer "$@"