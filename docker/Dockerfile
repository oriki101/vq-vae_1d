FROM nvcr.io/nvidia/pytorch:19.10-py3

RUN DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Tokyo
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone 

# https://qiita.com/yohm/items/047b2e68d008ebb0f001
RUN apt-get update && apt-get -y \
    install gosu golang-go \
&& apt-get clean \
&& rm -rf /var/lib/apt/lists/*

RUN apt-get update -y && apt-get -y upgrade && apt-get install -y \
    wget curl git apt-transport-https icewm xvfb x11vnc tmux python-opengl imagemagick htop \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*

RUN pip install wrapt --upgrade --ignore-installed
RUN pip install gym roboschool pybullet pyopengl pylint natsort
RUN conda install -y pyqt

# Since uid and gid will change at entrypoint, anything can be used
ARG USER_ID=1000
ARG GROUP_ID=1000
ENV USER_NAME=developer
RUN groupadd -g ${GROUP_ID} ${USER_NAME} && \
    useradd -d /home/${USER_NAME} -m -s /bin/bash -u ${USER_ID} -g ${GROUP_ID} ${USER_NAME}
WORKDIR /home/${USER_NAME}

USER ${USER_NAME}
ENV HOME /home/${USER_NAME}

# https://www.hanselman.com/blog/HowToMakeAPrettyPromptInWindowsTerminalWithPowerlineNerdFontsCascadiaCodeWSLAndOhmyposh.aspx
RUN go get -u github.com/justjanne/powerline-go

RUN echo $'\n\
GOPATH=$HOME/go \n\
function _update_ps1() { \n\
    PS1="$($GOPATH/bin/powerline-go -error $?)" \n\
} \n\
if [ "$TERM" != "linux" ] && [ -f "$GOPATH/bin/powerline-go" ]; then \n\
    PROMPT_COMMAND="_update_ps1; $PROMPT_COMMAND" \n\
fi \n\
' >> $HOME/.bashrc

USER root
COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["jupyter", "lab", "--no-browser", "--ip=0.0.0.0","--NotebookApp.token=''"]

## add time news roman
# matplotlibでTimes New Romanが意図せずボールド体になってしまうときの対処法
# https://qiita.com/Miyabi1456/items/ef7a83c239cf0d9478f9
# path: /opt/conda/lib/python3.6/site-packages/matplotlib/font_manger.py
# matplotlibでTimes New Romanを使うためのTips
# http://kenbo.hatenablog.com/entry/2018/11/28/111639
