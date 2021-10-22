# vq-vae_1d
![build](https://img.shields.io/badge/build-pass-green.svg?style=flat)
![version](https://img.shields.io/badge/version-v0.01-blue.svg?style=flat)
![platform](https://img.shields.io/badge/platform-linux-lightgrey.svg?style=flat)
![container](https://img.shields.io/badge/container-ready-green.svg?style=flat)
## 1. <a name='Overview'></a>Overview
VQ-VAEで1次元信号を学習するためのリポジトリです。

##  2. <a name='TableofContents'></a>Table of Contents
1. [Overview](#Overview)
2. [Table of Contents](#TableofContents)
3. [Status](#Status)
4. [Quick Start](#QuickStart)
5. [API](#API)
6. [Artifacts](#Artifacts)
7. [Developer Information](#DeveloperInformation)
8. [Citations](#Citations)
9. [License](#License)
    <!-- 10. [Code Structure](#CodeStructure)
    11. [Requirement](#Requirement)
    12. [Dependences](#Dependences)
    13. [Installation](#Installation) -->

##  3. <a name='Status'></a>Status
- E4 wristband, cometa pico, tobii glassesの信号を学習できるよう、それぞれのデータセットを用意

- /vq-vae_1d/utils/*datasets.pyを参考に新たなデータセットクラスを定義すると様々なデータで学習可能

- notebooとスクリプトを通してE4 wristband, cometa pico, tobii glassesの信号を学習可能

##  4. <a name='QuickStart'></a>Quick Start
### 4.1 学習環境の設定
```bash
$git clone ****
$cd vq-vae_1d/docker
$./run.sh
#open http://<ip address>:8888
#open the notebook
```

### 4.2 学習データの設定
```bash
/vq-vae_1d/utils/*datasets.py
```
内部のデータのロードを行うパスを自身で置いた学習データへのパスに変更


##  5. <a name='API'></a>API
cometa picoとtobii glasses(瞳孔径)のスクリプトを利用した学習方法

引数
```bash
'--data_type', type=str, default='gaze', help='学習するデータのタイプを選択　ex: gaze, emg, bvp, gsr, temperature'
'--f_name', type=str, default='vq_vae', help='学習した重みを保存するファイルの選択'
'--num_hiddens', type=int, default=32, help='parameter of vq-vae encoder and decoder'
'--num_residual_hiddens', type=int, default=32, help='parameter of vq-vae'
'--num_residual_layers', type=int, default=2, help='the number of residual layer'
'--embedding_dim', type=int, default=8, help='embedding dimension'
'--num_embeddings', type=int, default=128, help='the number of embeddings'
'--commitment_cost', type=float, default=0.25, help='commitment cost'
'--decay', type=float, default=0.99, help='decay'
'--epoch', type=int, default=500, help='the number of epochs'
'--lr', type=float, default=1e-3, help='learning rate'
'--batch_size', type=int, default=32, help='batch size'
```


利用例
```bash
python /vq-vae_1d/script/train.py --data_type 'gaze' --epoch 500 --lr 0.001 --batch_size 256
```


##  6. <a name='Artifacts'></a>Artifacts

##  7. <a name='DeveloperInformation'></a>Developer Information
- Developer: Kazuaki Ohmori
- Maintainer: Kazuaki Ohmori

##  8. <a name='Citations'></a>Citations

##  9. <a name='License'></a>License
This project is licensed under the MIT license.
