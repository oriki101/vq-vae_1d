import kfp.dsl as dsl
import kfp.gcp as gcp
import kfp.compiler as compiler
from kfp import components
from kubernetes import client as k8s_client

def git_clone_op(repo_url, password):
    volume_op = dsl.VolumeOp(
        name='create pipeline volume',
        resource_name='pipeline-pvc',
        modes=['ReadWriteMany'],
        size="1Gi"
    )

    image = 'alpine/git:latest'

    commands = [
        f"git clone {repo_url} /workspace/vq-vae_1d",
        # f"cd /vq-vae_1d",
        # "apk --update add sshpass",
        # # "apt-get -y install sshpass",
        # "mkdir ~/.ssh",
        # # 'echo "Host *" >> ~/.ssh/config',
        # # 'echo "StrictHostKeyChecking no" >> ~/.ssh/config',
        # "sshpass -p 'House101' scp -r -o StrictHostKeyChecking=no k_ohmori@192.168.11.107:~/hdd/ohmori/data .",
        # "rm -rf /var/lib/apt/lists/*",
        # "rm /var/cache/apk/*",
        "pwd",
        'ls',
    ]

    # Mount Git repository in /workspace
    op = dsl.ContainerOp(
        name='git clone',
        image=image,
        command=['sh'],
        arguments=['-c', ' && '.join(commands)],
        container_kwargs={'image_pull_policy': 'IfNotPresent'},
        pvolumes={'/workspace': volume_op.volume}
    )

    return op

@dsl.pipeline(
    name='vq-vae pipeline',
    description='A pipeline to train vq-vae'
)
def vae_pipeline(data_type='gaze', f_name='vq_vae', num_hiddens=128, num_residual_hiddens=32,
                num_residual_layers=2, embedding_dim=8, num_embeddings=128,
                commitment_cost=0.25, decay=0.99, epoch=500,
                lr = 1e-3, batch_size=32,
                repo_url='http://OHMORI:house101@zaku.sys.es.osaka-u.ac.jp:10080/OHMORI/vq-vae_1d.git'):

    git_clone = git_clone_op(repo_url=repo_url, password='house101')

    # commands = [
    #     "ls",
    #     'cd /home/jovyan/vq-vae_1d/data',
    #     "ls",
    #     "cd /home/jovyan/vq-vae_1d",
    #     "pwd",
    #     "ls"
    # ]

    train=dsl.ContainerOp(
        name='VQ-VAE',
        image='zaku.sys.es.osaka-u.ac.jp:10081/ohmori/vq-vae_1d:v0.0.0',
        # command=['sh'],
        # arguments=['-c', ' && '.join(commands)],
        command=['python3', 'train.py'],
        arguments=[
            '--data_type', data_type,
            '--f_name', f_name,
            '--num_hiddens', num_hiddens,
            '--num_residual_hiddens', num_residual_hiddens,
            '--num_residual_layers', num_residual_layers,
            '--embedding_dim', embedding_dim,
            '--num_embeddings', num_embeddings,
            '--commitment_cost', commitment_cost,
            '--decay', decay,
            '--epoch', epoch,
            '--lr', lr,
            '--batch_size', batch_size
        ],
        pvolumes={'/home/jovyan': git_clone.pvolume},
        file_outputs={
            'MLPipeline UI metadata': '/mlpipeline-ui-metadata.json'
        }
    )

    train.add_pvolumes({'/workspace': git_clone.pvolume})
    # train.add_volume_mount(volume_mount=k8s_client.V1VolumeMount(mount_path='/home/jovyan',name=git_clone.name))

    nfs_volume_source = k8s_client.V1NFSVolumeSource(server="192.168.11.107", path="/data/share/data")
    train.add_volume(k8s_client.V1Volume(name='data', nfs=nfs_volume_source))
    train.add_volume_mount(k8s_client.V1VolumeMount(mount_path='/home/jovyan/vq-vae_1d/data',name='data'))

    nfs_volume_source = k8s_client.V1NFSVolumeSource(server="192.168.11.107", path="/data/share/pth")
    train.add_volume(k8s_client.V1Volume(name='pth', nfs=nfs_volume_source))
    train.add_volume_mount(k8s_client.V1VolumeMount(mount_path='/home/jovyan/vq-vae_1d/script/pth',name='pth'))

    nfs_volume_source = k8s_client.V1NFSVolumeSource(server="192.168.11.107", path="/data/share/logs")
    train.add_volume(k8s_client.V1Volume(name='logs', nfs=nfs_volume_source))
    train.add_volume_mount(k8s_client.V1VolumeMount(mount_path='/home/jovyan/vq-vae_1d/script/logs',name='logs'))

    train.container.working_dir = "/home/jovyan/vq-vae_1d/script"


if __name__ == '__main__':
    compiler.Compiler().compile(vae_pipeline, 'vq-vae-pipeline.zip')
