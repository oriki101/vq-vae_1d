import kfp.dsl as dsl
import kfp.gcp as gcp
import kfp.compiler as compiler
from kfp import components
from kubernetes import client as k8s_client

def git_clone_op(repo_url, volume_op):
    #Volume for storing Git repository (PV)
    # volume_op = dsl.VolumeOp(
    #     name='create pipeline volume',
    #     resource_name='pipeline-pvc',
    #     modes=['ReadWriteMany'],
    #     size="1Gi"
    # )

    image = 'alpine/git:latest'

    commands = [
        f"git clone {repo_url} /workspace/vq-vae_1d",
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
def vae_pipeline(data_type='gaze', f_name='vq_vae', num_hiddens=32, num_residual_hiddens=32,
                num_residual_layers=2, embedding_dim=8, num_embeddings=128,
                commitment_cost=0.25, decay=0.99, epoch=500,
                lr = 1e-3, batch_size=32,
                repo_url='http://OHMORI:house101@zaku.sys.es.osaka-u.ac.jp:10080/OHMORI/vq-vae_1d.git'):
    #secret setting for pipeline
    dsl.get_pipeline_conf().set_image_pull_secrets([k8s_client.V1LocalObjectReference(name="regcred")])

    # Volume for storing Git repository (PV)
    volume_op = dsl.VolumeOp(
        name='create pipeline volume',
        resource_name='pipeline-pvc',
        modes=['ReadWriteMany'],
        size="1Gi"
    )

    git_clone = git_clone_op(repo_url=repo_url, volume_op=volume_op)

    # commands = [
    #     "ls",
    #     # "pwd",
    #     # 'cd /home/jovyan/vq-vae_1d/data',
    #     # "ls",
    #     # "cd /home/jovyan/vq-vae_1d",
    #     # "ls",
    #     "pwd"
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
        # output Tensorboard, 上手く動作しない
        file_outputs={
            'MLPipeline UI metadata': '/mlpipeline-ui-metadata.json'
        }
    ).set_gpu_limit(gpu=1, vendor='nvidia')

    # train.add_pvolumes({'/workspace': git_clone.pvolume})
    # train.add_volume_mount(volume_mount=k8s_client.V1VolumeMount(mount_path='/home/jovyan',name=git_clone.name))

    #set nfs information and mount nfs on Container(named train)
    nfs_volume_source = k8s_client.V1NFSVolumeSource(server="192.168.11.107", path="/data/share/data")
    train.add_volume(k8s_client.V1Volume(name='data', nfs=nfs_volume_source))
    train.add_volume_mount(k8s_client.V1VolumeMount(mount_path='/home/jovyan/vq-vae_1d/data',name='data'))

    nfs_volume_source = k8s_client.V1NFSVolumeSource(server="192.168.11.107", path="/data/share/pth")
    train.add_volume(k8s_client.V1Volume(name='pth', nfs=nfs_volume_source))
    train.add_volume_mount(k8s_client.V1VolumeMount(mount_path='/home/jovyan/vq-vae_1d/script/pth',name='pth'))

    nfs_volume_source = k8s_client.V1NFSVolumeSource(server="192.168.11.107", path="/data/share/logs")
    train.add_volume(k8s_client.V1Volume(name='logs', nfs=nfs_volume_source))
    train.add_volume_mount(k8s_client.V1VolumeMount(mount_path='/home/jovyan/vq-vae_1d/script/logs',name='logs'))

    train.add_volume(k8s_client.V1Volume(name="dshm", empty_dir=k8s_client.V1EmptyDirVolumeSource(medium="Memory")))
    train.add_volume_mount(k8s_client.V1VolumeMount(mount_path='/dev/shm',name='dshm'))

    #set workdirectory
    train.container.working_dir = "/home/jovyan/vq-vae_1d/script"

    #delete PV?
    # volume_op.delete().after(train)
    # dsl.ResourceOp(
    #     name='delete-volume',
    #     # k8s_resource=volume_op.k8s_resource,
    #     k8s_resource=train.pvolume.k8s_resource,
    #     action='delete'
    # ).after(train)


if __name__ == '__main__':
    compiler.Compiler().compile(vae_pipeline, 'vq-vae-pipeline.zip')
