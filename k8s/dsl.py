import kfp.dsl as dsl
import kfp.gcp as gcp
import kfp.compiler as compiler

def git_clone_op(repo_url, password):
    volume_op = dsl.VolumeOp(
        name='create pipeline volume',
        resource_name='pipeline-pvc',
        modes=['ReadWriteMany'],
        size="30Gi"
    )

    image = 'alphine/git:latest'

    commands = [
        f"git clone {repo_url}",
        f"cd vq-vae_1d",
        "apt-get update",
        "apt-get -y install sshpass",
        "sshpass -p 'house101' ssh -f k_ohmori@192.168.11.107:~/hdd/ohmori/data ."
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
def vae_pipeline(f_name='vq_vae', num_hiddens=128, num_residual_hiddens=32,
                num_residual_layers=2, embedding_dim=8, num_embeddings=128,
                commitment_cost=0.25, decay=0.99, epoch=500,
                lr = 1e-3, batch_size=32,
                repo_url='http://OHMORI:house101@zaku.sys.es.osaka-u.ac.jp:10080/OHMORI/vq-vae_1d.git'):

    git_clone = git_clone_op(repo_url=repo_url, password='house101')

    train=dsl.ContainerOp(
        name='VQ-VAE',
        image='zaku.sys.es.osaka-u.ac.jp:10081/OHMORI/vq-vae_1d:ohmori',
        command=['python3', '~/workspace/vq-vae_1d/script/train.py'],
        arguments=[
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
        pvolumes={'/workspace': git_clone.pvolume},
        file_outputs={
            'MLPipeline UI metadata': '/mlpipeline-ui-metadata.json'
        }
    )

if __name__ == '__main__':
    compiler.Compiler().compile(vae_pipeline, 'vq-vae-pipeline.zip')