_base_ = [
    '../../_base_/models/dcgan/dcgan_128x128.py',
    '../../_base_/datasets/unconditional_imgs_128x128.py',
    '../../_base_/default_runtime.py'
]

# define dataset
# you must set `samples_per_gpu` and `imgs_root`
data = dict(
    samples_per_gpu=64,
    workers_per_gpu=8,
    train=dict(imgs_root='/nfs/home/daft69828/mmgeneration/data/modified_dataset/light_ring'),
    val=dict(imgs_root='/nfs/home/daft69828/mmgeneration/data/modified_dataset/light_ring')
)

# adjust running config
lr_config = None
checkpoint_config = dict(interval=20, by_epoch=True, max_keep_ckpts=5000)
train_cfg = dict(disc_steps=1)
custom_hooks = [
    dict(
        type='VisualizeUnconditionalSamples',
        output_dir='training_samples',
        interval=1000)
]

evaluation = dict(
    type='GenerativeEvalHook',
    interval=1000,
    metrics=dict(
        type='FID',
        num_images=10000,
        inception_pkl='/nfs/home/daft69828/mmgeneration/work_dirs/inception_pkl/IS_stylegan_ring.pkl',
        inception_args=dict(
            type='StyleGAN',
            inception_path='/nfs/home/daft69828/mmgeneration/inception-2015-12-05.pt'),
        bgr2rgb=True))

total_iters = 20000000

# use ddp wrapper for faster training
use_ddp_wrapper = True
find_unused_parameters = False

runner = dict(
    type='DynamicIterBasedRunner',
    is_dynamic_ddp=False,  # Note that this flag should be False.
    pass_training_status=True)

optimizer = dict(
    generator=dict(type='Adam', lr=0.0001, betas=(0.5, 0.999)),
    discriminator=dict(type='Adam', lr=0.000025, betas=(0.5, 0.999)))
