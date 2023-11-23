"""Note that this config is just for testing."""

_base_ = [
    '../_base_/datasets/lsun_stylegan.py',
    '../_base_/models/stylegan/stylegan2_base.py',
    '../_base_/default_runtime.py'
]

model = dict(generator=dict(out_size=128), discriminator=dict(in_size=128))

data = dict(
    samples_per_gpu=32,
    workers_per_gpu=8,
    train=dict(imgs_root='/nfs/home/daft69828/mmgeneration/data/dataset1/dark_color_point'),
    val=dict(imgs_root='/nfs/home/daft69828/mmgeneration/data/dataset1/dark_color_point')
)

ema_half_life = 10.  # G_smoothing_kimg

custom_hooks = [
    dict(
        type='VisualizeUnconditionalSamples',
        output_dir='training_samples',
        interval=10000)
#    dict(
#        type='ExponentialMovingAverageHook',
#        module_keys=('generator_ema', ),
#        interval=1,
#        interp_cfg=dict(momentum=0.5**(32. / (ema_half_life * 1000.))),
#        priority='VERY_HIGH')
]

checkpoint_config = dict(interval=20, by_epoch=True)
lr_config = None

# log_config = dict(
#    interval=100,
#    hooks=[
#        dict(type='TextLoggerHook'),
#        # dict(type='TensorboardLoggerHook'),
#    ])

evaluation = dict(
    type='GenerativeEvalHook',
    interval=10000,
    metrics=dict(
        type='FID',
        num_images=8886,
        inception_pkl='/nfs/home/daft69828/mmgeneration/work_dirs/inception_pkl/IS_black.pkl',
        inception_args=dict(
            type='StyleGAN',
            inception_path='/nfs/home/daft69828/mmgeneration/inception-2015-12-05.pt'),
        bgr2rgb=True))

total_iters = 8000002  # need to modify

#metrics = dict(
#    fid50k=dict(
#        type='FID', num_images=50000, inception_pkl=None, bgr2rgb=True),
#    pr50k3=dict(type='PR', num_images=50000, k=3),
#    ppl_wend=dict(type='PPL', space='W', sampling='end', num_images=50000))
