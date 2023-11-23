_base_ = [
    '../../_base_/datasets/unconditional_imgs_128x128.py',
    '../../_base_/models/wgangp/wgangp_base.py',
    '../../_base_/default_runtime.py'
]

data = dict(
    samples_per_gpu=32,
    workers_per_gpu=8,
    train=dict(imgs_root='/nfs/home/daft69828/mmgeneration/data/modified_dataset/white_point'),
    val=dict(imgs_root='/nfs/home/daft69828/mmgeneration/data/modified_dataset/white_point')
)

checkpoint_config = dict(interval=20, by_epoch=True)
log_config = dict(interval=100, hooks=[dict(type='TextLoggerHook')])

custom_hooks = [
    dict(
        type='VisualizeUnconditionalSamples',
        output_dir='training_samples',
        interval=5000)
]

# evaluation = dict(
#     type='GenerativeEvalHook',
#     interval=5000,
#     metrics=dict(
#         type='FID',
#         num_images=2117,
#         inception_pkl='/nfs/home/daft69828/mmgeneration/work_dirs/inception_pkl/IS_black.pkl',
#         inception_args=dict(
#             type='StyleGAN',
#             inception_path='/nfs/home/daft69828/mmgeneration/inception-2015-12-05.pt'),
#         bgr2rgb=True))

lr_config = None
total_iters = 16000000

metrics = dict(
    ms_ssim10k=dict(type='MS_SSIM', num_images=10000),
    swd16k=dict(type='SWD', num_images=16384, image_shape=(3, 128, 128)))
