import ml_collections
from dataclasses import dataclass

@dataclass
class Args:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
#follow t2i_256px_clip_dimr.py
model = Args(
    adapter_in_embed=2048,
    #adapter_in_token=576, #already cut to 77
    channels = 4,
    block_grad_to_lowres = False,
    norm_type = "TDRMSN",
    use_t2i = True,
    clip_dim=768,                                               # 768 for CLIP, 4096 for qisheng-Janus, 2048 for JanusPro1B
    num_clip_token=77,
    gradient_checking=True,
    cfg_indicator=0.1,
    textVAE = Args(
        num_blocks = 11,
        hidden_dim = 1024,  
        hidden_token_length = 256, 
        num_attention_heads = 8,
        dropout_prob = 0.1,
    ),
    stage_configs = [
            Args(
                block_type = "TransformerBlock", 
                dim = 1024,  
                hidden_dim = 2048,  
                num_attention_heads = 16,  
                num_blocks = 65, 
                max_height = 16,
                max_width = 16,
                image_input_ratio = 1,
                input_feature_ratio = 2,
                final_kernel_size = 3,
                dropout_prob = 0,
            ),
            Args(
                block_type = "ConvNeXtBlock", 
                dim = 512,  
                hidden_dim = 1024,  
                kernel_size = 7, 
                num_blocks = 32,  
                max_height = 32,
                max_width = 32,
                image_input_ratio = 1,
                input_feature_ratio = 1,
                final_kernel_size = 3,
                dropout_prob = 0,
            ),
    ],
)


def d(**kwargs):
    """Helper of creating a config dict."""
    return ml_collections.ConfigDict(initial_dictionary=kwargs)


def get_config():
    config = ml_collections.ConfigDict()

    config.seed = 1234                                          # random seed
    config.z_shape = (4, 32, 32)                                # image latent size

    config.autoencoder = d(
        pretrained_path='/storage/v-jinpewang/lab_folder/qisheng_data/assets/stable-diffusion/autoencoder_kl.pth', # path of pretrained VAE CKPT from LDM
        scale_factor=0.23010
    )

    config.pretrained_path = "/storage/v-jinpewang/lab_folder/qisheng_data/t2i_256px_clip_dimr.pth"

    config.train = d(
        n_steps=600000,                                        # total training iterations
        batch_size=6,                                           # overall batch size across ALL gpus, where batch_size_per_gpu == batch_size / number_of_gpus
        mode='cond',
        log_interval=10,
        eval_interval=10000,                                       # iteration interval for visual testing on the specified prompt
        save_interval=20000,                                      # iteration interval for saving checkpoints and testing FID
        n_samples_eval=15,                                       # 从5降到3，减少显存使用
    )

    config.optimizer = d(
        name='adamw',   
        lr=0.00001,                                             # learning rate
        weight_decay=0.03,
        betas=(0.9, 0.9),
    )

    config.lr_scheduler = d(
        name='customized',
        warmup_steps=5000                                       # warmup steps
    )

    global model
    config.nnet = d(
        name='dimr',
        model_args=model,
    )
    config.loss_coeffs = [1/4, 1]                          # weight on loss, only needed for DiMR. Here, loss = 1/4 * loss_block1 + 1/2 * loss_block2 + 1 * loss_block3
    
    config.dataset = d(
        name='textimage_features',                               # dataset name
        resolution=256,                                         # dataset resolution
        llm='t5',                                               # language model to generate language embedding (t5 means Janus-Pro-7B/1B)
        train_feature_dir='/storage/v-jinpewang/lab_folder/junchao/crossflow_data/test/vis_data/train_features_2D',  # training set feature directory
        train_img_path='/storage/v-jinpewang/lab_folder/junchao/data/image_eidt_dataset/processed_data_wo_textbox/addtion/ultraedit/output',     # training images path
        val_feature_dir='/storage/v-jinpewang/lab_folder/junchao/crossflow_data/test/vis_data/test_features',        # validation set feature directory
        run_vis_feature_dir='/storage/v-jinpewang/lab_folder/junchao/crossflow_data/test/vis_data/run_vis',     # run_vis feature directory
        cfg=False
    )

    config.sample = d(
        sample_steps=50,                                        # sample steps duing inference/testing
        n_samples=500,                                        # number of samples for testing (during training, we sample 10K images, which is hardcoded in the training script)
        mini_batch_size=2,                                     # batch size for testing (i.e., the number of images generated per GPU)
        cfg=False,
        scale=7,                                                # cfg scale
        path='/storage/v-jinpewang/lab_folder/junchao/Crossflow_training/Crossflow_2d/sample/samplesave_wo_textbox'
    )

    return config

