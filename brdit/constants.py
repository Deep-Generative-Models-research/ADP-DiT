import torch
from brdit.diffusion.gaussian_diffusion import GaussianDiffusion

# =======================================================

NOISE_SCHEDULES = {
    "linear",
    "scaled_linear",
    "squaredcos_cap_v2",
}

PREDICT_TYPE = {
    "epsilon",
    "sample",
    "v_prediction",
}

# =======================================================

NEGATIVE_PROMPT = 'Bad eyes, terrible face, disfigured, bad art, deformed, extra limbs, blurry colors, blurry, repetitive, sickly, defective'

# =======================================================
TRT_MAX_BATCH_SIZE = 1
TRT_MAX_WIDTH = 1280
TRT_MAX_HEIGHT = 1280

# =======================================================
# Constants about models
# =======================================================

VAE_EMA_PATH = "ckpts/t2i/sdxl-vae-fp16-fix"
# bert 모델
# TOKENIZER = "ckpts/t2i/tokenizer"
# TEXT_ENCODER = 'ckpts/t2i/clip_text_encoder'
# clip 모델
TOKENIZER = "ckpts/dialoggen/openai/clip-vit-large-patch14-336"
TEXT_ENCODER = 'ckpts/dialoggen/openai/clip-vit-large-patch14-336'

SAMPLER_FACTORY = {
    'ddpm': {
        'scheduler': 'DDPMScheduler',
        'name': 'DDPM',
        'kwargs': {
            'steps_offset': 1,
            'clip_sample': False,
            'clip_sample_range': 1.0,
            'beta_schedule': 'scaled_linear',
            'beta_start': 0.00085,
            'beta_end': 0.02,
            'prediction_type': 'v_prediction',
        }
    },
    'ddim': {
        'scheduler': 'DDIMScheduler',
        'name': 'DDIM',
        'kwargs': {
            'steps_offset': 1,
            'clip_sample': False,
            'clip_sample_range': 1.0,
            'beta_schedule': 'scaled_linear',
            'beta_start': 0.00085,
            'beta_end': 0.02,
            'prediction_type': 'v_prediction',
        }
    },
    'dpmms': {
        'scheduler': 'DPMSolverMultistepScheduler',
        'name': 'DPMMS',
        'kwargs': {
            'beta_schedule': 'scaled_linear',
            'beta_start': 0.00085,
            'beta_end': 0.02,
            'prediction_type': 'v_prediction',
            'trained_betas': None,
            'solver_order': 2,
            'algorithm_type': 'dpmsolver++',
        }
    },
   # Corrected samplers with properly initialized GaussianDiffusion
    'dpmpp_2m_sde_exp': {
        'scheduler': GaussianDiffusion.dpmpp_2m_sde_exponential_sample,
        'name': 'DPM++ 2M SDE Exp',
        'kwargs': {
            'beta_schedule': 'scaled_linear',
            'beta_start': 0.00085,
            'beta_end': 0.02,
            'prediction_type': 'v_prediction',
            'clip_denoised': True,
            'algorithm_type': 'dpmpp_2m_sde_exponential',
        }
    },
    'dpmpp_3m_sde_karras': {
        'scheduler': GaussianDiffusion.dpmpp_3m_sde_karras_sample,
        'name': 'DPM++ 3M SDE Karras',
        'kwargs': {
            'beta_schedule': 'scaled_linear',
            'beta_start': 0.00085,
            'beta_end': 0.02,
            'prediction_type': 'v_prediction',
            'clip_denoised': True,
            'algorithm_type': 'dpmpp_3m_sde_karras',
        }
    },
    'dpmpp_3m_sde_exp': {
        'scheduler': GaussianDiffusion.dpmpp_3m_sde_exp_sample,
        'name': 'DPM++ 3M SDE Exp',
        'kwargs': {
            'beta_schedule': 'scaled_linear',
            'beta_start': 0.00085,
            'beta_end': 0.02,
            'prediction_type': 'v_prediction',
            'clip_denoised': True,
            'algorithm_type': 'dpmpp_3m_sde_exponential',
        }
    },
}