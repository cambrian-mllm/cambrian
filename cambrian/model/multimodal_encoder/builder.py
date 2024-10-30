import os
import copy

from ezcolorlog import root_logger as logger

from .clip_encoder import ClipVisionTower
from .dfn_clip_encoder import DfnClipVisionTower
from .siglip_encoder import SiglipVisionTower
from .eva_clip_encoder import EvaClipVisionTower
from .clip_convnext_encoder import CLIPConvNextTower
from .dino_encoder import DinoVisionTower
from .ijepa_encoder import IJepaVisionTower
from .mae_encoder import MAEVisionTower
from .midas_encoder import MiDaSVisionTower
from .moco_encoder import MoCoVisionTower
from .hybrid_encoder import HybridVisionTower
from .supervised_vit_encoder import SupervisedViT_VisionTower
from .sam_encoder import SAMVisionTower
from .diffusion_encoder import DiffusionVisionTower
from .maws_encoder import MawsVisionTower
from concurrent.futures import ThreadPoolExecutor

def build_vision_tower(vision_tower_cfg, **kwargs):
    vision_tower = getattr(vision_tower_cfg, 'mm_vision_tower', getattr(vision_tower_cfg, 'vision_tower', None))

    if vision_tower is None or not isinstance(vision_tower, str):
        raise ValueError(f'Vision Tower is not specified in the config: {vision_tower_cfg}')
    if "maws" in vision_tower.lower():
        logger.info(f"Loading **MAWS** Vision Tower: {vision_tower}")
        return MawsVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
        
    # CLIP-based Vision Towers
    if "openai/clip" in vision_tower.lower():
        logger.info(f"Loading **OpenAI CLIP** Vision Tower: {vision_tower}")
        return ClipVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
    if "apple/dfn" in vision_tower.lower():
        logger.info(f"Loading **Apple DFN CLIP** Vision Tower: {vision_tower}")
        return DfnClipVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
    if "siglip" in vision_tower.lower():
        logger.info(f"Loading **SigLIP CLIP** Vision Tower: {vision_tower}")
        return SiglipVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
    if "eva/clip" in vision_tower.lower():
        logger.info(f"Loading **EVA CLIP** Vision Tower: {vision_tower}")
        return EvaClipVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
    if "clip-convnext" in vision_tower.lower():
        logger.info(f"Loading **ConvNeXt CLIP** Vision Tower: {vision_tower}")
        return CLIPConvNextTower(vision_tower, args=vision_tower_cfg, **kwargs)

    # SSL-based Vision Towers
    if "dinov2" in vision_tower.lower():
        logger.info(f"Loading **DINO Vision Tower: {vision_tower}")
        return DinoVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
    if "mae" in vision_tower.lower():
        logger.info(f"Loading **MAE** Vision Tower: {vision_tower}")
        return MAEVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
    if "moco" in vision_tower.lower():
        logger.info(f"Loading **MoCo** Vision Tower: {vision_tower}")
        return MoCoVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
    if "ijepa" in vision_tower.lower():
        logger.info(f"Loading **IJepa** Vision Tower: {vision_tower}")
        return IJepaVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)

    # Supervised Vision Towers
    if "supervised-vit" in vision_tower.lower():
        logger.info(f"Loading **Supervised** Vision Tower: {vision_tower}")
        return SupervisedViT_VisionTower(vision_tower, args=vision_tower_cfg, **kwargs)

    # Other Vision Towers
    if "hybridmodel" in vision_tower.lower():
        logger.info(f"Loading **Hybrid** Vision Tower: {vision_tower}")
        return HybridVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
    if "diffusion" in vision_tower.lower():
        logger.info(f"Loading **Diffusion CLIP** Vision Tower: {vision_tower}")
        return DiffusionVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
    if "midas" in vision_tower.lower():
        logger.info(f"Loading **MiDaS** Vision Tower: {vision_tower}")
        return MiDaSVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
    if "sam" in vision_tower.lower():
        logger.info(f"Loading **SAM Vision Tower: {vision_tower}")
        return SAMVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)

    raise ValueError(f'Unknown vision tower: {vision_tower}')



def build_vision_tower_aux_list(vision_tower_cfg, **kwargs):
    vision_tower_aux_name_list = getattr(vision_tower_cfg, 'mm_vision_tower_aux_list', getattr(vision_tower_cfg, 'vision_tower_aux_list', None))
    vision_tower_aux_token_len_list = getattr(vision_tower_cfg, 'mm_vision_tower_aux_token_len_list', getattr(vision_tower_cfg, 'vision_tower_aux_token_len_list', None))
    def worker(vision_tower_aux_name,vision_tower_aux_token_len):
        config = copy.deepcopy(vision_tower_cfg)
        vision_tower = None
        vision_tower_aux_name += "-interp{}".format(vision_tower_aux_token_len)
        if "maws" in vision_tower_aux_name.lower():
            logger.info(f"Loading **MAWS** Vision Tower: {vision_tower_aux_name}")
            vision_tower=MawsVisionTower(vision_tower_aux_name, args=config, **kwargs)

        # CLIP-based Vision Towers
        elif "openai/clip" in vision_tower_aux_name.lower():
            logger.info(f"Loading **OpenAI CLIP** Vision Tower: {vision_tower_aux_name}")
            vision_tower=ClipVisionTower(vision_tower_aux_name, args=config, **kwargs)
        elif "apple/dfn" in vision_tower_aux_name.lower():
            logger.info(f"Loading **Apple DFN CLIP** Vision Tower: {vision_tower_aux_name}")
            vision_tower=DfnClipVisionTower(vision_tower_aux_name, args=config, **kwargs)
        elif "siglip" in vision_tower_aux_name.lower():
            logger.info(f"Loading **SigLIP CLIP** Vision Tower: {vision_tower_aux_name}")
            vision_tower=SiglipVisionTower(vision_tower_aux_name, args=config, **kwargs)
        elif "eva/clip" in vision_tower_aux_name.lower():
            logger.info(f"Loading **EVA CLIP** Vision Tower: {vision_tower_aux_name}")
            vision_tower=EvaClipVisionTower(vision_tower_aux_name, args=config, **kwargs)
        elif "clip-convnext" in vision_tower_aux_name.lower():
            logger.info(f"Loading **ConvNeXt CLIP** Vision Tower: {vision_tower_aux_name}")
            vision_tower=CLIPConvNextTower(vision_tower_aux_name, args=config, **kwargs)

        # SSL-based Vision Towers
        elif "dinov2" in vision_tower_aux_name.lower():
            logger.info(f"Loading **DINO Vision Tower: {vision_tower_aux_name}")
            vision_tower=DinoVisionTower(vision_tower_aux_name, args=config, **kwargs)
        elif "mae" in vision_tower_aux_name.lower():
            logger.info(f"Loading **MAE** Vision Tower: {vision_tower_aux_name}")
            vision_tower=MAEVisionTower(vision_tower_aux_name, args=config, **kwargs)
        elif "moco" in vision_tower_aux_name.lower():
            logger.info(f"Loading **MoCo** Vision Tower: {vision_tower_aux_name}")
            vision_tower=MoCoVisionTower(vision_tower_aux_name, args=config, **kwargs)
        elif "ijepa" in vision_tower_aux_name.lower():
            logger.info(f"Loading **IJepa** Vision Tower: {vision_tower_aux_name}")
            vision_tower=IJepaVisionTower(vision_tower_aux_name, args=config, **kwargs)

        # Supervised Vision Towers
        elif "supervised-vit" in vision_tower_aux_name.lower():
            logger.info(f"Loading **Supervised** Vision Tower: {vision_tower_aux_name}")
            vision_tower=SupervisedViT_VisionTower(vision_tower_aux_name, args=config, **kwargs)

        # Other Vision Towers
        elif "hybridmodel" in vision_tower_aux_name.lower():
            logger.info(f"Loading **Hybrid** Vision Tower: {vision_tower_aux_name}")
            vision_tower=HybridVisionTower(vision_tower_aux_name, args=config, **kwargs)
        elif "diffusion" in vision_tower_aux_name.lower():
            logger.info(f"Loading **Diffusion CLIP** Vision Tower: {vision_tower_aux_name}")
            vision_tower=DiffusionVisionTower(vision_tower_aux_name, args=config, **kwargs)
        elif "midas" in vision_tower_aux_name.lower():
            logger.info(f"Loading **MiDaS** Vision Tower: {vision_tower_aux_name}")
            vision_tower=MiDaSVisionTower(vision_tower_aux_name, args=config, **kwargs)
        elif "sam" in vision_tower_aux_name.lower():
            logger.info(f"Loading **SAM Vision Tower: {vision_tower_aux_name}")
            vision_tower=SAMVisionTower(vision_tower_aux_name, args=config, **kwargs)
        else:
            raise ValueError(f'Unknown vision tower: {vision_tower_aux_name}')
        return vision_tower
    with ThreadPoolExecutor() as executor:
        vision_tower_aux_list = executor.map(worker, vision_tower_aux_name_list,vision_tower_aux_token_len_list)
    return list(vision_tower_aux_list)
