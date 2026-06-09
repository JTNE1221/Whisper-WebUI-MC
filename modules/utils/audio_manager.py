from typing import Optional, Union
import os
import numpy as np
from faster_whisper.audio import decode_audio

from modules.utils.logger import get_logger

logger = get_logger()


def validate_audio(audio: Union[str, np.ndarray, None] = None):
    """Validate audio file and check if it's corrupted"""
    # [修复]: 拦截 None 值，防止 os.path.exists(None) 导致崩溃
    if audio is None:
        return False

    if isinstance(audio, np.ndarray):
        return True

    if not os.path.exists(audio):
        logger.info(f"The file {audio} does not exist. Please check the path.")
        return False

    try:
        # [优化]: 使用 _ 接收返回值，校验完成后立刻释放内存，防止大文件导致内存飙升
        _ = decode_audio(audio)
        return True
    except Exception as e:
        logger.info(f"The file {audio} is not able to open or corrupted. Please check the file. {e}")
        return False
