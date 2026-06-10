from typing import Optional, Union, List, Dict
import numpy as np
import torchaudio
import soundfile as sf
import os
import torch
import gc
import gradio as gr
from datetime import datetime
import traceback
from pathlib import Path  # 用于扫描模型文件

# --- 修改导入路径，使用相对路径或绝对路径 ---
# 假设 paths.py 已经被修正
from modules.utils.paths import DEFAULT_PARAMETERS_CONFIG_PATH, UVR_MODELS_DIR, UVR_OUTPUT_DIR
from modules.utils.files_manager import load_yaml, save_yaml, is_video
from modules.diarize.audio_loader import load_audio
from modules.utils.logger import get_logger

logger = get_logger()

try:
    from uvr.models import MDX, Demucs, VrNetwork, MDXC
except Exception as e:
    logger.warning(
        "Failed to import uvr. BGM separation feature will not work. "
        "Please open an issue on GitHub if you encounter this error. "
        f"Error: {type(e).__name__}: {traceback.format_exc()}"
    )

class MusicSeparator:
    def __init__(self, model_dir: Optional[str] = UVR_MODELS_DIR, output_dir: Optional[str] = UVR_OUTPUT_DIR):
        self.model = None
        self.device = self.get_device()
        self.available_devices = ["cpu", "cuda", "xpu", "mps"]
        self.model_dir = model_dir
        self.output_dir = output_dir
        
        instrumental_output_dir = os.path.join(self.output_dir, "instrumental")
        vocals_output_dir = os.path.join(self.output_dir, "vocals")
        os.makedirs(instrumental_output_dir, exist_ok=True)
        os.makedirs(vocals_output_dir, exist_ok=True)
        
        self.audio_info = None
        
        # --- 【核心功能扩展】动态扫描所有 UVR 模型 ---
        self.available_models = self._scan_uvr_models()
        
        if not self.available_models:
            # 如果没扫描到模型，使用默认的（作为保底）
            self.available_models = ["UVR-MDX-NET-Inst_HQ_4", "UVR-MDX-NET-Inst_3"]
            logger.warning("No UVR model files found in the directory. Using default model names as fallback.")
        
        self.default_model = self.available_models[0]
        self.current_model_size = self.default_model
        
        self.model_config = {
            "segment": 256,
            "split": True
        }
    
    def _scan_uvr_models(self) -> List[str]:
        """
        扫描模型目录下的所有 .onnx 和 .pth 模型文件
        """
        model_dir = Path(self.model_dir)
        if not model_dir.exists():
            logger.error(f"Model directory {self.model_dir} does not exist.")
            return []
        
        # 支持的后缀
        suffixes = ('.onnx', '.pth', '.ckpt', '.pt')
        model_files = []
        
        for file in model_dir.iterdir():
            if file.suffix.lower() in suffixes and file.is_file():
                model_files.append(file.stem) # 只添加文件名
        
        if model_files:
            logger.info(f"Successfully scanned {len(model_files)} models: {model_files}")
        else:
            logger.warning(f"No model files found in {self.model_dir}. Please check the directory.")
            
        return sorted(model_files) # 排序输出

    
    def update_model(self, model_name: str = None, device: Optional[str] = None, segment_size: int = 256):
        """
        Update model with dynamic config loading (Sidecar YAML/JSON) and 
        memory injection to bypass uvr library constraints.
        """
        import os
        import json
        import hashlib
        import traceback
        
        if model_name is None:
            model_name = self.default_model
            
        if device is None:
            device = self.device
            
        self.device = device
        self.model_config = {
            "segment": segment_size,
            "split": True
        }
        
        # 1. 检查并获取文件路径
        model_path = os.path.join(self.model_dir, model_name)
        if not os.path.exists(model_path):
            logger.error(f"Model file not found: {model_path}")
            model_name = self.default_model
            model_path = os.path.join(self.model_dir, model_name)
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Default model also not found: {model_path}")
        
        # 2. 清理文件名后缀
        clean_model_name = model_name
        for ext in ['.onnx', '.pth', '.ckpt', '.pt']:
            if model_name.lower().endswith(ext):
                clean_model_name = model_name[:-len(ext)]
                break
        
        # 3. 动态读取本地配套配置文件 (Sidecar Config)
        native_config = {}
        yaml_path = os.path.join(self.model_dir, f"{clean_model_name}.yaml")
        json_path = os.path.join(self.model_dir, f"{clean_model_name}.json")
        
        if os.path.exists(yaml_path):
            logger.info(f"Found sidecar YAML config for {clean_model_name}")
            try:
                native_config = load_yaml(yaml_path) 
            except Exception as e:
                logger.warning(f"Failed to parse YAML config: {e}")
        elif os.path.exists(json_path):
            logger.info(f"Found sidecar JSON config for {clean_model_name}")
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    native_config = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to parse JSON config: {e}")
        else:
            logger.info(f"No sidecar config found for {clean_model_name}. Using fallback generic parameters.")
            # 兜底配置：绝大多数 MDX 模型的通用安全参数
            native_config = {
                "dim_f": 3072, "dim_t": 256, "n_fft": 6144, 
                "hop": 1024, "overlap": 0.25, "compensate": 1.035,
                "primary_stem": "Instrumental"
            }

        # 合并最终配置
        combined_metadata = {**native_config, **self.model_config}

        # 4. 加载模型与内存注入破解
        try:
            if self.model is not None:
                self.offload()
                
            logger.info(f"Loading UVR model: {model_name} on {self.device}")
            
            # ==========================================
            # 终极魔法：破解 uvr 库的双重限制
            # ==========================================
            import uvr.models
            from uvr.models_dir.mdx import mdx_interface as mdx_api
            
            # 破解关卡 1：强行注册模型名字 (解决 models.json KeyError)
            if hasattr(uvr.models, 'models_json'):
                if 'mdx' not in uvr.models.models_json:
                    uvr.models.models_json['mdx'] = {}
                if clean_model_name not in uvr.models.models_json['mdx']:
                    logger.info(f"Injecting name '{clean_model_name}' into internal models_json registry.")
                    uvr.models.models_json['mdx'][clean_model_name] = {"model_path": ""}
                    
            # 破解关卡 2：强行注册 MD5 哈希参数 (解决 model_data.json KeyError)
            if hasattr(uvr.models.MDX, 'models_data'):
                # 调用官方库的方法计算本地 onnx 的哈希值
                model_hash = mdx_api.get_model_hash_from_path(model_path)
                
                # 如果这个哈希不在它的字典里，我们将 YAML/兜底 的参数翻译成官方格式并注入内存
                if model_hash not in uvr.models.MDX.models_data:
                    logger.info(f"Injecting MD5 {model_hash} for {clean_model_name} into memory.")
                    uvr.models.MDX.models_data[model_hash] = {
                        "compensate": combined_metadata.get("compensate", 1.035),
                        "mdx_dim_f_set": combined_metadata.get("dim_f", 3072),
                        "mdx_dim_t_set": combined_metadata.get("dim_t", 256),
                        "mdx_n_fft_scale_set": combined_metadata.get("n_fft", 6144),
                        "primary_stem": combined_metadata.get("primary_stem", "Instrumental")
                    }
            # ==========================================
            
            # 使用官方接口初始化模型
            self.model = MDX(
                name=clean_model_name,
                model_dir=self.model_dir,    # <--- 改成传目录，它会自己根据名字去找文件
                other_metadata=combined_metadata,
                device=self.device,
                logger=logger
            )
            self.current_model_size = model_name
            
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            logger.error(traceback.format_exc())
            if self.model:
                del self.model
                self.model = None
            raise

    def separate(self, audio: Union[str, np.ndarray], model_name: str, device: Optional[str] = None, segment_size: int = 256, save_file: bool = False, progress: gr.Progress = gr.Progress()) -> tuple[np.ndarray, np.ndarray, List]:
        """Separate the background music from the audio."""
        if isinstance(audio, str):
            output_filename, ext = os.path.basename(audio), ".wav"
            output_filename, orig_ext = os.path.splitext(output_filename)
            if is_video(audio):
                audio = load_audio(audio)
                sample_rate = 16000
            else:
                self.audio_info = torchaudio.info(audio)
                sample_rate = self.audio_info.sample_rate
        else:
            timestamp = datetime.now().strftime("%m%d%H%M%S")
            output_filename, ext = f"UVR-{timestamp}", ".wav"
            sample_rate = 16000

        model_config = {
            "segment": segment_size,
            "split": True
        }

        # 检查是否需要更新模型
        if (self.model is None or 
            self.current_model_size != model_name or 
            self.model_config != model_config or 
            getattr(self.model, 'sample_rate', None) != sample_rate or 
            self.device != device):
            
            progress(0, desc="Initializing UVR Model..")
            self.update_model(model_name=model_name, device=device, segment_size=segment_size)
            # 如果 update_model 成功，self.model 已经被赋值
            if self.model is None:
                raise RuntimeError("Model initialization failed.")
            self.model.sample_rate = sample_rate

        progress(0, desc="Separating background music from the audio.. (It may take a while)")
        
        try:
            result = self.model(audio)
            instrumental, vocals = result["instrumental"].T, result["vocals"].T
        except Exception as e:
            logger.error(f"Separation failed: {e}")
            raise

        file_paths = []
        if save_file:
            instrumental_output_path = os.path.join(self.output_dir, "instrumental", f"{output_filename}-instrumental{ext}")
            vocals_output_path = os.path.join(self.output_dir, "vocals", f"{output_filename}-vocals{ext}")
            
            # 确保输出目录存在
            os.makedirs(os.path.dirname(instrumental_output_path), exist_ok=True)
            os.makedirs(os.path.dirname(vocals_output_path), exist_ok=True)
            
            sf.write(instrumental_output_path, instrumental, sample_rate, format="WAV")
            sf.write(vocals_output_path, vocals, sample_rate, format="WAV")
            file_paths += [instrumental_output_path, vocals_output_path]

        return instrumental, vocals, file_paths

    def separate_files(self, files: List, model_name: str, device: Optional[str] = None, segment_size: int = 256, save_file: bool = True, progress: gr.Progress = gr.Progress()) -> List[str]:
        """Separate the background music from the audio files."""
        self.cache_parameters(model_size=model_name, segment_size=segment_size)
        
        # 处理多个文件，但只返回最后一个（用于 Gradio 音频播放器显示）
        last_file_paths = []
        for file_path in files:
            instrumental, vocals, file_paths = self.separate(
                audio=file_path,
                model_name=model_name,
                device=device,
                segment_size=segment_size,
                save_file=save_file,
                progress=progress
            )
            if file_paths:
                last_file_paths = file_paths
                
        return last_file_paths

    @staticmethod
    def get_device():
        if torch.cuda.is_available():
            return "cuda"
        if torch.xpu.is_available():
            return "xpu"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"

    def offload(self):
        """Offload the model and free up the memory"""
        if self.model is not None:
            del self.model
            self.model = None
        if self.device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_max_memory_allocated()
        if self.device == "xpu":
            torch.xpu.empty_cache()
            torch.xpu.reset_accumulated_memory_stats()
            torch.xpu.reset_peak_memory_stats()
        gc.collect()
        self.audio_info = None

    @staticmethod
    def cache_parameters(model_size: str, segment_size: int):
        cached_params = load_yaml(DEFAULT_PARAMETERS_CONFIG_PATH)
        cached_uvr_params = cached_params.get("bgm_separation", {})
        uvr_params_to_cache = {
            "model_size": model_size,
            "segment_size": segment_size
        }
        cached_uvr_params.update(uvr_params_to_cache)
        cached_params["bgm_separation"] = cached_uvr_params
        save_yaml(cached_params, DEFAULT_PARAMETERS_CONFIG_PATH)
