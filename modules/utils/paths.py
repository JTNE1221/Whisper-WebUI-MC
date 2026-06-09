import os

# --- 【核心修复】动态确定项目根目录 ---
# 获取当前文件 (paths.py) 的绝对路径，然后回溯到项目根目录
# 假设目录结构为: 项目根目录/modules/utils/paths.py
# 如果结构不同，请调整 '../..' 的层级数
WEBUI_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --- 定义目录结构 ---
MODELS_DIR = os.path.join(WEBUI_DIR, "models")
WHISPER_MODELS_DIR = os.path.join(MODELS_DIR, "Whisper")
FASTER_WHISPER_MODELS_DIR = os.path.join(WHISPER_MODELS_DIR, "faster-whisper")
INSANELY_FAST_WHISPER_MODELS_DIR = os.path.join(WHISPER_MODELS_DIR, "insanely-fast-whisper")
NLLB_MODELS_DIR = os.path.join(MODELS_DIR, "NLLB")
DIARIZATION_MODELS_DIR = os.path.join(MODELS_DIR, "Diarization")

# 【关键修改】指向你存放 UVR 模型的目录
# 请确保你的模型放在: 项目根目录/models/UVR/MDX_Net_Models/
UVR_MODELS_DIR = os.path.join(MODELS_DIR, "UVR", "MDX_Net_Models") 

CONFIGS_DIR = os.path.join(WEBUI_DIR, "configs")
DEFAULT_PARAMETERS_CONFIG_PATH = os.path.join(CONFIGS_DIR, "default_parameters.yaml")
I18N_YAML_PATH = os.path.join(CONFIGS_DIR, "translation.yaml")

OUTPUT_DIR = os.path.join(WEBUI_DIR, "outputs")
TRANSLATION_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "translations")
UVR_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "UVR")
UVR_INSTRUMENTAL_OUTPUT_DIR = os.path.join(UVR_OUTPUT_DIR, "instrumental")
UVR_VOCALS_OUTPUT_DIR = os.path.join(UVR_OUTPUT_DIR, "vocals")

BACKEND_DIR_PATH = os.path.join(WEBUI_DIR, "backend")
SERVER_CONFIG_PATH = os.path.join(BACKEND_DIR_PATH, "configs", "config.yaml")
SERVER_DOTENV_PATH = os.path.join(BACKEND_DIR_PATH, "configs", ".env")
BACKEND_CACHE_DIR = os.path.join(BACKEND_DIR_PATH, "cache")

# --- 自动创建目录 ---
# 确保所有必要的目录都存在
for dir_path in [MODELS_DIR, WHISPER_MODELS_DIR, FASTER_WHISPER_MODELS_DIR, 
                 INSANELY_FAST_WHISPER_MODELS_DIR, NLLB_MODELS_DIR, 
                 DIARIZATION_MODELS_DIR, UVR_MODELS_DIR, CONFIGS_DIR, 
                 OUTPUT_DIR, TRANSLATION_OUTPUT_DIR, UVR_INSTRUMENTAL_OUTPUT_DIR, 
                 UVR_VOCALS_OUTPUT_DIR, BACKEND_CACHE_DIR]:
    try:
        os.makedirs(dir_path, exist_ok=True)
    except Exception as e:
        print(f"Warning: Failed to create directory {dir_path}: {e}")
