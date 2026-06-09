# Adapted from https://github.com/SYSTRAN/faster-whisper/blob/master/faster_whisper/vad.py

from faster_whisper.vad import VadOptions, get_speech_timestamps
import numpy as np
from typing import BinaryIO, Union, List, Optional, Tuple
import faster_whisper
from faster_whisper.transcribe import SpeechTimestampsMap
import gradio as gr

from modules.whisper.data_classes import *


class SileroVAD:
    def __init__(self):
        self.sampling_rate = 16000
        # 移除 self.model = None，交由官方 API 自动在底层管理模型生命周期

    def run(self,
            audio: Union[str, BinaryIO, np.ndarray],
            vad_parameters: VadOptions,
            progress: gr.Progress = gr.Progress()
            ) -> Tuple[np.ndarray, List[dict]]:
        """
        Run VAD

        Parameters
        ----------
        audio: Union[str, BinaryIO, np.ndarray]
            Audio path or file binary or Audio numpy array
        vad_parameters:
            Options for VAD processing.
        progress: gr.Progress
            Indicator to show progress directly in gradio.

        Returns
        ----------
        np.ndarray
            Pre-processed audio with VAD
        List[dict]
            Chunks of speeches to be used to restore the timestamps later
        """

        sampling_rate = self.sampling_rate

        if not isinstance(audio, np.ndarray):
            audio = faster_whisper.decode_audio(audio, sampling_rate=sampling_rate)

        if vad_parameters is None:
            vad_parameters = VadOptions()
        elif isinstance(vad_parameters, dict):
            vad_parameters = VadOptions(**vad_parameters)
            
        # [核心优化区]：直接调用 faster-whisper 1.2.1 原生的 get_speech_timestamps
        # 彻底解决旧版手写逻辑导致的 1D array 报错，且完美兼容 V5/V6 模型
        speech_chunks = get_speech_timestamps(
            audio=audio,
            vad_options=vad_parameters,
        )

        audio = self.collect_chunks(audio, speech_chunks)

        return audio, speech_chunks

    @staticmethod
    def collect_chunks(audio: np.ndarray, chunks: List[dict]) -> np.ndarray:
        """Collects and concatenates audio chunks."""
        if not chunks:
            return np.array([], dtype=np.float32)

        return np.concatenate([audio[chunk["start"]: chunk["end"]] for chunk in chunks])

    @staticmethod
    def format_timestamp(
        seconds: float,
        always_include_hours: bool = False,
        decimal_marker: str = ".",
    ) -> str:
        assert seconds >= 0, "non-negative timestamp expected"
        milliseconds = round(seconds * 1000.0)

        hours = milliseconds // 3_600_000
        milliseconds -= hours * 3_600_000

        minutes = milliseconds // 60_000
        milliseconds -= minutes * 60_000

        seconds = milliseconds // 1_000
        milliseconds -= seconds * 1_000

        hours_marker = f"{hours:02d}:" if always_include_hours or hours > 0 else ""
        return (
            f"{hours_marker}{minutes:02d}:{seconds:02d}{decimal_marker}{milliseconds:03d}"
        )

    def restore_speech_timestamps(
        self,
        segments: List[Segment],
        speech_chunks: List[dict],
        sampling_rate: Optional[int] = None,
    ) -> List[Segment]:
        if sampling_rate is None:
            sampling_rate = self.sampling_rate

        ts_map = SpeechTimestampsMap(speech_chunks, sampling_rate)

        for segment in segments:
            if segment.words:
                words = []
                for word in segment.words:
                    # Ensure the word start and end times are resolved to the same chunk.
                    middle = (word.start + word.end) / 2
                    chunk_index = ts_map.get_chunk_index(middle)
                    word.start = ts_map.get_original_time(word.start, chunk_index)
                    word.end = ts_map.get_original_time(word.end, chunk_index)
                    words.append(word)

                segment.start = words[0].start
                segment.end = words[-1].end
                segment.words = words

            else:
                segment.start = ts_map.get_original_time(segment.start)
                segment.end = ts_map.get_original_time(segment.end)

        return segments
