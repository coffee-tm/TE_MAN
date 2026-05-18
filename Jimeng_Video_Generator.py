from __future__ import annotations

import json
import base64
import mimetypes
import os
import sys
import time
import wave
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple

import requests
from requests.exceptions import JSONDecodeError as RequestsJSONDecodeError

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
if MODULE_DIR not in sys.path:
    sys.path.insert(0, MODULE_DIR)

try:
    from server import PromptServer  # noqa: F401
except Exception:  # pragma: no cover
    class _DummyPromptServer:
        instance = None

    PromptServer = _DummyPromptServer()  # type: ignore

import comfy.model_management
import comfy.utils

try:
    from .logger import logger
    from .config_manager import ConfigManager
except ImportError:
    from logger import logger
    from config_manager import ConfigManager

try:
    from comfy.comfy_types import IO
    from comfy_api.input_impl import VideoFromFile
    _HAS_COMFY_VIDEO = True
except Exception:  # pragma: no cover
    IO = None  # type: ignore
    VideoFromFile = None  # type: ignore
    _HAS_COMFY_VIDEO = False


CONFIG_MANAGER = ConfigManager(MODULE_DIR)


def _ensure_not_interrupted() -> None:
    comfy.model_management.throw_exception_if_processing_interrupted()


def _resolve_files_upload_url(api_base_url: str) -> str:
    base = (api_base_url or "").strip().rstrip("/")
    if not base:
        raise ValueError("未配置有效的 API URL")
    if base.endswith("/v1/files"):
        return base
    if base.endswith("/files"):
        return base
    if base.endswith("/v1"):
        return f"{base}/files"
    return f"{base}/v1/files"


def _resolve_jimeng_submit_url(api_base_url: str) -> str:
    base = (api_base_url or "").strip().rstrip("/")
    if not base:
        raise ValueError("未配置有效的 API URL")
    if base.endswith("/v1/videos"):
        return base
    if base.endswith("/videos"):
        return base
    if base.endswith("/v1"):
        return f"{base}/videos"
    return f"{base}/v1/videos"


def _resolve_jimeng_fetch_url(api_base_url: str, task_id: str) -> str:
    base = (api_base_url or "").strip().rstrip("/")
    if not base:
        raise ValueError("未配置有效的 API URL")
    task = str(task_id or "").strip()
    if not task:
        raise ValueError("未提供有效的任务 ID")
    if base.endswith("/v1/videos"):
        return f"{base}/{task}"
    if base.endswith("/videos"):
        return f"{base}/{task}"
    if base.endswith("/v1"):
        return f"{base}/videos/{task}"
    return f"{base}/v1/videos/{task}"


def _resolve_jimeng_model(duration: int) -> str:
    duration_int = int(duration or 5)
    model_map = {
        5: "dance2-fast-5s",
        10: "dance2-fast-10s",
        15: "dance2-fast-15s",
    }
    if duration_int not in model_map:
        raise ValueError(f"不支持的时长选项: {duration_int}，仅支持 5 / 10 / 15 秒")
    return model_map[duration_int]


def _get_output_dir() -> str:
    try:
        import folder_paths  # type: ignore

        path = folder_paths.get_output_directory()
        if path:
            return path
    except Exception:
        pass

    current = MODULE_DIR
    for _ in range(6):
        candidate = os.path.join(current, "output")
        if os.path.isdir(candidate):
            return candidate
        parent = os.path.dirname(current)
        if parent == current:
            break
        current = parent

    return os.path.join(os.path.dirname(os.path.dirname(MODULE_DIR)), "output")


def _download_video(
    video_url: str,
    api_key: str,
    timeout_s: int,
    bypass_proxy: bool,
) -> str:
    output_dir = _get_output_dir()
    os.makedirs(output_dir, exist_ok=True)

    ext = ".mp4"
    path_part = (video_url.split("?", 1)[0]).split("#", 1)[0]
    _, guessed_ext = os.path.splitext(path_part)
    if guessed_ext and len(guessed_ext) <= 6:
        ext = guessed_ext

    filename = f"jimeng_video_{int(time.time())}{ext}"
    filepath = os.path.join(output_dir, filename)

    session = requests.Session()
    try:
        if bypass_proxy:
            session.trust_env = False
        header_variants = [
            {"User-Agent": "ComfyUI-TE_API/1.0"},
        ]
        if api_key:
            header_variants.append(
                {
                    "User-Agent": "ComfyUI-TE_API/1.0",
                    "Authorization": f"Bearer {api_key}",
                }
            )

        last_error: Optional[Exception] = None
        for headers in header_variants:
            try:
                with session.get(video_url, headers=headers, stream=True, timeout=timeout_s) as resp:
                    resp.raise_for_status()
                    with open(filepath, "wb") as handle:
                        for chunk in resp.iter_content(chunk_size=1024 * 256):
                            _ensure_not_interrupted()
                            if not chunk:
                                continue
                            handle.write(chunk)
                    return filepath
            except Exception as exc:
                last_error = exc
                try:
                    if os.path.exists(filepath):
                        os.remove(filepath)
                except Exception:
                    pass
                continue
    finally:
        try:
            session.close()
        except Exception:
            pass

    if last_error is not None:
        raise last_error
    return filepath


def _image_tensor_to_png_bytes(image_tensor) -> bytes:
    try:
        import numpy as np
        from PIL import Image
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"缺少图像依赖（numpy/PIL）: {exc}")

    if image_tensor is None:
        raise ValueError("未提供有效的 IMAGE 输入")

    if hasattr(image_tensor, "cpu"):
        arr = image_tensor.cpu().numpy()
    else:
        arr = np.array(image_tensor)

    if arr.ndim == 4:
        arr = arr[0]
    if arr.ndim != 3:
        raise ValueError(f"不支持的 IMAGE 维度: {arr.shape}")

    if float(arr.max()) <= 1.0:
        arr = (np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)
    else:
        arr = np.clip(arr, 0.0, 255.0).astype(np.uint8)

    image = Image.fromarray(arr)
    buf = BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def _image_tensor_to_data_url(image_tensor) -> str:
    png_bytes = _image_tensor_to_png_bytes(image_tensor)
    encoded = base64.b64encode(png_bytes).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


def _bytes_to_data_url(file_bytes: bytes, mime_type: str) -> str:
    encoded = base64.b64encode(file_bytes).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"

def _audio_input_to_wav_bytes(audio_input) -> bytes:
    try:
        import numpy as np
        import torch
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"缺少音频依赖（numpy/torch）: {exc}")

    if not isinstance(audio_input, dict) or audio_input.get("waveform") is None:
        raise ValueError("未提供有效的 AUDIO 输入")

    waveform = audio_input["waveform"]
    sample_rate = int(audio_input.get("sample_rate", 44100))

    if torch.is_tensor(waveform):
        data = waveform.detach().cpu().numpy()
    else:
        data = np.array(waveform)

    if data.ndim == 3:
        data = data[0]
    if data.ndim == 1:
        data = data[None, :]
    if data.ndim != 2:
        raise ValueError(f"不支持的 AUDIO 维度: {data.shape}")

    channels, samples = data.shape
    pcm = (np.clip(data, -1.0, 1.0) * 32767.0).astype(np.int16)
    interleaved = pcm[0] if channels == 1 else np.transpose(pcm, (1, 0)).reshape(-1)

    buf = BytesIO()
    with wave.open(buf, "wb") as wav:
        wav.setnchannels(int(channels))
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(interleaved.tobytes())
    return buf.getvalue()


def _media_input_to_bytes(media_input, default_ext: str, label: str) -> Tuple[Optional[bytes], Optional[str]]:
    if media_input is None:
        return (None, None)

    get_stream = getattr(media_input, "get_stream_source", None)
    if callable(get_stream):
        try:
            source = media_input.get_stream_source()
            if isinstance(source, str):
                source = source.strip()
                if source and os.path.isfile(source):
                    with open(source, "rb") as handle:
                        return (handle.read(), os.path.basename(source))
                return (None, None)
            if isinstance(source, BytesIO):
                source.seek(0)
                data = source.read()
                if data:
                    return (data, f"reference_{label}{default_ext}")
                return (None, None)
            if hasattr(source, "read"):
                if hasattr(source, "seek"):
                    source.seek(0)
                data = source.read()
                if data:
                    return (data, f"reference_{label}{default_ext}")
                return (None, None)
        except Exception:
            return (None, None)

    if isinstance(media_input, str):
        path = media_input.strip()
        if path and os.path.isfile(path):
            with open(path, "rb") as handle:
                return (handle.read(), os.path.basename(path))
        return (None, None)

    if isinstance(media_input, dict):
        path = (
            media_input.get("path")
            or media_input.get("file")
            or media_input.get("file_path")
            or media_input.get("filename")
            or ""
        )
        path = str(path).strip() if path else ""
        if path and os.path.isfile(path):
            with open(path, "rb") as handle:
                return (handle.read(), os.path.basename(path))
        if label == "audio" and media_input.get("waveform") is not None:
            return (_audio_input_to_wav_bytes(media_input), f"reference_audio{default_ext}")
        return (None, None)

    for attr in ("path", "file_path"):
        path = getattr(media_input, attr, None)
        if isinstance(path, str):
            path = path.strip()
            if path and os.path.isfile(path):
                with open(path, "rb") as handle:
                    return (handle.read(), os.path.basename(path))

    return (None, None)


def _extract_uploaded_file_url(data: Any) -> str:
    if isinstance(data, str):
        return data if data.startswith("http") else ""
    if isinstance(data, dict):
        for key in ("url", "file_url", "download_url"):
            value = data.get(key)
            if isinstance(value, str) and value.startswith("http"):
                return value
        for value in data.values():
            picked = _extract_uploaded_file_url(value)
            if picked:
                return picked
    if isinstance(data, list):
        for item in data:
            picked = _extract_uploaded_file_url(item)
            if picked:
                return picked
    return ""


def _extract_jimeng_video_url(data: Any) -> str:
    if isinstance(data, str):
        return data if data.startswith("http") and ".mp4" in data.lower() else ""
    if isinstance(data, dict):
        for key in ("video", "video_url", "videoUrl", "output", "url"):
            value = data.get(key)
            if isinstance(value, str) and value.startswith("http"):
                return value

        nested = data.get("data")
        if isinstance(nested, dict):
            picked = _extract_jimeng_video_url(nested)
            if picked:
                return picked

        task_result = data.get("task_result")
        if isinstance(task_result, dict):
            picked = _extract_jimeng_video_url(task_result)
            if picked:
                return picked

        videos = data.get("videos")
        if isinstance(videos, list):
            for item in videos:
                if isinstance(item, dict):
                    url = item.get("url")
                    if isinstance(url, str) and url.startswith("http"):
                        return url

        for value in data.values():
            picked = _extract_jimeng_video_url(value)
            if picked:
                return picked

    if isinstance(data, list):
        for item in data:
            picked = _extract_jimeng_video_url(item)
            if picked:
                return picked
    return ""


def _read_json_response_or_raise(resp: requests.Response, label: str) -> Dict[str, Any]:
    try:
        data = resp.json()
    except (ValueError, RequestsJSONDecodeError):
        raw = (resp.text or "").strip()
        content_type = (resp.headers.get("content-type") or "").strip()
        snippet = raw[:2000] if raw else "<empty body>"
        raise RuntimeError(
            f"{label} 返回的不是 JSON。HTTP {resp.status_code}, Content-Type: {content_type or '<missing>'}, Body: {snippet}"
        )

    if not isinstance(data, dict):
        compact = json.dumps(data, ensure_ascii=False)[:2000]
        raise RuntimeError(f"{label} 返回 JSON 结构异常：{compact}")
    return data


class BananaJimengVideoNode:
    """
    ComfyUI 节点：即梦视频异步生成

    - POST /v1/videos 提交任务
    - GET /v1/videos/{task_id} 轮询状态
    """

    if _HAS_COMFY_VIDEO:
        RETURN_TYPES = (IO.VIDEO, "STRING", "STRING", "STRING")
        RETURN_NAMES = ("video", "video_url", "task_id", "response")
    else:  # pragma: no cover
        RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING")
        RETURN_NAMES = ("video_path", "video_url", "task_id", "response")

    FUNCTION = "generate_video"
    OUTPUT_NODE = True
    CATEGORY = "TE MAN/Jimeng"

    @classmethod
    def INPUT_TYPES(cls):
        audio_input_type = IO.AUDIO if _HAS_COMFY_VIDEO and hasattr(IO, "AUDIO") else "STRING"
        video_input_type = IO.VIDEO if _HAS_COMFY_VIDEO and hasattr(IO, "VIDEO") else "STRING"
        return {
            "required": {
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "一个电影感很强的雨夜街头镜头，霓虹灯反射在地面上",
                    "tooltip": "用于生成视频的提示词",
                }),
                "duration": ([5, 10, 15], {
                    "default": 5,
                    "tooltip": "视频时长（秒）",
                }),
                "aspect_ratio": (["1:1", "21:9", "16:9", "9:16", "4:3", "3:4", "adaptive"], {
                    "default": "16:9",
                    "tooltip": "视频比例，adaptive 表示根据图片自动调整",
                }),
            },
            "optional": {
                "api_key": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "API Key；留空则读取 config.ini",
                }),
                "api_base_url": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "服务地址，如 http://localhost:8000",
                }),
                "image": ("IMAGE", {
                    "tooltip": "第一张参考图（可选），按 b64 直传",
                }),
                "image_2": ("IMAGE", {
                    "tooltip": "第二张参考图（可选）",
                }),
                "image_3": ("IMAGE", {
                    "tooltip": "第三张参考图（可选）",
                }),
                "image_4": ("IMAGE", {
                    "tooltip": "第四张参考图（可选）",
                }),
                "image_5": ("IMAGE", {
                    "tooltip": "第五张参考图（可选）",
                }),
                "image_6": ("IMAGE", {
                    "tooltip": "第六张参考图（可选）",
                }),
                "image_7": ("IMAGE", {
                    "tooltip": "第七张参考图（可选）",
                }),
                "image_8": ("IMAGE", {
                    "tooltip": "第八张参考图（可选）",
                }),
                "image_9": ("IMAGE", {
                    "tooltip": "第九张参考图（可选）",
                }),
                "video_1": (video_input_type, {
                    "tooltip": "第一路参考视频（可选）",
                }),
                "video_2": (video_input_type, {
                    "tooltip": "第二路参考视频（可选）",
                }),
                "video_3": (video_input_type, {
                    "tooltip": "第三路参考视频（可选）",
                }),
                "audio_1": (audio_input_type, {
                    "tooltip": "第一路参考音频（可选）",
                }),
                "audio_2": (audio_input_type, {
                    "tooltip": "第二路参考音频（可选）",
                }),
                "audio_3": (audio_input_type, {
                    "tooltip": "第三路参考音频（可选）",
                }),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 2147483647,
                    "tooltip": "随机种子，0 为不传",
                }),
                "poll_interval_s": ("INT", {
                    "default": 5,
                    "min": 2,
                    "max": 60,
                    "tooltip": "轮询间隔（秒）",
                }),
                "绕过代理": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "需要忽略系统代理/梯子时开启",
                }),
            }
        }

    @staticmethod
    def _resolve_key_and_url(api_key: str, api_url: str) -> Tuple[str, str]:
        resolved_key = CONFIG_MANAGER.sanitize_api_key(api_key) or CONFIG_MANAGER.sanitize_api_key(
            CONFIG_MANAGER.load_api_key()
        )
        if not resolved_key:
            raise ValueError("请在 config.ini 中配置 API Key 或在节点中填写")

        resolved_url = (api_url or "").strip() or CONFIG_MANAGER.get_effective_api_base_url()
        resolved_url = resolved_url.strip()
        if not resolved_url:
            raise ValueError("请在 config.ini 中配置 API URL 或在节点中填写")

        return resolved_key, resolved_url

    def _media_bytes_to_data_url(
        self,
        file_bytes: bytes,
        filename: str,
    ) -> str:
        mime_type, _ = mimetypes.guess_type(filename)
        if not mime_type:
            mime_type = "application/octet-stream"
        return _bytes_to_data_url(file_bytes, mime_type)

    def generate_video(
        self,
        prompt: str,
        duration: int = 5,
        aspect_ratio: str = "16:9",
        api_key: str = "",
        api_base_url: str = "",
        image=None,
        image_2=None,
        image_3=None,
        image_4=None,
        image_5=None,
        image_6=None,
        image_7=None,
        image_8=None,
        image_9=None,
        video_1=None,
        video_2=None,
        video_3=None,
        audio_1=None,
        audio_2=None,
        audio_3=None,
        seed: int = 0,
        poll_interval_s: int = 5,
        绕过代理: bool = False,
    ):
        _ensure_not_interrupted()

        prompt_text = (prompt or "").strip()
        if not prompt_text:
            raise ValueError("请输入 prompt")

        resolved_key, resolved_url = self._resolve_key_and_url(api_key, api_base_url)
        submit_url = _resolve_jimeng_submit_url(resolved_url)
        bypass_proxy = bool(绕过代理)
        poll_interval_s = max(2, int(poll_interval_s or 5))
        resolution = "480p"
        cfg_scale = 0.5
        resolved_model = _resolve_jimeng_model(duration)

        masked_key = resolved_key[:8] + "..." + resolved_key[-4:] if len(resolved_key) > 12 else "***"
        logger.header("🎬 即梦视频生成任务")
        logger.info("API URL: [已脱敏]")
        logger.info(f"API Key: {masked_key}")
        logger.info(f"API: {submit_url}")
        logger.info(f"Model: {resolved_model}")
        logger.info(f"Duration: {duration}s")
        logger.info(f"Resolution: {resolution}")
        logger.info(f"Aspect Ratio: {aspect_ratio}")
        logger.info(f"CFG Scale: {cfg_scale}")
        logger.info(
            f"Mode: {'multi_reference' if any(v is not None for v in (image, image_2, image_3, image_4, image_5, image_6, image_7, image_8, image_9, video_1, video_2, video_3, audio_1, audio_2, audio_3)) else 'text_to_video'}"
        )
        logger.separator()

        progress_bar = comfy.utils.ProgressBar(100)
        progress_bar.update_absolute(5)

        payload: Dict[str, Any] = {
            "prompt": prompt_text,
            "model": resolved_model,
            "resolution": str(resolution or "480p"),
            "aspect_ratio": str(aspect_ratio or "16:9"),
            "cfg_scale": float(cfg_scale),
        }
        if int(seed or 0) > 0:
            payload["seed"] = int(seed)

        def scale_reference_image(img, index: int):
            return img

        primary_image = scale_reference_image(image, 1)
        uploaded_image_inputs = [
            scale_reference_image(img, idx)
            for idx, img in enumerate((image_2, image_3, image_4, image_5, image_6, image_7, image_8, image_9), start=2)
            if img is not None
        ]
        video_inputs = [vid for vid in (video_1, video_2, video_3) if vid is not None]
        audio_inputs = [aud for aud in (audio_1, audio_2, audio_3) if aud is not None]

        image_urls: List[str] = []
        video_urls: List[str] = []
        audio_urls: List[str] = []

        if primary_image is not None or uploaded_image_inputs or video_inputs or audio_inputs:
            progress_bar.update_absolute(15)

        if primary_image is not None:
            image_urls.append(_image_tensor_to_data_url(primary_image))
            logger.info("第一张参考图已按 b64 直传方式写入请求")

        for idx, img in enumerate(uploaded_image_inputs):
            image_urls.append(_image_tensor_to_data_url(img))
            progress_bar.update_absolute(min(22, 16 + idx))

        for idx, vid in enumerate(video_inputs):
            file_bytes, filename = _media_input_to_bytes(vid, ".mp4", "video")
            if not file_bytes:
                raise RuntimeError(f"第 {idx + 1} 路参考视频读取失败")
            video_data_url = self._media_bytes_to_data_url(
                file_bytes=file_bytes,
                filename=filename or f"reference_video_{idx+1}.mp4",
            )
            video_urls.append(video_data_url)
            progress_bar.update_absolute(min(24, 22 + idx))

        for idx, aud in enumerate(audio_inputs):
            file_bytes, filename = _media_input_to_bytes(aud, ".wav", "audio")
            if not file_bytes:
                raise RuntimeError(f"第 {idx + 1} 路参考音频读取失败")
            audio_data_url = self._media_bytes_to_data_url(
                file_bytes=file_bytes,
                filename=filename or f"reference_audio_{idx+1}.wav",
            )
            audio_urls.append(audio_data_url)
            progress_bar.update_absolute(min(25, 24 + idx))

        if image_urls or video_urls or audio_urls:
            logger.info(f"参考素材已按 b64 直传写入请求: 图 {len(image_urls)} / 视频 {len(video_urls)} / 音频 {len(audio_urls)}")

        if len(image_urls) == 1 and not video_urls and not audio_urls:
            payload["image_url"] = image_urls[0]
        elif image_urls:
            payload["image_urls"] = image_urls

        if video_urls:
            payload["video_urls"] = video_urls

        if audio_urls:
            payload["audio_urls"] = audio_urls

        if image_urls or video_urls or audio_urls:
            content: List[Dict[str, Any]] = [{"type": "text", "text": prompt_text}]
            for url in image_urls:
                content.append({"type": "image_url", "image_url": {"url": url}, "role": "reference_image"})
            for url in video_urls:
                content.append({"type": "video_url", "video_url": {"url": url}, "role": "reference_video"})
            for url in audio_urls:
                content.append({"type": "audio_url", "audio_url": {"url": url}, "role": "reference_audio"})
            payload["content"] = content

        progress_bar.update_absolute(25)

        session = requests.Session()
        task_id = ""
        video_url = ""
        try:
            if bypass_proxy:
                session.trust_env = False

            submit_resp = session.post(
                submit_url,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {resolved_key}",
                    "User-Agent": "ComfyUI-TE_API/1.0",
                },
                json=payload,
                timeout=120,
            )
            if submit_resp.status_code != 200:
                text = (submit_resp.text or "").strip()
                if len(text) > 2000:
                    text = text[:2000] + "…"
                raise RuntimeError(f"提交任务失败: {submit_resp.status_code} - {text}")

            submit_data = _read_json_response_or_raise(submit_resp, "提交任务接口")
            if submit_data.get("code") not in ("success", "SUCCESS", None):
                msg = submit_data.get("message") or submit_data.get("msg") or str(submit_data)
                raise RuntimeError(f"提交任务失败: {msg}")

            submit_data_field = submit_data.get("data")
            nested_task_id = ""
            if isinstance(submit_data_field, dict):
                nested_task_id = str(
                    submit_data_field.get("task_id")
                    or submit_data_field.get("id")
                    or submit_data_field.get("taskId")
                    or ""
                ).strip()

            task_id = str(
                nested_task_id
                or submit_data.get("task_id")
                or submit_data.get("id")
                or submit_data.get("taskId")
                or (submit_data_field if isinstance(submit_data_field, str) else "")
                or ""
            ).strip()
            if not task_id:
                raise RuntimeError(f"接口未返回 task_id：{json.dumps(submit_data, ensure_ascii=False)[:1000]}")

            logger.info(f"Task ID: {task_id}")
            progress_bar.update_absolute(40)

            while True:
                _ensure_not_interrupted()

                time.sleep(poll_interval_s)
                fetch_url = _resolve_jimeng_fetch_url(resolved_url, task_id)
                try:
                    status_resp = session.get(
                        fetch_url,
                        headers={
                            "Authorization": f"Bearer {resolved_key}",
                            "User-Agent": "ComfyUI-TE_API/1.0",
                        },
                        timeout=30,
                    )
                except requests.Timeout:
                    continue

                if status_resp.status_code != 200:
                    continue

                status_data = _read_json_response_or_raise(status_resp, "轮询任务接口")
                if not isinstance(status_data, dict):
                    continue

                if status_data.get("code") not in ("success", "SUCCESS", None):
                    continue

                data = status_data.get("data", {})
                if not isinstance(data, dict):
                    data = {}

                status = str(data.get("status") or status_data.get("status") or "").upper()
                progress_text = data.get("progress") or status_data.get("progress") or ""

                try:
                    if isinstance(progress_text, str) and progress_text.endswith("%"):
                        progress_num = int(progress_text.rstrip("%"))
                        progress_bar.update_absolute(min(90, 40 + progress_num * 50 // 100))
                    else:
                        progress_bar.update_absolute(60)
                except Exception:
                    progress_bar.update_absolute(60)

                if status in {"SUCCESS", "SUCCEEDED", "COMPLETED"}:
                    video_url = _extract_jimeng_video_url(data) or _extract_jimeng_video_url(status_data)
                    if video_url:
                        break
                    continue

                if status in {"FAILED", "FAILURE", "ERROR"}:
                    fail_reason = data.get("fail_reason") or data.get("message") or status_data.get("message") or "未知错误"
                    raise RuntimeError(f"视频生成失败: {fail_reason}")

            progress_bar.update_absolute(95)
            logger.success(f"视频生成完成: {video_url}")
            logger.info("开始下载视频到 ComfyUI/output ...")
            filepath = _download_video(
                video_url=video_url,
                api_key=resolved_key,
                timeout_s=300,
                bypass_proxy=bypass_proxy,
            )
            logger.success(f"视频已保存: {filepath}")

            response_info = json.dumps(
                {
                    "submit_endpoint": submit_url,
                    "fetch_endpoint": _resolve_jimeng_fetch_url(resolved_url, task_id),
                    "task_id": task_id,
                    "model": resolved_model,
                    "duration": int(duration),
                    "video_url": video_url,
                    "resolution": str(resolution or "480p"),
                    "reference_images": len(image_urls),
                    "reference_videos": len(video_urls),
                    "reference_audios": len(audio_urls),
                    "payload": payload,
                    "file": filepath,
                },
                ensure_ascii=False,
            )

            if _HAS_COMFY_VIDEO:
                progress_bar.update_absolute(100)
                return (VideoFromFile(open(filepath, "rb")), video_url, task_id, response_info)
            progress_bar.update_absolute(100)
            return (filepath, video_url, task_id, response_info)
        finally:
            try:
                session.close()
            except Exception:
                pass


NODE_CLASS_MAPPINGS = {
    "TE_image_pro_jimeng_video": BananaJimengVideoNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TE_image_pro_jimeng_video": "TE MAN Jimeng Video",
}
