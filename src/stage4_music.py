# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import base64
import datetime as dt
import json
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests
from loguru import logger
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOGS_DIR = PROJECT_ROOT / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)
logger.add(
    str(LOGS_DIR / "stage4_{time:YYYYMMDD_HHmmss}.log"),
    rotation="10 MB",
    level="INFO",
    encoding="utf-8",
)


DEFAULT_MUSIC_PROMPT = (
    "生成一段适合水排序游戏演示视频的 BGM，感觉干净、聪明、轻松、有趣。偏 mobile casual puzzle 风格，使用明亮但柔和的 synth、plucky mallet、轻微点击类节奏和水感音效，营造顺滑、专注、解压的氛围。要求可无缝循环，节奏稳定，不要抢画面。"
)


def _now() -> str:
    return dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@dataclass
class OpenRouterConfig:
    """OpenRouter/zhiqite request configuration."""

    api_key: str
    model: str
    base_url: str = "https://openrouter.ai/api/v1"
    timeout_seconds: int = 180
    referer: str = ""
    title: str = "video-generate"
    log_dir: str = ""


class OpenRouterHTTPClient:
    """OpenRouter-compatible client with structured IO/error logging."""

    def __init__(self, config: OpenRouterConfig) -> None:
        self.config = config
        self.base_url = config.base_url.rstrip("/")
        self.log_path: Path | None = None
        if config.log_dir:
            log_dir = Path(config.log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)
            self.log_path = log_dir / "openrouter_stage4_io.jsonl"

    def chat_audio(self, prompt: str, audio_format: str = "mp3", temperature: float = 0.9) -> bytes:
        """Request audio output through the streaming chat endpoint."""
        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": [{"role": "user", "content": prompt}],
            "modalities": ["text", "audio"],
            "audio": {"format": audio_format},
            "temperature": temperature,
            "stream": True,
        }

        voice = os.getenv("OPENROUTER_MUSIC_AUDIO_VOICE", "").strip()
        if voice:
            payload["audio"]["voice"] = voice

        try:
            audio_bytes = self._chat_audio_stream(payload)
            self._log("ok", payload, {"audio_bytes": len(audio_bytes)})
            return audio_bytes
        except MusicApiError as exc:
            self._log(
                "error",
                payload,
                {
                    "status_code": exc.status_code,
                    "retryable": exc.retryable,
                    "error": exc.detail,
                },
            )
            raise
        except Exception as exc:
            self._log("error", payload, {"error": str(exc)})
            raise

    def _headers(self) -> dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.config.api_key}",
        }
        if self.config.referer:
            headers["HTTP-Referer"] = self.config.referer
        if self.config.title:
            headers["X-Title"] = self.config.title
        return headers

    def _chat_audio_stream(self, payload: dict[str, Any]) -> bytes:
        url = f"{self.base_url}/chat/completions"
        response = requests.post(
            url,
            headers=self._headers(),
            json=payload,
            timeout=self.config.timeout_seconds,
            stream=True,
        )
        self._raise_for_music_api_error(response)

        content_type = response.headers.get("content-type", "").lower()
        if content_type.startswith("audio/"):
            return response.content

        audio_chunks: list[str] = []
        for raw_line in response.iter_lines():
            if not raw_line:
                continue
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line.startswith("data: "):
                continue

            data = line[len("data: ") :].strip()
            if data == "[DONE]":
                break

            try:
                chunk = json.loads(data)
            except json.JSONDecodeError:
                continue

            delta = ((chunk.get("choices") or [{}])[0].get("delta") or {})
            audio = delta.get("audio") or {}
            if audio.get("data"):
                audio_chunks.append(str(audio["data"]))

        if not audio_chunks:
            raise RuntimeError("音乐接口未在流式响应中返回 delta.audio.data。")

        return base64.b64decode("".join(audio_chunks))

    @staticmethod
    def _raise_for_music_api_error(response: requests.Response) -> None:
        if response.status_code < 400:
            return

        detail = response.text.strip()
        if len(detail) > 2000:
            detail = detail[:2000] + "...<truncated>"

        retryable = response.status_code == 429 or response.status_code >= 500
        raise MusicApiError(
            status_code=response.status_code,
            url=response.url,
            detail=detail or "<empty response body>",
            retryable=retryable,
        )

    def _log(self, stage: str, request_payload: dict[str, Any], response_payload: dict[str, Any]) -> None:
        if self.log_path is None:
            return
        obj = {
            "timestamp": _now(),
            "stage": stage,
            "model": self.config.model,
            "request": request_payload,
            "response": response_payload,
        }
        with self.log_path.open("a", encoding="utf-8") as file_obj:
            file_obj.write(json.dumps(obj, ensure_ascii=False) + "\n")


class VideoMusicMixer:
    """阶段四：为阶段三成品生成 BGM，并用 FFmpeg 与原音轨混音。"""

    def __init__(
        self,
        processed_dir: str = "data/processed",
        video_volume: float = 1.0,
        bgm_volume: float = 0.6,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        prompt: str = DEFAULT_MUSIC_PROMPT,
        log_dir: str | None = None,
    ) -> None:
        self.processed_dir = Path(processed_dir)
        self.video_volume = max(0.0, float(video_volume))
        self.bgm_volume = max(0.0, float(bgm_volume))
        self.model = (
            model
            or os.getenv("OPENROUTER_MUSIC_MODEL", "google/lyria-3-clip-preview").strip()
            or "google/lyria-3-clip-preview"
        )
        self.prompt = prompt.strip() or DEFAULT_MUSIC_PROMPT

        resolved_api_key = api_key or os.getenv("OPENROUTER_API_KEY", "sk-or-v1-5794a8b038307965ef5bcdfea40fcfc18").strip()
        if not resolved_api_key:
            raise ValueError("缺少 OpenRouter API Key，请设置 OPENROUTER_API_KEY 或传入 --api-key。")

        resolved_base_url = (
            base_url
            or os.getenv("OPENROUTER_MUSIC_BASE_URL", "https://apirouter.zhiqiteai.cn/ApiRouterServ/v1").strip()
            or os.getenv("OPENROUTER_BASE_URL", "").strip()
            or "https://openrouter.ai/api/v1"
        )
        resolved_log_dir = (
            log_dir
            if log_dir is not None
            else os.getenv("OPENROUTER_STAGE4_LOG_DIR", str(LOGS_DIR / "openrouter_stage4")).strip()
        )
        self.client = OpenRouterHTTPClient(
            OpenRouterConfig(
                api_key=resolved_api_key,
                model=self.model,
                base_url=resolved_base_url,
                timeout_seconds=int(os.getenv("OPENROUTER_MUSIC_TIMEOUT", "180")),
                referer=os.getenv("OPENROUTER_REFERER", "").strip(),
                title=os.getenv("OPENROUTER_TITLE", "video-generate").strip(),
                log_dir=resolved_log_dir,
            )
        )

    def run(self, only_video: str | None = None) -> list[dict[str, str]]:
        """遍历 data/processed/{video_name}/ 下的无 BGM 成品并输出 final 视频。"""
        if not self.processed_dir.exists():
            logger.warning("未找到成品目录: {}", self.processed_dir)
            return []

        video_paths = self._collect_video_paths(only_video=only_video)
        if not video_paths:
            logger.warning("未找到待混音视频，目录: {}", self.processed_dir)
            return []

        logger.info(
            "阶段四启动 | 待处理视频 {} 个 | 模型 {} | 原音量 {:.2f} | BGM 音量 {:.2f}",
            len(video_paths),
            self.model,
            self.video_volume,
            self.bgm_volume,
        )

        outputs: list[dict[str, str]] = []
        for index, video_path in enumerate(video_paths, start=1):
            logger.info("正在处理 ({}/{}): {}", index, len(video_paths), video_path)
            try:
                bgm_path = self.generate_bgm_for_video(video_path)
                output_path = self.mix_audio(video_path=video_path, bgm_path=bgm_path)
                outputs.append(
                    {
                        "video_path": self._json_path(video_path),
                        "bgm_path": self._json_path(bgm_path),
                        "output_path": self._json_path(output_path),
                    }
                )
            except Exception as exc:
                logger.exception("阶段四处理失败，已跳过: {} | 错误: {}", video_path, exc)

        logger.info("阶段四结束 | 成功输出 {} 个 final 视频", len(outputs))
        return outputs

    def generate_bgm_for_video(self, video_path: Path) -> Path:
        """调用音乐模型生成音频，并独立保存到视频所在目录。"""
        timestamp = self._timestamp()
        bgm_path = video_path.with_name(f"{video_path.stem}_bgm_{timestamp}.mp3")

        logger.info("正在生成 BGM: {} | prompt: {}", bgm_path.name, self.prompt)
        audio_bytes = self._generate_music_with_retry()
        if not audio_bytes:
            raise RuntimeError("音乐模型未返回可保存的音频内容。")

        bgm_path.write_bytes(audio_bytes)
        logger.info("BGM 已保存: {} | 大小 {:.1f} KB", bgm_path, len(audio_bytes) / 1024)
        return bgm_path

    def mix_audio(self, video_path: Path, bgm_path: Path) -> Path:
        """使用 FFmpeg 将原视频音轨和 BGM 混合，画面流拷贝。"""
        timestamp = self._timestamp()
        output_path = video_path.with_name(f"{video_path.stem}_final_{timestamp}.mp4")
        filter_complex = (
            f"[0:a]volume={self.video_volume}[a1];"
            f"[1:a]volume={self.bgm_volume}[a2];"
            "[a1][a2]amix=inputs=2:duration=first:dropout_transition=2[a]"
        )
        command = [
            "ffmpeg",
            "-y",
            "-i",
            str(video_path),
            "-i",
            str(bgm_path),
            "-filter_complex",
            filter_complex,
            "-map",
            "0:v",
            "-map",
            "[a]",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-b:a",
            "128k",
            str(output_path),
        ]

        logger.info("正在混音: {} + {} -> {}", video_path.name, bgm_path.name, output_path.name)
        self._run_ffmpeg(command, f"混音失败: {video_path}")
        logger.info("final 视频已生成: {}", output_path)
        return output_path

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=12),
        retry=retry_if_exception(lambda exc: not isinstance(exc, MusicApiError) or exc.retryable),
        reraise=True,
    )
    def _generate_music_with_retry(self) -> bytes:
        """请求 OpenRouter 兼容接口，并提取返回的音频 bytes。"""
        return self.client.chat_audio(prompt=self.prompt, audio_format="mp3", temperature=0.9)

    def _extract_audio_bytes(self, response: Any) -> bytes:
        if isinstance(response, dict):
            choices = response.get("choices") or []
            if not choices:
                raise RuntimeError(f"音乐模型返回中未找到 choices: {response}")
            choice = choices[0]
            message = choice.get("message", {}) if isinstance(choice, dict) else {}
        else:
            choice = response.choices[0]
            message = choice.message

        audio = self._read_audio_field(message, "audio")

        if audio is None and isinstance(message, dict):
            audio = message.get("audio")

        if audio is None:
            content = self._read_audio_field(message, "content")
            if isinstance(content, list):
                audio = self._audio_part_from_content(content)
            elif isinstance(content, str) and self._looks_like_url(content):
                return self._download_audio(content)

        if audio is None:
            raise RuntimeError(f"音乐模型返回中未找到 audio 字段: {response}")

        audio_data = self._read_audio_field(audio, "data")
        if audio_data:
            return base64.b64decode(audio_data)

        audio_url = self._read_audio_field(audio, "url") or self._read_audio_field(audio, "audio_url")
        if audio_url:
            return self._download_audio(audio_url)

        raw_bytes = self._read_audio_field(audio, "bytes")
        if isinstance(raw_bytes, bytes):
            return raw_bytes

        raise RuntimeError(f"无法解析音乐模型音频返回: {audio}")

    def _audio_part_from_content(self, content: list[Any]) -> Any | None:
        for part in content:
            if not isinstance(part, dict):
                continue
            if part.get("type") in {"audio", "input_audio", "output_audio"}:
                return part.get("audio") or part
        return None

    @staticmethod
    def _read_audio_field(audio: Any, key: str) -> Any:
        if isinstance(audio, dict):
            return audio.get(key)
        return getattr(audio, key, None)

    @staticmethod
    def _looks_like_url(value: str) -> bool:
        parsed = urlparse(value.strip())
        return parsed.scheme in {"http", "https"} and bool(parsed.netloc)

    @staticmethod
    def _download_audio(url: str) -> bytes:
        logger.info("正在下载 BGM 音频: {}", url)
        response = requests.get(url, timeout=120)
        response.raise_for_status()
        return response.content

    def _collect_video_paths(self, only_video: str | None = None) -> list[Path]:
        roots: list[Path]
        if only_video:
            roots = [self.processed_dir / only_video.strip()]
        else:
            roots = [path for path in sorted(self.processed_dir.iterdir()) if path.is_dir()]

        video_paths: list[Path] = []
        for root in roots:
            if not root.exists():
                logger.warning("跳过不存在的视频成品目录: {}", root)
                continue
            for path in sorted(root.glob("v*.mp4")):
                if "_final_" in path.stem or "_bgm_" in path.stem:
                    continue
                video_paths.append(path)
        return video_paths

    @staticmethod
    def _run_ffmpeg(command: list[str], error_prefix: str) -> None:
        try:
            subprocess.run(
                command,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
        except FileNotFoundError as exc:
            raise RuntimeError("未找到 ffmpeg，请确认已安装并加入 PATH。") from exc
        except subprocess.CalledProcessError as exc:
            detail = (exc.stderr or "").strip() or "ffmpeg 未返回详细错误。"
            raise RuntimeError(f"{error_prefix}\n{detail}") from exc

    @staticmethod
    def _json_path(path: Path) -> str:
        return path.as_posix()

    @staticmethod
    def _timestamp() -> str:
        return time.strftime("%Y%m%d_%H%M%S")


class MusicApiError(RuntimeError):
    """包含音乐生成接口响应正文的错误，便于定位 4xx 参数问题。"""

    def __init__(self, status_code: int, url: str, detail: str, retryable: bool) -> None:
        self.status_code = status_code
        self.url = url
        self.detail = detail
        self.retryable = retryable
        super().__init__(
            f"Music API HTTP {status_code} ({'retryable' if retryable else 'not retryable'}) "
            f"for {url}: {detail}"
        )


def load_env_file(env_path: Path) -> None:
    """轻量读取 .env，避免强制引入 python-dotenv 依赖。"""
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def main() -> None:
    parser = argparse.ArgumentParser(description="阶段四：大模型生成 BGM 并混音")
    parser.add_argument(
        "--processed-dir",
        default="data/processed",
        help="阶段三成品目录（默认: data/processed）",
    )
    parser.add_argument(
        "--video-name",
        default=None,
        help="仅处理指定视频子目录（例如 level4）",
    )
    parser.add_argument(
        "--video-volume",
        type=float,
        default=1.0,
        help="原视频音轨音量权重（默认: 1.0）",
    )
    parser.add_argument(
        "--bgm-volume",
        type=float,
        default=0.6,
        help="BGM 音量权重（默认: 0.6）",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="音乐生成模型（默认: google/lyria-3-clip-preview）",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="OpenRouter API Key（可选，默认读取 OPENROUTER_API_KEY）",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="OpenRouter API Base URL（可选，默认读取 OPENROUTER_BASE_URL）",
    )
    parser.add_argument(
        "--prompt",
        default=DEFAULT_MUSIC_PROMPT,
        help="BGM 生成提示词",
    )
    parser.add_argument(
        "--log-dir",
        default=None,
        help="OpenRouter 请求/响应 JSONL 日志目录（默认: logs/openrouter_stage4）",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    os.chdir(project_root)
    load_env_file(project_root / ".env")

    started_at = time.perf_counter()
    logger.info("=== 阶段四启动：大模型生成 BGM 并混音 ===")
    logger.info("项目根目录: {}", project_root)

    mixer = VideoMusicMixer(
        processed_dir=args.processed_dir,
        video_volume=args.video_volume,
        bgm_volume=args.bgm_volume,
        model=args.model,
        api_key=args.api_key,
        base_url=args.base_url,
        prompt=args.prompt,
        log_dir=args.log_dir,
    )
    outputs = mixer.run(only_video=args.video_name)
    elapsed = time.perf_counter() - started_at

    logger.info("=== 阶段四完成：生成 {} 个 final 视频，耗时 {:.2f}s ===", len(outputs), elapsed)


if __name__ == "__main__":
    main()
