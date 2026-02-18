from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
import re
import time
from typing import TYPE_CHECKING

import numpy as np
import torch

from src.logger.logger import get_logger
from src.utils.geometry import Size
from src.models.s2m2.src.s2m2.core.model.s2m2 import S2M2
from src.models.s2m2.src.s2m2.core.utils.image_utils import image_crop, image_pad

if TYPE_CHECKING:
	from src.config_loader.s2m2_config import S2M2Config

logger = get_logger(__name__)


@dataclass(frozen=True)
class S2M2Result:
	"""S2M2（Stereo Matching）推論結果。

	- disparity: 左画像視点の視差マップ（ピクセル単位）[H, W]
	  - 右画像の対応点は (x - disparity[y, x], y) 側に現れる想定
	- occlusion: オクルージョン（非対応/隠れ）推定マップ [H, W]
	  - 元実装の説明では 0 が「オクルード」側
	  - 点群の作成に便利
	- confidence: 信頼度マップ [H, W]
	  - 元実装の説明では「視差誤差が 4px 未満なら 1」相当のスコア
	  - 点群の作成に便利
	- avg_confidence: confidence の平均（中心領域で平均化）
	- runtime_ms: 推論時間（ms）。warmup は含めない
	- orig_shape: 入力画像の元サイズ
	- used_shape: モデル入力に使ったサイズ（32の倍数にcropした後など）
	"""
	disparity: np.ndarray
	occlusion: np.ndarray
	confidence: np.ndarray
	avg_confidence: float
	runtime_ms: float
	orig_shape: Size
	used_shape: Size


# ===========================================================================
# 抽象基底クラス（Strategyインターフェース）
# ===========================================================================

class S2M2Backend(ABC):
	"""ステレオマッチングバックエンドの共通インターフェース。

	バックエンド（PyTorch / TensorRT）によらず同一のインターフェースを提供する。
	前処理・後処理のロジックはここに集約する。
	"""

	# ------------------------------------------------------------------
	# 抽象メソッド：各バックエンドで実装する
	# ------------------------------------------------------------------

	@abstractmethod
	def predict(
		self,
		left: np.ndarray,
		right: np.ndarray,
		*,
		n_repeat: int = 1,
		crop_to_multiple_of_32: bool = True,
	) -> S2M2Result:
		"""左右画像から視差・オクルージョン・信頼度を推論する。"""
		...

	# ------------------------------------------------------------------
	# 共通ユーティリティ
	# ------------------------------------------------------------------

	@staticmethod
	def _resolve_device(device: str | None) -> torch.device:
		"""device文字列からtorch.deviceを解決する。"""
		if device is None:
			return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
		return torch.device(device)

	@staticmethod
	def _preprocess(
		left: np.ndarray,
		right: np.ndarray,
		*,
		device: torch.device,
		crop_to_multiple_of_32: bool,
	) -> tuple[torch.Tensor, torch.Tensor, int, int, int, int]:
		"""入力チェック・サイズ調整・Tensor変換を行う共通前処理。

		Returns:
			left_torch, right_torch, orig_h, orig_w, used_h, used_w
		"""
		# 入力チェック
		if left is None or right is None:
			raise ValueError("left/right is None")
		if left.shape[:2] != right.shape[:2]:
			raise ValueError(f"left/right shape mismatch: left={left.shape} right={right.shape}")
		if left.ndim != 3 or right.ndim != 3:
			raise ValueError("left/right must be HxWxC images")

		orig_h, orig_w = left.shape[:2]

		# -------------------------------
		# サイズ調整（任意）
		# - demo実装に合わせて 32 の倍数にcropする（パディング前提のため）
		# -------------------------------
		if crop_to_multiple_of_32:
			used_h = (orig_h // 32) * 32
			used_w = (orig_w // 32) * 32
			if used_h <= 0 or used_w <= 0:
				raise ValueError(f"Image too small for 32x crop: {left.shape}")
			left = left[:used_h, :used_w]
			right = right[:used_h, :used_w]
		else:
			used_h, used_w = orig_h, orig_w

		# Torch Tensor化（BCHW）
		left_torch = torch.from_numpy(left).permute(-1, 0, 1).unsqueeze(0).to(device)
		right_torch = torch.from_numpy(right).permute(-1, 0, 1).unsqueeze(0).to(device)

		return left_torch, right_torch, orig_h, orig_w, used_h, used_w

	@staticmethod
	def _post_process(
		disp_t: torch.Tensor,
		occ_t: torch.Tensor,
		conf_t: torch.Tensor,
		img_height: int,
		img_width: int,
		runtime_ms: float,
	) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
		"""推論結果テンソルをcrop/変換してnumpy出力に整形する。"""
		disp_t = image_crop(disp_t, (img_height, img_width)).squeeze().float()
		occ_t = image_crop(occ_t, (img_height, img_width)).squeeze().float()
		conf_t = image_crop(conf_t, (img_height, img_width)).squeeze().float()

		disp_np = disp_t.detach().cpu().numpy().astype(np.float32, copy=False)
		occ_np = occ_t.detach().cpu().numpy().astype(np.float32, copy=False)
		conf_np = conf_t.detach().cpu().numpy().astype(np.float32, copy=False)

		margin = 100
		if conf_np.shape[0] > 2 * margin and conf_np.shape[1] > 2 * margin:
			avg_conf_score = float(conf_np[margin:-margin, margin:-margin].mean())
		else:
			avg_conf_score = float(conf_np.mean())

		return disp_np, occ_np, conf_np, avg_conf_score, float(runtime_ms)


# ===========================================================================
# PyTorchバックエンド
# ===========================================================================

class S2M2PyTorchBackend(S2M2Backend):
	"""PyTorch（.pth）モデルを使ったステレオマッチングバックエンド。

	PyTorch固有のオプション（amp, torch_compile など）はここでのみ保持する。
	"""

	def __init__(
		self,
		model_path: str | Path,
		*,
		device: str | None = None,
		use_positivity: bool = True,
		refine_iter: int = 3,
		amp: bool = True,
		torch_compile: bool = False,
		verbose: bool = False,
	) -> None:
		self.model_path = Path(model_path)
		self.use_positivity = use_positivity
		self.refine_iter = refine_iter
		self.amp = amp
		self.torch_compile = torch_compile
		self.verbose = verbose
		self.device = self._resolve_device(device)

		self.model = self._load_model(S2M2)
		self._log_init()

	# ------------------------------------------------------------------
	# 推論
	# ------------------------------------------------------------------

	def predict(
		self,
		left: np.ndarray,
		right: np.ndarray,
		*,
		n_repeat: int = 1,
		crop_to_multiple_of_32: bool = False,
	) -> S2M2Result:
		"""左右画像から視差・オクルージョン・信頼度を推論する。"""
		left_torch, right_torch, orig_h, orig_w, used_h, used_w = self._preprocess(
			left, right, device=self.device, crop_to_multiple_of_32=crop_to_multiple_of_32
		)

		pred_disp, pred_occ, pred_conf, avg_conf, runtime_ms = self._run_inference(
			left_torch, right_torch, n_repeat=max(1, int(n_repeat))
		)

		return S2M2Result(
			disparity=pred_disp,
			occlusion=pred_occ,
			confidence=pred_conf,
			avg_confidence=avg_conf,
			runtime_ms=runtime_ms,
			orig_shape=Size(width=float(orig_w), height=float(orig_h)),
			used_shape=Size(width=float(used_w), height=float(used_h)),
		)

	@torch.no_grad()
	def _run_inference(
		self,
		left_torch: torch.Tensor,
		right_torch: torch.Tensor,
		*,
		n_repeat: int,
	) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
		"""前処理（pad）→推論→後処理（crop）をまとめて実行する。"""
		img_height, img_width = left_torch.shape[-2:]
		left_pad = image_pad(left_torch, 32).to(self.device)
		right_pad = image_pad(right_torch, 32).to(self.device)

		use_amp = bool(self.amp and self.device.type == "cuda")
		with torch.inference_mode():
			if self.device.type == "cuda":
				starter = torch.cuda.Event(enable_timing=True)
				ender = torch.cuda.Event(enable_timing=True)
				starter.record()
				with torch.amp.autocast(enabled=use_amp, device_type="cuda", dtype=torch.float16):
					for _ in range(n_repeat):
						pred_disp_t, pred_occ_t, pred_conf_t = self.model(left_pad, right_pad)
				ender.record()
				torch.cuda.synchronize()
				runtime_ms = starter.elapsed_time(ender) / n_repeat
			else:
				start_time = time.perf_counter()
				for _ in range(n_repeat):
					pred_disp_t, pred_occ_t, pred_conf_t = self.model(left_pad, right_pad)
				end_time = time.perf_counter()
				runtime_ms = ((end_time - start_time) * 1000.0) / n_repeat

		return self._post_process(pred_disp_t, pred_occ_t, pred_conf_t, img_height, img_width, runtime_ms)

	# ------------------------------------------------------------------
	# 初期化ヘルパー
	# ------------------------------------------------------------------

	def _parse_model_config(self) -> tuple[int, int]:
		""".pthファイル名からCHxxxNTRxを抽出する。"""
		match = re.search(r"CH(\d+)NTR(\d+)", self.model_path.stem)
		if match is None:
			raise ValueError(
				"model_path filename must include CH{feature_channels}NTR{n_transformer}: "
				f"{self.model_path.name}"
			)
		return int(match.group(1)), int(match.group(2))

	def _load_model(self, model_cls: type[torch.nn.Module]) -> torch.nn.Module:
		"""モデルの生成・重みロードを行う。"""
		if not self.model_path.exists():
			raise FileNotFoundError(f"Checkpoint not found: {self.model_path}")

		feature_channels, n_transformer = self._parse_model_config()
		model = model_cls(
			feature_channels=feature_channels,
			dim_expansion=1,
			num_transformer=n_transformer,
			use_positivity=self.use_positivity,
			refine_iter=self.refine_iter,
		)

		checkpoint = self._torch_load_checkpoint(self.model_path)
		state_dict = checkpoint.get("state_dict") if isinstance(checkpoint, dict) else None
		if state_dict is None:
			raise ValueError(f"Invalid checkpoint format (missing 'state_dict'): {self.model_path}")

		getattr(model, "my_load_state_dict")(state_dict)
		model.eval()
		model = model.to(self.device)

		if self.torch_compile:
			try:
				model = torch.compile(model)  # type: ignore[attr-defined]
			except Exception as e:
				logger.warning("torch.compile failed; continuing without compile: %s", e)

		return model

	@staticmethod
	def _torch_load_checkpoint(ckpt_path: Path):
		"""チェックポイントを互換オプションでロードする。"""
		try:
			return torch.load(ckpt_path, weights_only=True)
		except TypeError:
			return torch.load(ckpt_path)

	def _log_init(self) -> None:
		"""初期化時の状態をログ出力する。"""
		cuda_available = torch.cuda.is_available()
		cuda_device_count = torch.cuda.device_count() if cuda_available else 0
		logger.info(
			"S2M2PyTorchBackend init: model_path=%s device=%s use_positivity=%s "
			"refine_iter=%s amp=%s torch_compile=%s",
			self.model_path,
			self.device,
			self.use_positivity,
			self.refine_iter,
			self.amp,
			self.torch_compile,
		)
		logger.info(
			"S2M2PyTorchBackend device status: cuda_available=%s device_count=%s",
			cuda_available,
			cuda_device_count,
		)


# ===========================================================================
# TensorRTバックエンド
# ===========================================================================

class S2M2TensorRTBackend(S2M2Backend):
	"""TensorRT（.engine）エンジンを使ったステレオマッチングバックエンド。

	TensorRT固有のセットアップ（エンジンロード・バインディング）はここでのみ管理する。
	PyTorch固有の amp / torch_compile などは一切保持しない。
	"""

	def __init__(
		self,
		model_path: str | Path,
		*,
		device: str | None = None,
		verbose: bool = False,
	) -> None:
		self.model_path = Path(model_path)
		self.verbose = verbose
		self.device = self._resolve_device(device)

		self.trt_engine = None
		self.trt_context = None
		self.trt_bindings: dict[str, dict[str, object]] = {}
		self.trt_input_names: list[str] = []
		self.trt_output_names: list[str] = []

		self._load_trt_engine(self.model_path)
		self._log_init()

	# ------------------------------------------------------------------
	# 推論
	# ------------------------------------------------------------------

	def predict(
		self,
		left: np.ndarray,
		right: np.ndarray,
		*,
		n_repeat: int = 1,
		crop_to_multiple_of_32: bool = False,
	) -> S2M2Result:
		"""左右画像から視差・オクルージョン・信頼度を推論する。"""
		left_torch, right_torch, orig_h, orig_w, used_h, used_w = self._preprocess(
			left, right, device=self.device, crop_to_multiple_of_32=crop_to_multiple_of_32
		)

		img_height, img_width = left_torch.shape[-2:]
		left_pad = image_pad(left_torch, 32).to(self.device)
		right_pad = image_pad(right_torch, 32).to(self.device)

		pred_disp_t, pred_occ_t, pred_conf_t, runtime_ms = self._run_trt_inference(
			left_pad, right_pad, n_repeat=max(1, int(n_repeat))
		)
		pred_disp, pred_occ, pred_conf, avg_conf, runtime_ms = self._post_process(
			pred_disp_t, pred_occ_t, pred_conf_t, img_height, img_width, runtime_ms
		)

		return S2M2Result(
			disparity=pred_disp,
			occlusion=pred_occ,
			confidence=pred_conf,
			avg_confidence=avg_conf,
			runtime_ms=runtime_ms,
			orig_shape=Size(width=float(orig_w), height=float(orig_h)),
			used_shape=Size(width=float(used_w), height=float(used_h)),
		)

	def _run_trt_inference(
		self,
		left_pad: torch.Tensor,
		right_pad: torch.Tensor,
		*,
		n_repeat: int,
	) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
		"""TensorRTで推論し、出力テンソルと実行時間を返す。"""
		if self.trt_engine is None or self.trt_context is None:
			raise RuntimeError("TensorRT engine is not initialized")
		if self.device.type != "cuda":
			raise RuntimeError("TensorRT inference requires CUDA device")

		engine = self.trt_engine
		context = self.trt_context
		use_new_api = hasattr(engine, "num_io_tensors")

		left_name, right_name = self.trt_input_names
		left_binding = self.trt_bindings[left_name]
		right_binding = self.trt_bindings[right_name]

		# 入力サイズ検証
		for pad, binding in ((left_pad, left_binding), (right_pad, right_binding)):
			expected = binding["shape"]
			if len(expected) >= 4:
				exp_h, exp_w = expected[-2], expected[-1]
				if exp_h not in (-1, pad.shape[-2]) or exp_w not in (-1, pad.shape[-1]):
					raise ValueError(
						"TensorRT engine input size mismatch: "
						f"expected=({exp_h}, {exp_w}) got=({pad.shape[-2]}, {pad.shape[-1]})"
					)

		left_pad = left_pad.contiguous().to(dtype=torch.from_numpy(np.empty((), left_binding["dtype"])).dtype)
		right_pad = right_pad.contiguous().to(dtype=torch.from_numpy(np.empty((), right_binding["dtype"])).dtype)

		if use_new_api:
			context.set_input_shape(left_name, tuple(left_pad.shape))
			context.set_input_shape(right_name, tuple(right_pad.shape))
		else:
			context.set_binding_shape(int(left_binding["index"]), tuple(left_pad.shape))
			context.set_binding_shape(int(right_binding["index"]), tuple(right_pad.shape))

		out_tensors: dict[str, torch.Tensor] = {}
		for name in self.trt_output_names:
			binding = self.trt_bindings[name]
			shape = (
				tuple(context.get_tensor_shape(name))
				if use_new_api
				else tuple(context.get_binding_shape(int(binding["index"])))
			)
			type_torch = torch.from_numpy(np.empty((), binding["dtype"])).dtype
			out_tensors[name] = torch.empty(shape, device=self.device, dtype=type_torch)

		stream = torch.cuda.current_stream(device=self.device)
		stream_handle = stream.cuda_stream

		starter = torch.cuda.Event(enable_timing=True)
		ender = torch.cuda.Event(enable_timing=True)
		starter.record()
		for _ in range(n_repeat):
			if use_new_api:
				context.set_tensor_address(left_name, int(left_pad.data_ptr()))
				context.set_tensor_address(right_name, int(right_pad.data_ptr()))
				for name, tensor in out_tensors.items():
					context.set_tensor_address(name, int(tensor.data_ptr()))
				context.execute_async_v3(stream_handle)
			else:
				bindings = [0] * engine.num_bindings
				bindings[int(left_binding["index"])] = int(left_pad.data_ptr())
				bindings[int(right_binding["index"])] = int(right_pad.data_ptr())
				for name, tensor in out_tensors.items():
					bindings[int(self.trt_bindings[name]["index"])] = int(tensor.data_ptr())
				context.execute_async_v2(bindings=bindings, stream_handle=stream_handle)
		ender.record()
		torch.cuda.synchronize()
		runtime_ms = starter.elapsed_time(ender) / n_repeat

		return (
			out_tensors[self.trt_output_names[0]],
			out_tensors[self.trt_output_names[1]],
			out_tensors[self.trt_output_names[2]],
			runtime_ms,
		)

	# ------------------------------------------------------------------
	# 初期化ヘルパー
	# ------------------------------------------------------------------

	def _load_trt_engine(self, engine_path: Path) -> None:
		"""TensorRTエンジンをロードして実行コンテキストを準備する。"""
		if not engine_path.exists():
			raise FileNotFoundError(f"Engine file not found: {engine_path}")
		if self.device.type != "cuda":
			raise RuntimeError("TensorRT inference requires CUDA device")

		try:
			import tensorrt as trt
		except Exception as exc:  # pragma: no cover - optional dependency
			raise RuntimeError("TensorRT is not available. Please install TensorRT Python bindings.") from exc

		logger.info("Loading TensorRT engine: %s", engine_path.name)
		trt_logger = trt.Logger(trt.Logger.INFO if self.verbose else trt.Logger.WARNING)
		with open(engine_path, "rb") as f, trt.Runtime(trt_logger) as runtime:
			engine = runtime.deserialize_cuda_engine(f.read())
		if engine is None:
			raise RuntimeError(f"Failed to deserialize TensorRT engine: {engine_path}")

		self.trt_engine = engine
		self.trt_context = engine.create_execution_context()
		self._prepare_trt_bindings(trt)

	def _prepare_trt_bindings(self, trt) -> None:
		"""TensorRTの入出力バインディング情報を解析して保持する。"""
		if self.trt_engine is None:
			raise RuntimeError("TensorRT engine is not initialized")

		engine = self.trt_engine
		use_new_api = hasattr(engine, "num_io_tensors")
		bindings: dict[str, dict[str, object]] = {}
		if use_new_api:
			for i in range(engine.num_io_tensors):
				name = engine.get_tensor_name(i)
				bindings[name] = {
					"index": i,
					"shape": tuple(engine.get_tensor_shape(name)),
					"dtype": trt.nptype(engine.get_tensor_dtype(name)),
					"is_input": engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT,
				}
		else:
			for i in range(engine.num_bindings):
				name = engine.get_binding_name(i)
				bindings[name] = {
					"index": i,
					"shape": tuple(engine.get_binding_shape(i)),
					"dtype": trt.nptype(engine.get_binding_dtype(i)),
					"is_input": engine.binding_is_input(i),
				}

		input_candidates = ["input_left", "left", "input0", "input"]
		input_candidates_right = ["input_right", "right", "input1"]
		output_candidates = [
			("output_disp", "disparity", "disp", "output0"),
			("output_occ", "occlusion", "occ", "output1"),
			("output_conf", "confidence", "conf", "output2"),
		]

		input_left = next((n for n in input_candidates if n in bindings), None)
		input_right = next((n for n in input_candidates_right if n in bindings), None)
		if input_left is None or input_right is None:
			raise RuntimeError(f"TensorRT input names not found. Available={sorted(bindings)}")

		outputs: list[str] = []
		for candidates in output_candidates:
			found = next((n for n in candidates if n in bindings), None)
			if found is None:
				raise RuntimeError(f"TensorRT output names not found. Available={sorted(bindings)}")
			outputs.append(found)

		self.trt_bindings = bindings
		self.trt_input_names = [input_left, input_right]
		self.trt_output_names = outputs

	def _log_init(self) -> None:
		"""初期化時の状態をログ出力する。"""
		logger.info(
			"S2M2TensorRTBackend init: model_path=%s device=%s",
			self.model_path,
			self.device,
		)


# ===========================================================================
# Factoryクラス
# ===========================================================================
class S2M2Wrapper:
	"""明示的なバックエンド生成を行うファクトリクラス。"""

	@staticmethod
	def from_pytorch(
		model_path: str | Path,
		device: str = "cuda",
		use_positivity: bool = False,
		refine_iter: int = 3,
		amp: bool = True,
		torch_compile: bool = False,
		verbose: bool = False,
	) -> S2M2PyTorchBackend:
		"""PyTorchバックエンドを明示的な引数で生成する。"""
		return S2M2PyTorchBackend(
			model_path=Path(model_path),
			device=device,
			use_positivity=use_positivity,
			refine_iter=refine_iter,
			amp=amp,
			torch_compile=torch_compile,
			verbose=verbose,
		)

	@staticmethod
	def from_tensorrt(
		model_path: str | Path,
		device: str = "cuda",
		verbose: bool = False,
	) -> S2M2TensorRTBackend:
		"""TensorRTバックエンドを最小限の引数で生成する。"""
		return S2M2TensorRTBackend(
			model_path=Path(model_path),
			device=device,
			verbose=verbose,
		)

	@staticmethod
	def from_config(config: "S2M2Config") -> S2M2Backend:
		"""（おまけ）既存のConfigオブジェクトからも生成できるように残しておく"""
		path = Path(config.model_path)
		if path.suffix.lower() == ".pth":
			return S2M2Wrapper.from_pytorch(
				model_path=path,
				device=config.device,
				use_positivity=config.use_positivity,
				refine_iter=config.refine_iter,
				amp=config.amp,
				torch_compile=config.torch_compile,
				verbose=config.verbose,
			)
		else:
			return S2M2Wrapper.from_tensorrt(path, device=config.device, verbose=config.verbose)