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
from src.models.s2m2.src.s2m2.core.model.s2m2 import S2M2
from src.models.s2m2.src.s2m2.core.utils.image_utils import image_crop, image_pad
from src.utils.geometry import Size

if TYPE_CHECKING:
	from src.config_loader.s2m2_config import S2M2Config

logger = get_logger(__name__)


@dataclass(frozen=True)
class S2M2Result:
	"""S2M2（Stereo Matching）推論結果。"""
	disparity: np.ndarray
	occlusion: np.ndarray
	confidence: np.ndarray
	avg_confidence: float
	runtime_ms: float
	orig_shape: Size
	used_shape: Size


class S2M2Wrapper(ABC):
	"""S2M2推論の共通インターフェース。"""

	@abstractmethod
	def _predict_processed(
		self,
		left_torch: torch.Tensor,
		right_torch: torch.Tensor,
		*,
		n_repeat: int,
	) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
		"""前処理済みテンソルから推論し、整形済み結果を返す。"""
		...

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
		pred_disp, pred_occ, pred_conf, avg_conf, runtime_ms = self._predict_processed(
			left_torch,
			right_torch,
			n_repeat=self._normalize_repeat(n_repeat),
		)
		return self._build_result(
			pred_disp,
			pred_occ,
			pred_conf,
			avg_conf,
			runtime_ms,
			orig_h=orig_h,
			orig_w=orig_w,
			used_h=used_h,
			used_w=used_w,
		)

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
		"""入力チェック・サイズ調整・Tensor変換を行う共通前処理。"""
		if left is None or right is None:
			raise ValueError("left/right is None")
		if left.shape[:2] != right.shape[:2]:
			raise ValueError(f"left/right shape mismatch: left={left.shape} right={right.shape}")
		if left.ndim != 3 or right.ndim != 3:
			raise ValueError("left/right must be HxWxC images")

		orig_h, orig_w = left.shape[:2]
		if crop_to_multiple_of_32:
			used_h = (orig_h // 32) * 32
			used_w = (orig_w // 32) * 32
			if used_h <= 0 or used_w <= 0:
				raise ValueError(f"Image too small for 32x crop: {left.shape}")
			left = left[:used_h, :used_w]
			right = right[:used_h, :used_w]
		else:
			used_h, used_w = orig_h, orig_w

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

	@staticmethod
	def _normalize_repeat(n_repeat: int) -> int:
		return max(1, int(n_repeat))

	@staticmethod
	def _build_result(
		disparity: np.ndarray,
		occlusion: np.ndarray,
		confidence: np.ndarray,
		avg_confidence: float,
		runtime_ms: float,
		*,
		orig_h: int,
		orig_w: int,
		used_h: int,
		used_w: int,
	) -> S2M2Result:
		return S2M2Result(
			disparity=disparity,
			occlusion=occlusion,
			confidence=confidence,
			avg_confidence=avg_confidence,
			runtime_ms=runtime_ms,
			orig_shape=Size(width=float(orig_w), height=float(orig_h)),
			used_shape=Size(width=float(used_w), height=float(used_h)),
		)


class S2M2PyTorchWrapper(S2M2Wrapper):
	"""PyTorch（.pth）モデルを使った S2M2 推論実装。"""

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

	@torch.no_grad()
	def _predict_processed(
		self,
		left_torch: torch.Tensor,
		right_torch: torch.Tensor,
		*,
		n_repeat: int,
	) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
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
		cuda_available = torch.cuda.is_available()
		cuda_device_count = torch.cuda.device_count() if cuda_available else 0
		logger.info(
			"S2M2PyTorchWrapper init: model_path=%s device=%s use_positivity=%s "
			"refine_iter=%s amp=%s torch_compile=%s",
			self.model_path,
			self.device,
			self.use_positivity,
			self.refine_iter,
			self.amp,
			self.torch_compile,
		)
		logger.info(
			"S2M2PyTorchWrapper device status: cuda_available=%s device_count=%s",
			cuda_available,
			cuda_device_count,
		)


class S2M2TensorRTWrapper(S2M2Wrapper):
	"""TensorRT（.engine）エンジンを使った S2M2 推論実装。"""

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

	def _predict_processed(
		self,
		left_torch: torch.Tensor,
		right_torch: torch.Tensor,
		*,
		n_repeat: int,
	) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
		img_height, img_width = left_torch.shape[-2:]
		left_pad = image_pad(left_torch, 32).to(self.device)
		right_pad = image_pad(right_torch, 32).to(self.device)

		pred_disp_t, pred_occ_t, pred_conf_t, runtime_ms = self._run_trt_inference(
			left_pad, right_pad, n_repeat=n_repeat
		)
		return self._post_process(
			pred_disp_t, pred_occ_t, pred_conf_t, img_height, img_width, runtime_ms
		)

	def _run_trt_inference(
		self,
		left_pad: torch.Tensor,
		right_pad: torch.Tensor,
		*,
		n_repeat: int,
	) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
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

	def _load_trt_engine(self, engine_path: Path) -> None:
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
		logger.info(
			"S2M2TensorRTWrapper init: model_path=%s device=%s",
			self.model_path,
			self.device,
		)


def create_s2m2_wrapper(config: "S2M2Config") -> S2M2Wrapper:
	"""S2M2Config から適切な推論実装を生成する。"""
	path = Path(config.model_path)
	suffix = path.suffix.lower()

	if suffix == ".pth":
		return S2M2PyTorchWrapper(
			model_path=path,
			device=config.device,
			use_positivity=config.use_positivity,
			refine_iter=config.refine_iter,
			amp=config.amp,
			torch_compile=config.torch_compile,
			verbose=config.verbose,
		)
	if suffix == ".engine":
		return S2M2TensorRTWrapper(
			model_path=path,
			device=config.device,
			verbose=config.verbose,
		)

	raise ValueError(
		f"Unsupported model format: '{suffix}'. "
		"Supported formats: .pth (PyTorch), .engine (TensorRT)"
	)
