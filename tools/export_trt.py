import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import tensorrt as trt
import torch

# プロジェクトルートの取得（スクリプトの場所から2階層上）
PROJECT_ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

@dataclass(frozen=True)
class ExportTRTConfig:
    # -------------------------------
    # モデル設定
    # -------------------------------
    model_type: Literal["S", "M", "L", "XL"] = "L"
    num_refine: int = 3
    allow_negative: bool = False

    # -------------------------------
    # 入力サイズ
    # 画像のサイズが大きすぎるとなぜかCUDAのツールが対応していないらしくTensorRTのビルドが失敗する
    # -------------------------------
    img_width: int = 1280
    img_height: int = 736

    # -------------------------------
    # TensorRT設定
    # -------------------------------
    precision: Literal["fp16", "tf32", "fp32"] = "fp16"

    # -------------------------------
    # パス設定（すべて weights フォルダ直下）
    # -------------------------------
    # 読み込み元（pthファイル等がある場所）
    weight_load_dir: Path = PROJECT_ROOT / "weights"
    # 出力先（onnx, engine を置く場所）
    output_dir: Path = PROJECT_ROOT / "weights"
    
    force_onnx: bool = False


CONFIG = ExportTRTConfig()


def build_engine(onnx_file_path: Path, engine_file_path: Path, precision: str) -> None:
    """TensorRT Python API を使って ONNX から TensorRT エンジンをビルドする（trtexec 不要）"""
    logger = trt.Logger(trt.Logger.INFO)

    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser  = trt.OnnxParser(network, logger)
    config  = builder.create_builder_config()

    # ONNXファイルの読み込み・パース
    onnx_file_path_str = str(onnx_file_path.absolute())
    if not parser.parse_from_file(onnx_file_path_str):
        for error in range(parser.num_errors):
            print(parser.get_error(error))
        raise RuntimeError(f"Failed to parse ONNX file: {onnx_file_path_str}")

    # 精度設定
    if precision == "fp16":
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            # trtexec --precisionConstraints=obey 相当
            config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
            # trtexec --layerPrecisions=node_linalg_vector_norm_2:fp32 相当
            for i in range(network.num_layers):
                layer = network.get_layer(i)
                if "node_linalg_vector_norm_2" in layer.name:
                    layer.precision = trt.float32
                    layer.set_output_type(0, trt.float32)
        else:
            print("Warning: This platform does not support fast FP16. Falling back to FP32.")
    elif precision == "tf32":
        config.set_flag(trt.BuilderFlag.TF32)
    elif precision == "fp32":
        # trtexec --noTF32 相当
        config.clear_flag(trt.BuilderFlag.TF32)

    print(f"Building engine: {engine_file_path}")
    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        raise RuntimeError("Failed to build TensorRT engine. Check the ONNX model and precision settings.")

    with open(engine_file_path, "wb") as f:
        f.write(serialized_engine)
    print(f"Build complete! Saved to: {engine_file_path}")


def export_onnx_from_pth(onnx_path: Path) -> None:
    """PyTorchモデルをロードしてONNXにエクスポートする"""
    from src.models.s2m2.src.s2m2.core.utils.model_utils import load_model
    from src.models.s2m2.src.s2m2.tools.export_model import export_onnx

    # CPUでモデルをロード
    model = load_model(
        str(CONFIG.weight_load_dir), 
        CONFIG.model_type, 
        not CONFIG.allow_negative, 
        CONFIG.num_refine, 
        "cpu"
    )
    if model is None:
        raise RuntimeError("Failed to load S2M2 model")

    # ダミー入力の作成
    left_torch = torch.zeros((1, 3, CONFIG.img_height, CONFIG.img_width), dtype=torch.float32)
    right_torch = torch.zeros((1, 3, CONFIG.img_height, CONFIG.img_width), dtype=torch.float32)

    print(f"Exporting ONNX: {onnx_path}")
    export_onnx(model, str(onnx_path), left_torch, right_torch)


def main() -> None:
    # 出力先ディレクトリの確保
    CONFIG.output_dir.mkdir(parents=True, exist_ok=True)

    v = torch.__version__
    parts = v.split('.')
    # ONNXファイル名（サンプルに準拠）
    onnx_path = CONFIG.output_dir / f"{CONFIG.model_type}_{CONFIG.img_width}_{CONFIG.img_height}_torch{parts[0]}{parts[1]}.onnx"
    # Engineファイル名
    engine_path = CONFIG.output_dir / f"{CONFIG.model_type}_{CONFIG.img_width}_{CONFIG.img_height}_{CONFIG.precision}.engine"

    # 1. ONNXエクスポート（存在しない場合、または強制フラグ時）
    if CONFIG.force_onnx or not onnx_path.exists():
        export_onnx_from_pth(onnx_path)
    else:
        print(f"ONNX exists, skip export: {onnx_path}")

    # 2. TensorRTエンジンビルド
    build_engine(onnx_path, engine_path, CONFIG.precision)


if __name__ == "__main__":
    main()