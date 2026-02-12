import os
import argparse
import tensorrt as trt

# プロジェクトのルートディレクトリ設定
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, default='S')
    parser.add_argument('--img_width', type=int, default=800)
    parser.add_argument('--img_height', type=int, default=1088)
    parser.add_argument('--precision', type=str, choices=['fp16', 'tf32', 'fp32'], default='fp16')
    return parser

def build_engine(onnx_file_path, engine_file_path, precision):
    # TensorRTのロガー初期化
    logger = trt.Logger(trt.Logger.INFO)
    
    # ビルダー、ネットワーク、パーサーの作成
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    config = builder.create_builder_config()

    # ONNXファイルの読み込み
    with open(onnx_file_path, 'rb') as model:
        if not parser.parse(model.read()):
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return None

    # 精度（Precision）の設定
    if precision == 'fp16':
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            # 特定レイヤーの精度固定（trtexecのprecisionConstraints相当）
            config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
            # node_linalg_vector_norm_2 という名前のレイヤーをFP32に指定
            for i in range(network.num_layers):
                layer = network.get_layer(i)
                if "node_linalg_vector_norm_2" in layer.name:
                    layer.precision = trt.float32
                    layer.set_output_type(0, trt.float32)
        else:
            print("Warning: This platform does not support fast FP16. Falling back to FP32.")
    
    elif precision == 'tf32':
        # TensorRTはデフォルトでTF32が有効な場合が多いですが明示的に設定
        config.set_flag(trt.BuilderFlag.TF32)
    
    elif precision == 'fp32':
        # TF32を無効化して厳密なFP32にする
        config.clear_flag(trt.BuilderFlag.TF32)

    # エンジンのビルドと保存
    print(f"Building engine: {engine_file_path}")
    serialized_engine = builder.build_serialized_network(network, config)
    
    with open(engine_file_path, "wb") as f:
        f.write(serialized_engine)
    print("Build complete!")

def main(args):
    onnx_dir = os.path.join(project_root, "weights/onnx_save")
    trt_dir = os.path.join(project_root, "weights/trt_save")
    os.makedirs(trt_dir, exist_ok=True)

    # ファイルパスの構築
    # ※torchバージョンの取得は元のロジックを継承（適宜調整してください）
    import torch
    v = torch.__version__
    onnx_file_path = os.path.join(onnx_dir, f'S2M2_{args.model_type}_{args.img_width}_{args.img_height}_v2_torch{v[0]}{v[2]}.onnx')
    trt_file_path = os.path.join(trt_dir, f'S2M2_{args.model_type}_{args.img_width}_{args.img_height}_{args.precision}.engine')

    if not os.path.exists(onnx_file_path):
        print(f"Error: ONNX file not found at {onnx_file_path}")
        return

    build_engine(onnx_file_path, trt_file_path, args.precision)

if __name__ == '__main__':
    args = get_args_parser().parse_args()
    main(args)