"""
Convert the trained AI SpillGuard PyTorch model (U-Net + ResNet34) to ONNX.

This produces a browser-runnable model so inference can run client-side via
ONNX Runtime Web on a static Vercel deployment (no Python backend needed).

Output: Vercel_version/model/spillguard.onnx
        Vercel_version/model/spillguard.int8.onnx (quantized, smaller — shipped)

Run from the project root with the venv python:
    venv/Scripts/python.exe Vercel_version/tools/convert_to_onnx.py
"""

import os
import torch
import numpy as np
import segmentation_models_pytorch as smp

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
OUT_DIR = os.path.join(HERE, "..", "model")
os.makedirs(OUT_DIR, exist_ok=True)

IMG_SIZE = 256
NUM_CLASSES = 4


def find_model_path():
    candidates = [
        os.path.join(PROJECT_ROOT, "best_model.pth"),
        os.path.join(PROJECT_ROOT, "checkpoints", "best_model.pth"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    raise FileNotFoundError("No best_model.pth found in project root or checkpoints/")


def load_model(checkpoint_path):
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=3,
        classes=NUM_CLASSES,
        activation=None,
    )
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        model.load_state_dict(ckpt["state_dict"])
    else:
        model.load_state_dict(ckpt)
    model.eval()
    return model


def main():
    model_path = find_model_path()
    print(f"Loading model from: {model_path}")
    model = load_model(model_path)

    dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE, dtype=torch.float32)
    fp32_path = os.path.join(OUT_DIR, "spillguard.onnx")

    print("Exporting to ONNX (fp32)...")
    torch.onnx.export(
        model,
        dummy,
        fp32_path,
        input_names=["input"],
        output_names=["logits"],
        opset_version=17,
        dynamic_axes=None,  # fixed 1x3x256x256 for best ORT-Web perf
        do_constant_folding=True,
        dynamo=False,  # use the stable TorchScript exporter
    )
    print(f"  -> {fp32_path} ({os.path.getsize(fp32_path)/1e6:.1f} MB)")

    # Sanity check parity between torch and onnxruntime
    try:
        import onnxruntime as ort

        sess = ort.InferenceSession(fp32_path, providers=["CPUExecutionProvider"])
        with torch.inference_mode():
            torch_out = model(dummy).numpy()
        onnx_out = sess.run(None, {"input": dummy.numpy()})[0]
        max_diff = np.abs(torch_out - onnx_out).max()
        torch_arg = torch_out.argmax(1)
        onnx_arg = onnx_out.argmax(1)
        agree = (torch_arg == onnx_arg).mean() * 100
        print(f"  parity: max logit diff={max_diff:.2e}, argmax agreement={agree:.3f}%")
    except Exception as e:
        print(f"  (parity check skipped: {e})")

    # Quantize to int8 for a much smaller browser download
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType

        int8_path = os.path.join(OUT_DIR, "spillguard.int8.onnx")
        print("Quantizing (dynamic int8)...")
        quantize_dynamic(
            fp32_path,
            int8_path,
            weight_type=QuantType.QUInt8,
        )
        print(f"  -> {int8_path} ({os.path.getsize(int8_path)/1e6:.1f} MB)")
    except Exception as e:
        print(f"  (quantization skipped: {e})")

    print("Done.")


if __name__ == "__main__":
    main()
