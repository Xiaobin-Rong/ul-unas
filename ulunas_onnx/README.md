# ULUNAS 流式 ONNX 推理

## 运行

在项目根目录执行：

```bash
python ulunas_onnx/stream/ulunas_stream.py
```

会生成：
- `ulunas_onnx/onnx_models/ulunas_stream.onnx`
- `ulunas_onnx/onnx_models/ulunas_stream_simple.onnx`
- `audio/enh_stream.wav`
- `audio/enh_onnx.wav`

## ONNX 输入输出

输入：
- `mix`: `[B, 257, 1, 2]`
- `conv_cache`: `[B, 5358]`
- `tfa_cache`: `[B, 402]`
- `inter_cache`: `[B, 1056]`

输出：
- `enh`: `[B, 257, 1, 2]`
- `conv_cache_out`: `[B, 5358]`
- `tfa_cache_out`: `[B, 402]`
- `inter_cache_out`: `[B, 1056]`

说明：按帧输入（`T=1`），每帧将输出 cache 作为下一帧输入。

## ONNXRuntime 示例

```python
import numpy as np
import onnxruntime as ort
import soundfile as sf
import torch

session = ort.InferenceSession(
    "ulunas_onnx/onnx_models/ulunas_stream_simple.onnx",
    providers=["CPUExecutionProvider"],
)

# 读取音频
wav, sr = sf.read("audio/noisy/0174.wav", dtype="float32")
assert sr == 16000
x = torch.from_numpy(wav)[None]  # [1, N]

# STFT -> [1, 257, T, 2]
window = torch.hann_window(512)
spec_c = torch.stft(
    x, n_fft=512, hop_length=256, win_length=512, window=window, return_complex=True
)
spec = torch.view_as_real(spec_c).numpy()

conv_cache = np.zeros((1, 5358), dtype=np.float32)
tfa_cache = np.zeros((1, 402), dtype=np.float32)
inter_cache = np.zeros((1, 1056), dtype=np.float32)

# 逐帧流式推理（T=1）
outs = []
for i in range(spec.shape[2]):
    out_i, conv_cache, tfa_cache, inter_cache = session.run(
        [],
        {
            "mix": spec[:, :, i : i + 1, :],
            "conv_cache": conv_cache,
            "tfa_cache": tfa_cache,
            "inter_cache": inter_cache,
        },
    )
    outs.append(out_i)

# 拼回频谱并 ISTFT
enh_spec = np.concatenate(outs, axis=2)
enh_spec_t = torch.from_numpy(enh_spec)
enh_spec_c = torch.complex(enh_spec_t[..., 0], enh_spec_t[..., 1])
enh_wav = torch.istft(
    enh_spec_c[0],
    n_fft=512,
    hop_length=256,
    win_length=512,
    window=window,
    onesided=True,
    length=x.shape[1],
).numpy()

sf.write("audio/enh_onnx_runtime.wav", enh_wav, 16000)
```
