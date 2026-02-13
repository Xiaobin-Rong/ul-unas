# UL-UNAS
[![arxiv](https://img.shields.io/badge/arXiv-Paper-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2503.00340)
[![demo](https://img.shields.io/badge/GitHub-Demo-orange.svg)](https://xiaobin-rong.github.io/ul-unas_demo/)

🎉 This is the official implementation of our *IEEE TASLP* paper: 

[UL-UNAS: Ultra-Lightweight U-Nets for Real-Time Speech Enhancement via Network Architecture Search](https://arxiv.org/abs/2503.00340)

## 🔥 News
- [**2026-2-13**] Added streaming inference code `ulunas_onnx` implemented by [Kailai Shen](https://github.com/Shenkailai).
- [**2026-2-3**] The updated paper is uploaded to arxiv.
- [**2026-2-1**] The pre-trained checkpoint is released.
- [**2026-1-28**] The model implementation is released.

## Inference
To run inference on audio files, use:

```bash
python inference --input_dir <input_dir> --output_dir <output_dir> [options]
```

| Argument       | Requirement / Default | Description                                                  |
|----------------|-----------------------|--------------------------------------------------------------|
| `--input_dir`  | **required**          | Path to the input directory containing audio files.          |
| `--output_dir` | **required**          | Path to the output directory where enhanced files will be saved. |
| `--device`     | default: `cuda:0`     | Torch device to run inference on, e.g., `cuda:0`, `cuda:1`, or `cpu`. |
| `--extension`  | default: `.wav`       | Audio file extension to process.                             |

## Citation
If you find this work useful, please cite our paper:
```bibtex
@misc{rong2025ulunas,
      title={UL-UNAS: Ultra-Lightweight U-Nets for Real-Time Speech Enhancement via Network Architecture Search}, 
      author={Xiaobin Rong and Dahan Wang and Yuxiang Hu and Changbao Zhu and Kai Chen and Jing Lu},
      year={2025},
      eprint={2503.00340},
      archivePrefix={arXiv},
      primaryClass={eess.AS},
      url={https://arxiv.org/abs/2503.00340}, 
}
```

## Contact
Xiaobin Rong: [xiaobin.rong@smail.nju.edu.cn](mailto:xiaobin.rong@smail.nju.edu.cn)
