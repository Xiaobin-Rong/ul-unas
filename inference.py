
import torch
import os
import sys
import soundfile as sf
from tqdm import tqdm

sys.path.append("models")
from ulunas import ULUNAS

model_ckpt_path = "./checkpoints/model_trained_on_dns3.tar"



def inference_file(input_file, output_file, model):
    """
    Run inference on a single audio file and save the result.
    Args:
        input_file (str): Path to input audio file.
        output_file (str): Path to save enhanced audio.
        model: Initialized model for inference.
    """
    audio, fs = sf.read(input_file, dtype='float32')
    input_tensor = torch.FloatTensor(audio).unsqueeze(
        0).to(next(model.parameters()).device)
    with torch.inference_mode():
        output = model(input_tensor)
    enhanced = output.cpu().detach().numpy().squeeze()
    sf.write(output_file, enhanced, fs)


def inference_folder(input_dir, output_dir, model, extension='.wav'):
    """
    Run inference on all audio files in a folder and save results to output_dir.
    Args:
        input_dir (str): Directory with input audio files.
        output_dir (str): Directory to save enhanced files.
        model: Initialized model for inference.
        extension (str): File extension to filter (default: '.wav').
    """
    os.makedirs(output_dir, exist_ok=True)
    for fname in tqdm(os.listdir(input_dir)):
        if fname.lower().endswith(extension):
            in_path = os.path.join(input_dir, fname)
            out_path = os.path.join(output_dir, fname)
            inference_file(in_path, out_path, model)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Run ULUNAS inference on audio files.")
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Input directory with audio files')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for enhanced files')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Torch device (default: cuda:0)')
    parser.add_argument('--extension', type=str, default='.wav',
                        help='Audio file extension (default: .wav)')
    args = parser.parse_args()

    device = torch.device(args.device)
    model = ULUNAS().to(device).eval()
    model.load_state_dict(
        torch.load(model_ckpt_path, map_location=args.device)['model']
    )

    inference_folder(args.input_dir, args.output_dir,
                     model, extension=args.extension)