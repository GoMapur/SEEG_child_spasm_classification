# SEEG Child Spasm Classification - Release

## Overview

This codebase provides tools for training, inference, and model interpretation (including occlusion-based analysis) for scalp EEG data in child spasm classification tasks.

## Requirements

- Python 3.7+
- PyTorch
- numpy, pandas, matplotlib, tqdm, natsort, scikit-learn, seaborn, cv2, captum

Install dependencies:
```sh
pip install -r requirements.txt
```

## Data Preparation

- Prepare your EEG data in `.npz` format.
- Create a metadata JSON file describing patients, recordings, and splits (see `src/dataloaders/scalp_eeg_fft_metadata.py` for format).

## Training

To train a model (example):
```sh
python train.py -c config/train_config.ini
```

## Inference

To run inference on test data:
```sh
python inference.py --fold_id 0 --metadata_path path/to/metadata.json --data_dir path/to/data --checkpoint_path path/to/model_checkpoint.pth --output_dir path/to/output_dir
```

## Model Interpretation (Occlusion Analysis)

To run frequency patient-wise occlusion analysis:
```sh
python interpret_freq.py \
  --metadata_path path/to/metadata.json \
  --data_dir path/to/data \
  --save_path path/to/save_dir \
  --ckpt_dir path/to/ckpt_dir \
  --log_dir path/to/log_dir \
  --experiment_name 22-12-03_FFT_5fold_balanced_Sleep_butter \
  --device cuda:0 \
  --threshold 0.5 \
  --wanted_test_set 0-49 \
  --split test
```

This will:
- Run occlusion analysis for each fold and patient.
- Save results and group-wise mean/std occlusion plots in `save_path`.

## Notes

- All configuration is now handled via command-line arguments.
- Ensure all paths and experiment names are set correctly before running.
- See code comments for further customization options.

## License

[Your License Here]
