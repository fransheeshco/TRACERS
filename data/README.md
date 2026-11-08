# Local data

This directory is reserved for local datasets and generated artifacts:

```text
data/
├── raw/          # Source audio and charts
├── processed/    # Preprocessed PyTorch tensors
└── checkpoints/  # Trained model weights
```

The contents of those directories are intentionally ignored by Git. Audio may
be copyrighted, while tensors and checkpoints can be too large for an ordinary
Git repository. Create the directories locally when needed.
