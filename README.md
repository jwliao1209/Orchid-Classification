# Orchid_Classification

| Checkpoint      | Model           | Batch size | Epochs | Loss | Optimizer                           | Scheduler                | Best val acc |
| --------------- | --------------- | ---------- | ------ | ---- | ------------------------------------| ------------------------ | ------------ |
| 04-11-00-15-58  | EfficientNet-b4 | 64         | 100    | CE   | AdamW (lr=1e-3,  weight decay=1e-4) | Step (size=3, gamma=0.8) | 82.19 (75)   |
