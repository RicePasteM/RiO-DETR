# RT-DETRv2-OBB-M DOTA-v1.0 Multi-Scale Training Logs

Training completed all 102 epochs. The run was interrupted during epoch 7
because the original two-GPU allocation exhausted device memory, then resumed
from the epoch 6 checkpoint on four GPUs with the same total batch size of 32.

- [Initial run: epochs 0–6 and interrupted epoch 7](rtdetrv2_obb_hgnetv2_m_dota_1_ms_part1.log)
- [Four-GPU resume: epoch 7–101](rtdetrv2_obb_hgnetv2_m_dota_1_ms_resume.log)

The full odd-epoch evaluation record is available in the
[evaluation CSV](../eval/rtdetrv2_obb_hgnetv2_m_dota_1_ms_eval.csv). Epoch 43
was selected by the highest VOC mAP/AP50.
