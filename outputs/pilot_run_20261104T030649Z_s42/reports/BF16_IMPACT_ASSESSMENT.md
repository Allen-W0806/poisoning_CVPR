# Precision and numerical assessment

Executed precision: fp32. LoRA storage/linear compute and Adam moments are FP32.

Full FP32 is an isolated alternative, not proof that the existing BF16 ordinary-path issue is repaired. Casting adapters alone was already present in historical failing experiments. No upstream CUDA root cause is claimed.

Ordinary repeat/save-reload-next-update witness:

```json
{
  "status": "PASS",
  "precision": "fp32",
  "actual_optimizer_updates": 5,
  "checks": {
    "ordinary_first_update": {
      "exact": true,
      "tolerance": true,
      "tensor_count": 1538,
      "different_elements": 0,
      "max_absolute_error": 0.0,
      "structure_errors": []
    },
    "ordinary_second_update": {
      "exact": true,
      "tolerance": true,
      "tensor_count": 1538,
      "different_elements": 0,
      "max_absolute_error": 0.0,
      "structure_errors": []
    },
    "reload_completed_next_update": {
      "exact": true,
      "tolerance": true,
      "tensor_count": 1538,
      "different_elements": 0,
      "max_absolute_error": 0.0,
      "structure_errors": []
    }
  },
  "BF16_fix_claim": false,
  "protected_frozen_weights_unchanged": true
}
```

A passing witness covers this recorded model, source images, precision, two-update sequence and node. It does not prove all-run reproducibility or cross-hardware equality. Nonfinite values and replay failures stop training; raw gradients, postclip gradients, adapter and Adam states are retained for the witness.
