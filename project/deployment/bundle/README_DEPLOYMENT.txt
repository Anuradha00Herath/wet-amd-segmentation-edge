OCT wetAMD Segmentation — Mobile Deployment Bundle
====================================================

Models included:
  fp32: baseline_fp32_mobile_opt.onnx
  ptq_int8: baseline_int8_static_mobile_opt.onnx
  qat_int8: qat_int8_static_mobile_opt.onnx

Android integration:
  1. Copy *.onnx files to Android/app/src/main/assets/
  2. Copy model_metadata.json to assets/
  3. Build and run the Android app
