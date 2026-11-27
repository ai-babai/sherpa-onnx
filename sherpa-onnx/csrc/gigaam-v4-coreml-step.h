#ifndef SHERPA_ONNX_CSRC_GIGAAM_V4_COREML_STEP_H_
#define SHERPA_ONNX_CSRC_GIGAAM_V4_COREML_STEP_H_

#include <memory>
#include <string>
#include <vector>

#include "onnxruntime_cxx_api.h"

namespace sherpa_onnx {

class GigaAmCoreMlStep {
 public:
  explicit GigaAmCoreMlStep(const std::string &compiled_path);
  ~GigaAmCoreMlStep();

  std::vector<Ort::Value> ZeroState(size_t start) const;
  std::vector<Ort::Value> Run(const std::vector<Ort::Value> &inputs) const;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_GIGAAM_V4_COREML_STEP_H_
