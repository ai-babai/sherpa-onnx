// sherpa-onnx/csrc/gigaam-coreml-encoder.h
//
// Copyright (c)  2026  sherpa-onnx contributors

#ifndef SHERPA_ONNX_CSRC_GIGAAM_COREML_ENCODER_H_
#define SHERPA_ONNX_CSRC_GIGAAM_COREML_ENCODER_H_

#include <array>
#include <cstdint>
#include <memory>
#include <string>

#include "onnxruntime_cxx_api.h"  // NOLINT

namespace sherpa_onnx {

class GigaAmCoremlCtc {
 public:
  static constexpr int64_t kChannels = 64;
  static constexpr int64_t kVocabulary = 146;
  static constexpr int64_t kSubsampling = 4;

  GigaAmCoremlCtc(const std::string &path, const std::string &provider);
  ~GigaAmCoremlCtc();

  int64_t Frames() const;
  Ort::Value Forward(const Ort::Value &features, int64_t frames,
                     OrtAllocator *allocator) const;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

class GigaAmCoremlRnnt {
 public:
  GigaAmCoremlRnnt(const std::string &path, const std::string &provider);
  ~GigaAmCoremlRnnt();

  bool HasLanguageInput() const;
  std::array<Ort::Value, 2> Forward(const float *features, int64_t frames,
                                    const float *language,
                                    OrtAllocator *allocator) const;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_GIGAAM_COREML_ENCODER_H_
