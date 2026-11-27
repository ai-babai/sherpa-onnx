// sherpa-onnx/csrc/gigaam-coreml-encoder.mm
//
// Copyright (c)  2026  sherpa-onnx contributors

#include "sherpa-onnx/csrc/gigaam-coreml-encoder.h"

#import <CoreML/CoreML.h>
#import <Foundation/Foundation.h>

#include <algorithm>
#include <array>
#include <stdexcept>
#include <utility>

namespace sherpa_onnx {
namespace {

constexpr int64_t kChannels = GigaAmCoremlCtc::kChannels;

std::runtime_error CoremlError(const char *operation, NSError *error) {
  const char *reason = error.localizedDescription.UTF8String;
  return std::runtime_error(std::string(operation) + ": " +
                            (reason ? reason : "unknown Core ML error"));
}

MLModel *LoadCoreml(const std::string &path, const std::string &provider) {
  if (provider != "coreml" && provider != "cpu") {
    throw std::invalid_argument("Compiled GigaAM Core ML accepts provider=coreml or provider=cpu");
  }
  if (@available(macOS 15.0, *)) {
    NSURL *url = [NSURL fileURLWithPath:@(path.c_str()) isDirectory:YES];
    MLModelConfiguration *config = [[MLModelConfiguration alloc] init];
    config.computeUnits = provider == "coreml" ? MLComputeUnitsCPUAndGPU : MLComputeUnitsCPUOnly;
    NSError *error = nil;
    MLModel *model = [MLModel modelWithContentsOfURL:url configuration:config error:&error];
    if (!model) throw CoremlError("Load compiled GigaAM encoder", error);
    return model;
  }
  throw std::runtime_error("Compiled GigaAM Core ML requires macOS 15");
}

int64_t InputFrames(MLModel *model) {
  auto inputs = model.modelDescription.inputDescriptionsByName;
  auto audio = inputs[@"features"].multiArrayConstraint;
  auto length = inputs[@"feature_lengths"].multiArrayConstraint;
  if (audio.dataType != MLMultiArrayDataTypeFloat32 || audio.shape.count != 3 ||
      audio.shape[0].longLongValue != 1 || audio.shape[1].longLongValue != kChannels ||
      audio.shape[2].longLongValue <= 0 || audio.shape[2].longLongValue > 3000 ||
      length.dataType != MLMultiArrayDataTypeInt32 || ![length.shape isEqualToArray:@[ @1 ]]) {
    throw std::invalid_argument("Unexpected GigaAM Core ML feature signature");
  }
  return audio.shape[2].longLongValue;
}

NSString *ModelType(MLModel *model) {
  NSDictionary *metadata = model.modelDescription.metadata[MLModelCreatorDefinedKey];
  id type = metadata[@"gigaam_model_type"];
  return [type isKindOfClass:NSString.class] ? type : nil;
}

MLMultiArray *MakeArray(NSArray<NSNumber *> *shape, MLMultiArrayDataType type) {
  NSError *error = nil;
  MLMultiArray *value = [[MLMultiArray alloc] initWithShape:shape dataType:type error:&error];
  if (!value) throw CoremlError("Allocate GigaAM features", error);
  return value;
}

id<MLFeatureProvider> Predict(MLModel *model, const float *features, int64_t frames, int64_t stride,
                              int64_t capacity, const float *language) {
  if (frames < 0 || frames > capacity) {
    throw std::invalid_argument("Audio exceeds the compiled GigaAM frame limit");
  }
  auto audio = MakeArray(@[ @1, @(kChannels), @(capacity) ], MLMultiArrayDataTypeFloat32);
  auto *samples = static_cast<float *>(audio.dataPointer);
  for (int64_t channel = 0; channel < kChannels; ++channel) {
    if (frames) std::copy_n(features + channel * stride, frames, samples);
    std::fill(samples + frames, samples + capacity, 0.0f);
    samples += capacity;
  }
  auto length = MakeArray(@[ @1 ], MLMultiArrayDataTypeInt32);
  *static_cast<int32_t *>(length.dataPointer) = static_cast<int32_t>(frames);
  NSMutableDictionary<NSString *, MLFeatureValue *> *values = [@{
    @"features" : [MLFeatureValue featureValueWithMultiArray:audio],
    @"feature_lengths" : [MLFeatureValue featureValueWithMultiArray:length]
  } mutableCopy];
  if (language) {
    auto vector = MakeArray(@[ @1, @32 ], MLMultiArrayDataTypeFloat32);
    std::copy_n(language, 32, static_cast<float *>(vector.dataPointer));
    values[@"language_vector"] = [MLFeatureValue featureValueWithMultiArray:vector];
  }
  NSError *error = nil;
  MLDictionaryFeatureProvider *inputs =
      [[MLDictionaryFeatureProvider alloc] initWithDictionary:values error:&error];
  if (!inputs) throw CoremlError("Prepare GigaAM encoder inputs", error);
  id<MLFeatureProvider> output = [model predictionFromFeatures:inputs error:&error];
  if (!output) throw CoremlError("Predict GigaAM encoder", error);
  return output;
}

}  // namespace

class GigaAmCoremlCtc::Impl {
 public:
  Impl(const std::string &path, const std::string &provider) {
    @autoreleasepool {
      model_ = LoadCoreml(path, provider);
      if (![ModelType(model_) isEqualToString:@"v4-ctc-e2e"]) {
        throw std::invalid_argument("Unexpected GigaAM Core ML CTC model signature");
      }
      frames_ = InputFrames(model_);
    }
  }

  int64_t Frames() const { return frames_; }

  Ort::Value Forward(const Ort::Value &features, int64_t frames, OrtAllocator *allocator) const {
    @autoreleasepool {
      auto output =
          Predict(model_, features.GetTensorData<float>(), frames, frames_, frames_, nullptr);
      return CopyLogProbs([output featureValueForName:@"log_probs"].multiArrayValue, frames,
                          allocator);
    }
  }

 private:
  static Ort::Value CopyLogProbs(MLMultiArray *log_probs, int64_t frames, OrtAllocator *allocator) {
    const int64_t rows = frames / kSubsampling;
    if (log_probs.dataType != MLMultiArrayDataTypeFloat32 || log_probs.shape.count != 3 ||
        log_probs.shape[0].longLongValue != 1 || log_probs.shape[1].longLongValue < rows ||
        log_probs.shape[2].longLongValue != kVocabulary) {
      throw std::runtime_error("Unexpected GigaAM Core ML logits shape or type");
    }
    const std::array<int64_t, 3> shape{1, rows, kVocabulary};
    auto result = Ort::Value::CreateTensor<float>(allocator, shape.data(), shape.size());
    const auto *src = static_cast<const float *>(log_probs.dataPointer);
    auto *dst = result.GetTensorMutableData<float>();
    const auto time_stride = log_probs.strides[1].longLongValue;
    const auto token_stride = log_probs.strides[2].longLongValue;
    for (int64_t t = 0; t < rows; ++t) {
      for (int64_t k = 0; k < kVocabulary; ++k) {
        *dst++ = src[t * time_stride + k * token_stride];
      }
    }
    return result;
  }

  MLModel *model_ = nil;
  int64_t frames_ = 0;
};

GigaAmCoremlCtc::GigaAmCoremlCtc(const std::string &path, const std::string &provider)
    : impl_(std::make_unique<Impl>(path, provider)) {}

GigaAmCoremlCtc::~GigaAmCoremlCtc() = default;

int64_t GigaAmCoremlCtc::Frames() const { return impl_->Frames(); }

Ort::Value GigaAmCoremlCtc::Forward(const Ort::Value &features, int64_t frames,
                                    OrtAllocator *allocator) const {
  return impl_->Forward(features, frames, allocator);
}

class GigaAmCoremlRnnt::Impl {
 public:
  Impl(const std::string &path, const std::string &provider) {
    @autoreleasepool {
      model_ = LoadCoreml(path, provider);
      auto desc = model_.modelDescription;
      NSString *type = ModelType(model_);
      has_language_ = [type isEqualToString:@"multilingual-v2-rnnt"];
      if (!has_language_ && ![type isEqualToString:@"v4-rnnt"]) {
        throw std::invalid_argument("Unexpected GigaAM Core ML RNN-T model type");
      }
      auto inputs = desc.inputDescriptionsByName;
      if (has_language_) {
        auto vector = inputs[@"language_vector"].multiArrayConstraint;
        if (vector.dataType != MLMultiArrayDataTypeFloat32 ||
            ![vector.shape isEqualToArray:@[ @1, @32 ]]) {
          throw std::invalid_argument("Unexpected GigaAM Core ML language input");
        }
      }
      if (inputs.count != (has_language_ ? 3 : 2)) {
        throw std::invalid_argument("Unexpected GigaAM Core ML RNN-T inputs");
      }
      frames_ = InputFrames(model_);
    }
  }

  bool HasLanguageInput() const { return has_language_; }

  std::array<Ort::Value, 2> Forward(const float *features, int64_t frames, const float *language,
                                    OrtAllocator *allocator) const {
    @autoreleasepool {
      auto prediction = Predict(model_, features, frames, frames, frames_, language);
      auto length = [prediction featureValueForName:@"encoded_lengths"].multiArrayValue;
      if (length.dataType != MLMultiArrayDataTypeInt32 || ![length.shape isEqualToArray:@[ @1 ]]) {
        throw std::runtime_error("Unexpected GigaAM Core ML encoded length");
      }
      const int64_t rows = *static_cast<const int32_t *>(length.dataPointer);
      constexpr std::array<int64_t, 1> length_shape{1};
      auto encoded_len =
          Ort::Value::CreateTensor<int64_t>(allocator, length_shape.data(), length_shape.size());
      *encoded_len.GetTensorMutableData<int64_t>() = rows;
      auto heads = [prediction featureValueForName:@"head_outputs"].multiArrayValue;
      return {CopyHeads(heads, rows, allocator), std::move(encoded_len)};
    }
  }

 private:
  static Ort::Value CopyHeads(MLMultiArray *heads, int64_t rows, OrtAllocator *allocator) {
    if (heads.dataType != MLMultiArrayDataTypeFloat32 || heads.shape.count != 4 ||
        heads.shape[0].longLongValue != 2 || heads.shape[1].longLongValue != 1 ||
        heads.shape[2].longLongValue < rows || heads.shape[3].longLongValue != 320 || rows < 0) {
      throw std::runtime_error("Unexpected GigaAM Core ML head output");
    }
    const std::array<int64_t, 4> shape{2, 1, rows, 320};
    auto result = Ort::Value::CreateTensor<float>(allocator, shape.data(), shape.size());
    const auto *src = static_cast<const float *>(heads.dataPointer);
    auto *dst = result.GetTensorMutableData<float>();
    const auto head_stride = heads.strides[0].longLongValue;
    const auto time_stride = heads.strides[2].longLongValue;
    const auto feature_stride = heads.strides[3].longLongValue;
    for (int64_t head = 0; head < 2; ++head) {
      for (int64_t row = 0; row < rows; ++row) {
        for (int64_t feature = 0; feature < 320; ++feature) {
          *dst++ = src[head * head_stride + row * time_stride + feature * feature_stride];
        }
      }
    }
    return result;
  }

  MLModel *model_ = nil;
  int64_t frames_ = 0;
  bool has_language_ = false;
};

GigaAmCoremlRnnt::GigaAmCoremlRnnt(const std::string &path, const std::string &provider)
    : impl_(std::make_unique<Impl>(path, provider)) {}

GigaAmCoremlRnnt::~GigaAmCoremlRnnt() = default;

bool GigaAmCoremlRnnt::HasLanguageInput() const { return impl_->HasLanguageInput(); }

std::array<Ort::Value, 2> GigaAmCoremlRnnt::Forward(const float *features, int64_t frames,
                                                    const float *language,
                                                    OrtAllocator *allocator) const {
  return impl_->Forward(features, frames, language, allocator);
}

}  // namespace sherpa_onnx
