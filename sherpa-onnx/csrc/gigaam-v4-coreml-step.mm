#import <CoreML/CoreML.h>
#import <Foundation/Foundation.h>

#include "sherpa-onnx/csrc/gigaam-v4-coreml-step.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace sherpa_onnx {
namespace {

void Require(bool value, NSString *message) {
  if (!value) {
    throw std::runtime_error(message ? [message UTF8String] : "Core ML error");
  }
}

NSString *String(const std::string &text) {
  return [NSString stringWithUTF8String:text.c_str()];
}

MLMultiArray *WrapInput(const Ort::Value &input) {
  auto info = input.GetTensorTypeAndShapeInfo();
  const auto dimensions = info.GetShape();
  NSMutableArray<NSNumber *> *shape = [NSMutableArray arrayWithCapacity:dimensions.size()];
  NSMutableArray<NSNumber *> *strides = [NSMutableArray arrayWithCapacity:dimensions.size()];
  for (int64_t length : dimensions) [shape addObject:@(length)];
  int64_t stride = info.GetElementCount();
  for (int64_t length : dimensions) {
    stride /= length;
    [strides addObject:@(stride)];
  }
  NSError *error = nil;
  if (info.GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    auto *data = const_cast<float *>(input.GetTensorData<float>());
    MLMultiArray *array = [[MLMultiArray alloc] initWithDataPointer:data
        shape:shape dataType:MLMultiArrayDataTypeFloat32 strides:strides
        deallocator:nil error:&error];
    Require(array != nil, error.localizedDescription);
    return array;
  }
  Require(info.GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64 &&
              info.GetElementCount() == 1, @"Unexpected Core ML step input");
  auto scalar = input.GetTensorData<int64_t>()[0];
  Require(scalar >= 0 && scalar <= std::numeric_limits<int32_t>::max(),
          @"Core ML step counter is out of range");
  MLMultiArray *array = [[MLMultiArray alloc] initWithShape:shape
      dataType:MLMultiArrayDataTypeInt32 error:&error];
  Require(array != nil, error.localizedDescription);
  array[0] = @(scalar);
  return array;
}

Ort::Value ReadOutput(MLMultiArray *array) {
  Require(array != nil, @"Missing Core ML step output");
  std::vector<int64_t> shape;
  for (NSNumber *length in array.shape) shape.push_back(length.longLongValue);
  Ort::AllocatorWithDefaultOptions allocator;
  if (array.dataType == MLMultiArrayDataTypeInt32 && array.count == 1) {
    auto result = Ort::Value::CreateTensor<int64_t>(allocator, shape.data(), shape.size());
    result.GetTensorMutableData<int64_t>()[0] = [array[0] longLongValue];
    return result;
  }

  Require(array.dataType == MLMultiArrayDataTypeFloat32, @"Unexpected Core ML step output type");
  auto result = Ort::Value::CreateTensor<float>(allocator, shape.data(), shape.size());
  float *dest = result.GetTensorMutableData<float>();
  const float *source = static_cast<const float *>(array.dataPointer);
  size_t count = array.count;
  size_t stride = 1;
  bool contiguous = true;
  for (size_t i = shape.size(); i-- > 0;) {
    contiguous &= shape[i] == 1 || array.strides[i].unsignedLongLongValue == stride;
    stride *= shape[i];
  }
  if (contiguous) {
    std::copy_n(source, count, dest);
  } else {
    std::vector<size_t> strides;
    for (NSNumber *step in array.strides) strides.push_back(step.unsignedLongLongValue);
    for (size_t i = 0; i != count; ++i) {
      size_t index = i, offset = 0;
      for (size_t axis = shape.size(); axis-- > 0;) {
        offset += index % shape[axis] * strides[axis];
        index /= shape[axis];
      }
      dest[i] = source[offset];
    }
  }
  return result;
}

}  // namespace

class GigaAmCoreMlStep::Impl {
 public:
  MLModel *model;
  std::vector<std::string> input_names{"features", "feature_lengths", "position",
                                       "cache_valid"};
  std::vector<std::string> output_names{"encoded", "encoded_lengths",
                                        "next_position", "next_cache_valid"};
  std::vector<std::vector<int64_t>> cache_shapes;
};

GigaAmCoreMlStep::GigaAmCoreMlStep(const std::string &compiled_path)
    : impl_(std::make_unique<Impl>()) {
  @autoreleasepool {
    MLModelConfiguration *configuration = [[MLModelConfiguration alloc] init];
    configuration.computeUnits = MLComputeUnitsCPUAndGPU;
    NSURL *url = [NSURL fileURLWithPath:String(compiled_path) isDirectory:YES];
    NSError *error = nil;
    impl_->model = [MLModel modelWithContentsOfURL:url configuration:configuration error:&error];
    Require(impl_->model != nil, error.localizedDescription);
    for (int i = 0; i != 54; ++i) {
      const auto name = "cache_" + std::to_string(i);
      impl_->input_names.push_back(name);
      impl_->output_names.push_back("next_" + name);
    }

    const auto describe = [](NSDictionary<NSString *, MLFeatureDescription *> *features,
                             const std::vector<std::string> &names) {
      Require(features.count == names.size(), @"Unexpected GigaAM-v4 Core ML I/O count");
      std::vector<std::vector<int64_t>> shapes;
      for (size_t i = 0; i < names.size(); ++i) {
        const auto key = String(names[i]);
        MLMultiArrayConstraint *array = features[key].multiArrayConstraint;
        const auto dtype = i > 0 && i < 4 ? MLMultiArrayDataTypeInt32
                                           : MLMultiArrayDataTypeFloat32;
        Require(array != nil && array.dataType == dtype && array.shape.count,
                [NSString stringWithFormat:@"Unexpected Core ML step tensor %@", key]);
        std::vector<int64_t> shape;
        for (NSNumber *dim in array.shape) {
          Require(dim.longLongValue > 0, @"GigaAM-v4 Core ML requires positive static shapes");
          shape.push_back(dim.longLongValue);
        }
        shapes.push_back(std::move(shape));
      }
      return shapes;
    };
    const auto in = describe(impl_->model.modelDescription.inputDescriptionsByName,
                             impl_->input_names);
    const auto out = describe(impl_->model.modelDescription.outputDescriptionsByName,
                              impl_->output_names);
    Require(in[0] == std::vector<int64_t>{1, 16, 768} &&
                out[0] == std::vector<int64_t>{1, 768, 16},
            @"Unexpected GigaAM-v4 Core ML feature geometry");
    for (int i = 1; i != 4; ++i) {
      Require(in[i] == std::vector<int64_t>{1} && out[i] == in[i],
              @"Unexpected GigaAM-v4 Core ML scalar geometry");
    }
    Require(std::equal(in.begin() + 4, in.end(), out.begin() + 4),
            @"GigaAM-v4 Core ML cache geometry mismatch");
    impl_->cache_shapes.assign(in.begin() + 4, in.end());
  }
}

GigaAmCoreMlStep::~GigaAmCoreMlStep() = default;

std::vector<Ort::Value> GigaAmCoreMlStep::ZeroState(size_t start) const {
  if (start != 2) throw std::invalid_argument("Expected GigaAM-v4 encoder state");
  Ort::AllocatorWithDefaultOptions allocator;
  std::vector<Ort::Value> state;
  const int64_t scalar_shape = 1;
  for (int i = 0; i != 2; ++i) {
    auto value = Ort::Value::CreateTensor<int64_t>(allocator, &scalar_shape, 1);
    value.GetTensorMutableData<int64_t>()[0] = 0;
    state.push_back(std::move(value));
  }
  for (const auto &shape : impl_->cache_shapes) {
    auto value = Ort::Value::CreateTensor<float>(allocator, shape.data(), shape.size());
    std::fill_n(value.GetTensorMutableData<float>(),
                value.GetTensorTypeAndShapeInfo().GetElementCount(), 0.0f);
    state.push_back(std::move(value));
  }
  return state;
}

std::vector<Ort::Value> GigaAmCoreMlStep::Run(
    const std::vector<Ort::Value> &inputs) const {
  @autoreleasepool {
    Require(inputs.size() == impl_->input_names.size(), @"Unexpected GigaAM-v4 Core ML inputs");
    NSMutableDictionary *feed = [NSMutableDictionary dictionaryWithCapacity:inputs.size()];
    for (size_t i = 0; i != inputs.size(); ++i) {
      feed[String(impl_->input_names[i])] = WrapInput(inputs[i]);
    }
    NSError *error = nil;
    auto *provider = [[MLDictionaryFeatureProvider alloc] initWithDictionary:feed error:&error];
    Require(provider != nil, error.localizedDescription);
    id<MLFeatureProvider> outputs = [impl_->model predictionFromFeatures:provider error:&error];
    Require(outputs != nil, error.localizedDescription);
    std::vector<Ort::Value> result;
    result.reserve(impl_->output_names.size());
    for (const auto &name : impl_->output_names) {
      result.push_back(ReadOutput([outputs featureValueForName:String(name)].multiArrayValue));
    }
    return result;
  }
}

}  // namespace sherpa_onnx
