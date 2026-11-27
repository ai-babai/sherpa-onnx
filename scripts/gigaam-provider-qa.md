# GigaAM execution-provider verification

Use the combined qualification tree to exercise CTC, offline RNN-T and
streaming RNN-T from one build. The offline base is
`a39d5a503913ca600dcf653abcab0d1a4d695e27`. The streaming base is
`fc5b49d9becc340ffc8ec5b54d6e5a530183479b` from
[#4329](https://github.com/k2-fsa/sherpa-onnx/pull/4329).
Record the source commit or patch tree alongside the measurements.

## Build

```sh
# Linux x86_64 with CUDA 12 and cuDNN 9
cmake -S . -B build-cuda -DCMAKE_BUILD_TYPE=Release \
  -DSHERPA_ONNX_ENABLE_GPU=ON -DBUILD_SHARED_LIBS=ON \
  -DSHERPA_ONNX_ENABLE_PYTHON=ON -DSHERPA_ONNX_ENABLE_PORTAUDIO=OFF \
  -DSHERPA_ONNX_ENABLE_WEBSOCKET=OFF
cmake --build build-cuda --parallel 8 --target _sherpa_onnx sherpa-onnx-offline sherpa-onnx

# macOS arm64 with CoreML-enabled ONNX Runtime
cmake -S . -B build-mac -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_OSX_ARCHITECTURES=arm64 -DSHERPA_ONNX_ENABLE_PYTHON=ON \
  -DSHERPA_ONNX_ENABLE_PORTAUDIO=OFF -DSHERPA_ONNX_ENABLE_WEBSOCKET=OFF \
  -DSHERPA_ONNX_ENABLE_TESTS=ON
cmake --build build-mac --parallel 8 --target _sherpa_onnx sherpa-onnx-offline sherpa-onnx
cmake --build build-mac --parallel 8 --target gigaam-language-slot-test online-stream-test
ctest --test-dir build-mac -R '^(gigaam-language-slot-test|online-stream-test)$' --output-on-failure
```

The default ORT source packages in these builds pin 1.28.2. See the
[offline recipe](../python-api-examples/gigaam-offline-rnnt.md) and
[streaming recipe](gigaam-v4-streaming/README.md) for the pinned graphs,
provider arguments and CPU test commands.

## Verify placement

Set `provider="cuda:/absolute/path/provider.conf"` in Python or pass
`--provider=cuda:/absolute/path/provider.conf` to the CLI. For CoreML use
`coreml:` in place of `cuda:`. To collect a CPU control, use `cpu:` and a
different prefix. Use an existing writable directory and a new prefix for
each run.

```ini
LogSeverityLevel=0
ProfilingFilePrefix=/absolute/path/to/profile
```

For a CoreML MLProgram run, add `ModelFormat=MLProgram` and
`MLComputeUnits=CPUAndGPU` to the CoreML config. These options require ORT
1.20 or later. CoreML with no provider options uses the original flags API.

After the process exits, count the provider labels in each ORT profile.

```sh
jq -r '.[] | select(.cat == "Node") | .args.provider // empty' profile-encoder_*.json |
  sort | uniq -c
```

Repeat for `advance` offline or `frontend`, `decoder` and `joiner` in
streaming. CTC writes one profile with no graph suffix. Require accelerator
events in the graph under test before counting accelerator execution.
CoreML events confirm CoreML partition execution. A CoreML or Metal trace
is needed to identify physical GPU execution.

CPU is the default. A missing accelerator EP logs a diagnostic and uses CPU.
CUDA configuration and registration failures follow Sherpa's existing
logged fallback. CoreML registration errors and ORT session creation or
inference errors reach the caller. ORT can place individual operations on
CPU in an otherwise accelerated session.

The recognizers use synchronous ORT `Run` and exchange ordinary host
tensors. ORT fetches outputs to the host before Sherpa reads lengths, tokens
or logits. Streaming owns the returned state in each stream and uses views
only while the owning tensors are alive. The profiles include any device
copies incurred by this path.
[ORT placement and I/O Binding](https://onnxruntime.ai/docs/performance/tune-performance/iobinding.html)

## Observed limits

On macOS 15.8.1 and Apple M4 Pro with ORT 1.28.2, the stock RNN-T INT8
packages passed the CPU checks for v4 offline, Multilingual-v2 and v4
streaming, including all 12 cases in the GigaAM ONNX native scenario suite.
On this build, requesting CUDA logged the missing provider and produced
identical text and tokens on the pinned Sherpa Russian WAV.

With CoreML MLProgram and CPUAndGPU on the same host, stock v4 CTC FP32
and both offline RNN-T INT8 encoders failed at `self_attn/Reshape_3`
after partial CoreML placement. The stock streaming INT8 encoder failed
CoreML execution-plan creation with error -14. The CTC legacy-config
control had the same 211 CoreML partitions, provider event counts and
fused-node failure before and after the CoreML options patch. These are
stock-graph results for the tested sample and runtime.

On Linux x86_64 with an NVIDIA A40, Sherpa 2.0.3 stock CTC matched CPU
text for both smoke clips at FP32, FP16, INT8 and INT4. The FP32 CUDA
profile contained 2,001 CUDA and 850 CPU node events. Nsight recorded 1,586
GPU kernels in the traced FP32 run. These CTC results use the existing
Sherpa CTC implementation.
