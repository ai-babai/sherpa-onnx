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
  -DSHERPA_ONNX_USE_PRE_INSTALLED_ONNXRUNTIME_IF_AVAILABLE=OFF \
  -DSHERPA_ONNX_ENABLE_PYTHON=ON -DSHERPA_ONNX_ENABLE_PORTAUDIO=OFF \
  -DSHERPA_ONNX_ENABLE_WEBSOCKET=OFF
cmake --build build-cuda --parallel 8 --target _sherpa_onnx sherpa-onnx-offline sherpa-onnx

# macOS arm64 with CoreML-enabled ONNX Runtime
cmake -S . -B build-mac -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_OSX_ARCHITECTURES=arm64 -DSHERPA_ONNX_ENABLE_PYTHON=ON \
  -DSHERPA_ONNX_USE_PRE_INSTALLED_ONNXRUNTIME_IF_AVAILABLE=OFF \
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

The experimental fixed-width Apple package was tested with the
CoreML-enabled arm64 ORT 1.23.2 runtime. For this configuration, use a fresh
macOS build directory and set `SHERPA_ONNX_USE_PRE_INSTALLED_ONNXRUNTIME_IF_AVAILABLE=ON`
with these environment variables pointing to that ORT distribution.

```sh
export SHERPA_ONNXRUNTIME_INCLUDE_DIR="$ORT/include"
export SHERPA_ONNXRUNTIME_LIB_DIR="$ORT/lib"
```

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

| Configuration | Observed result |
| --- | --- |
| macOS CPU, ORT 1.28.2, stock RNN-T INT8 | 12/12 native scenarios. CUDA unavailable also produced identical text and tokens on the pinned Sherpa Russian WAV. |
| macOS CoreML, ORT 1.28.2, stock v4 CTC FP32 and offline RNN-T INT8 | `self_attn/Reshape_3` during inference after partial CoreML placement. |
| macOS CoreML, ORT 1.28.2, stock streaming INT8 | Encoder session failed CoreML execution-plan creation with error -14. |
| macOS CoreML, ORT 1.23.2, folded fixed-width streaming FP16 | CPU and CoreML each passed the registered 3-case Python regression. Apple GPU activity in all three warm decode windows. |
| macOS CoreML, ORT 1.23.2, compact fixed-width streaming FP16 | Python streaming regression 3/3. Physical Apple GPU activity in all three warm decode windows. Compared with the same graph on CPU, chunk p50 was 45.1 ms versus 34.6 ms over 390 measured chunks per provider. |
| Linux x86_64, A40 CUDA, ORT 1.28.2, stock GigaAM | CTC smoke matched CPU at FP32, FP16, INT8 and INT4. RNN-T CPU and CUDA scenario suites passed on the dynamic graphs. |

The CoreML results were measured on an M4 Pro running macOS 15.8.1.
The CTC legacy-config control had the same 211 CoreML partitions, provider
event counts and fused-node failure before and after the CoreML options
patch. The compact fixed FP16 encoder passed the CPU boundary probe with
ORT 1.28.2. The folded fixed FP16 export failed ORT 1.28.2 CPU initialization
with the default graph optimizations while replacing an initializer.

The folded encoder SHA-256 is
`5a5b3e2c49caa77ee6a6781ca5dc1ed04fa99f16bd4a6fc689a6f3c9b28f3f60`.
The compact encoder SHA-256 is
`a7ec95505813124917e1ffcdf87bb0577d921b629f35e3c4d478839e27b0bf67`.

For the Apple package, use the fixed encoder and its matching frontend
with the standard decoder, joiner and tokens. Add these lines to the CoreML
provider config.

```ini
ModelFormat=MLProgram
MLComputeUnits=CPUAndGPU
RequireStaticInputShapes=1
```

Run the [same streaming regression](gigaam-v4-streaming/README.md#streaming-regression-test)
with its normal fixture environment and the appropriate encoder filename.

```sh
SHERPA_ONNX_GIGAAM_STREAMING_ENCODER=v4_rnnt_640ms_stream_step.fp16.onnx \
SHERPA_ONNX_GIGAAM_TEST_PROVIDER=coreml:/absolute/path/provider.conf \
  ctest --test-dir build-mac-coreml -R '^test_online_gigaam_v4_py$' --output-on-failure
```

At a detected endpoint, reset advances the stream origin by the frames
already consumed and clears model state. A fixed-width encoder consumes up
to 64 mel frames per decode, leaving later queued frames available after
reset. An ordinary reset clears the input queue. The native tests cover
queued speech and a segment offset of 0.64 s, the final mel frame, the
48 kHz resampler tail and token-buffer rejection. The macOS C++ suite
passed all eight cases, and the Python suite passed all three with its
default CPU provider on both dynamic INT8 and fixed FP16.

On Linux x86_64 with an NVIDIA A40, Sherpa 2.0.3 stock CTC matched CPU
text for both smoke clips at FP32, FP16, INT8 and INT4. The FP32 CUDA
profile contained 2,001 CUDA and 850 CPU node events. Nsight recorded 1,586
GPU kernels in the traced FP32 run. These CTC results use the existing
Sherpa CTC implementation.

On the same Linux host, setting `CUDA_VISIBLE_DEVICES=""` logged CUDA
registration failure. The CTC, offline v4, Multilingual-v2 and streaming v4
smoke outputs matched their CPU controls and their profiles showed CPU EP
events only.

The RNN-T CUDA FP32 pilot recorded the following ORT node events on the A40
with ORT 1.28.2 and the combined source at `88b59ec6`. Counts are for the
profiled runs and vary by model and input. Nsight corroborated GPU kernel
execution.

| Graph | CUDA events | CPU events |
| --- | ---: | ---: |
| Offline v4 encoder | 2,169 | 945 |
| Offline v4 advance | 7,931 | 1,284 |
| Streaming v4 frontend | 3,344 | 2,420 |
| Streaming v4 encoder | 70,840 | 24,464 |
| Streaming v4 decoder | 495 | 0 |
| Streaming v4 joiner | 4,873 | 0 |

INT8 encoders placed `DynamicQuantizeLinear` and `MatMulInteger` on CPU
and incurred host-device copies. The INT4 encoder profiles contained
CUDA `MatMulNBits` events. In one quiet RU timing run, streaming FP32 warm
median went from 1,292 ms on CPU to 404 ms on CUDA. For INT8 it went from
757 ms to 1,072 ms. The profiled INT8 encoder accounted for most of the
CPU-to-CUDA latency increase in the sampled run.
