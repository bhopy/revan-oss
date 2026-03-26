# Building Revan

You need Windows 10/11, an NVIDIA GPU, and about 15 minutes.

## What you need installed

- **CUDA Toolkit 12.x** — https://developer.nvidia.com/cuda-downloads
- **Visual Studio 2019+** with the "Desktop development with C++" workload
- **CMake 3.18+** — https://cmake.org/download/
- **Rust** — https://rustup.rs/
- **Python 3.10+** — for the agents and benchmark scripts
- **Git** — to clone llama.cpp


## 1. Build llama.cpp

Revan links llama.cpp as a static library. It doesn't use llama-server or HTTP at all.

```bash
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp
git checkout b8067
```

b8067 is the tested version. Newer builds may work but I haven't verified them.

```bash
mkdir build && cd build
cmake .. -G "Visual Studio 16 2019" -A x64 \
  -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=75 \
  -DBUILD_SHARED_LIBS=OFF

cmake --build . --config Release
```

Change `CMAKE_CUDA_ARCHITECTURES` to match your GPU:
- `75` — Turing (RTX 2060, 2070, 2080)
- `86` — Ampere (RTX 3060, 3070, 3080)
- `89` — Ada (RTX 4060, 4070, 4080)

Run `nvidia-smi` if you're not sure what you have.

Once it builds, copy the output into the Revan project:

```bash
cd revan-oss
mkdir -p bin_lib/lib bin_lib/include

# Libraries
cp /path/to/llama.cpp/build/Release/llama.lib     bin_lib/lib/
cp /path/to/llama.cpp/build/Release/common.lib     bin_lib/lib/
cp /path/to/llama.cpp/build/Release/ggml.lib       bin_lib/lib/
cp /path/to/llama.cpp/build/Release/ggml-base.lib  bin_lib/lib/
cp /path/to/llama.cpp/build/Release/ggml-cpu.lib   bin_lib/lib/
cp /path/to/llama.cpp/build/Release/ggml-cuda.lib  bin_lib/lib/

# Headers
cp /path/to/llama.cpp/include/llama.h  bin_lib/include/
cp /path/to/llama.cpp/include/ggml.h   bin_lib/include/
cp /path/to/llama.cpp/common/common.h  bin_lib/include/
```

If you get missing header errors later, grab whatever else `revan.cpp` asks for from the llama.cpp source tree.


## 2. Build the engine

```bash
cd revan-oss
mkdir build && cd build
cmake .. -G "Visual Studio 16 2019" -A x64
cmake --build . --config Release
```

Output: `build/Release/revan.exe`

If cmake complains about missing headers or libs, double-check that `bin_lib/include/` and `bin_lib/lib/` have everything from step 1.


## 3. Build the hub

```bash
cd revan-oss/hub
cargo build --release
```

Output: `hub/target/release/revan-hub.exe`

Cargo pulls all dependencies automatically. Nothing special here.


## 4. Get a model

Any GGUF model works. I built and tested with OLMo 3.1 32B Instruct (Q4_K_M), which is ~18 GB:

https://huggingface.co/allenai/OLMo-2-0325-32B-Instruct-GGUF

If you want something smaller to test with first, Llama 3.2 3B (~1.9 GB) works too:

https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF

Put the .gguf file wherever you want.


## 5. Configure

Copy the example config and set your model path:

```bash
cp revan.toml.example revan.toml
```

Edit `revan.toml`:

```toml
[model]
path = "C:/Models/your-model-Q4_K_M.gguf"
n_gpu_layers = 23
ctx_size = 2048
n_threads = 6
```

`n_gpu_layers` depends on your VRAM. For 32B Q4_K_M:
- 6 GB VRAM → 15-18 layers
- 8 GB VRAM → 20-25 layers
- 12 GB VRAM → 38-45 layers
- 16 GB+ → 64 (everything on GPU)

For 7B or 3B models, just set it to 99 — the whole thing fits.

`n_threads` should be your physical core count. Don't count hyperthreads.

`hub.toml` ships ready to use. You shouldn't need to change it unless you renamed the pipe.


## 6. Run it

Three terminals:

**Engine:**
```bash
cd revan-oss
./build/Release/revan.exe
```

**Hub:**
```bash
cd revan-oss
./hub/target/release/revan-hub.exe
```

**Test:**
```bash
cd revan-oss
python tools/bench_revan.py
```

The benchmark runs yes/no, tool routing, and scoring tests. You'll see per-request timing and throughput numbers.


## Hub commands

Once the hub is running, type `help`:

```
help                              — show commands
status                            — running agents + VRAM usage
health                            — ping the engine
spawn <name>                      — start an agent from hub.toml
stop <name>                       — kill a running agent
dispatch <agent> <action> <json>  — send a task and wait for the result
quit                              — shut everything down
```

Example session:

```
> spawn file_scanner
spawned: file_scanner

> dispatch file_scanner classify {"path":"C:/some/file.toml"}
dispatched: abc-123
waiting for result...
result: status=ok, data={"category":"config","gen_ms":362}
```


## Verifying the KV cache

```bash
python tools/test_kv_cache.py
```

This sends the same system prompt multiple times and checks that prompt processing gets faster after the first request (cache hit). You should see 48-60% savings.


## Comparing quantizations

```bash
python tools/quant_ab_test.py --tag q4km     # run with model A
# swap models, then:
python tools/quant_ab_test.py --tag iq4xs    # run with model B
python tools/quant_ab_test.py --compare      # side-by-side results
```


## Troubleshooting

**`Cannot connect (error 2)`** — The engine isn't running. Start `revan.exe` first.

**`Cannot connect (error 231)`** — Pipe is busy. Another request is in progress. Wait a moment.

**Garbage output from inference** — Your model probably doesn't use ChatML format (`<|im_start|>`). OLMo does. If you're using a different model, check its expected prompt template.

**Slow prompt processing** — Not enough layers on GPU. Bump `n_gpu_layers` in `revan.toml` and watch VRAM with `nvidia-smi`.

**CUDA out of memory** — Too many layers on GPU. Lower `n_gpu_layers`.

**Hub can't find hub.toml** — Run it from the project root directory.

**Agent spawns but nothing happens** — Check stderr. Agent errors are forwarded to the hub's terminal output.

**cmake can't find CUDA** — Make sure `nvcc` is in your PATH after installing the CUDA Toolkit.


## Platform note

This is Windows-only. The engine uses Named Pipes, the hub uses Job Objects for process cleanup, VRAM monitoring uses NVML — all Windows APIs. Porting to Linux would mean swapping Named Pipes for Unix sockets and Job Objects for process groups. NVML itself works fine on Linux.
