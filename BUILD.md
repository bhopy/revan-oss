# Building Revan

Full walkthrough from zero to running. Takes about 15 minutes if nothing goes wrong.

**You'll need:**

- Windows 10 or 11
- An NVIDIA GPU (any modern one with CUDA support)
- [CUDA Toolkit 12.x](https://developer.nvidia.com/cuda-downloads) installed
- [Visual Studio 2019 or 2022](https://visualstudio.microsoft.com/) with the **"Desktop development with C++"** workload checked
- [CMake 3.18+](https://cmake.org/download/)
- [Rust](https://rustup.rs/)
- [Python 3.10+](https://www.python.org/) (for the agents and test scripts)
- [Git](https://git-scm.com/)

Everything below assumes you're running commands from a terminal that has your compiler and CUDA in PATH. The easiest way is to open the **"Developer Command Prompt for VS"** or just a regular terminal if you've got everything on your system PATH already.

---

## Step 1 — Build llama.cpp as static libraries

Revan doesn't use llama-server or any HTTP API. It links llama.cpp directly as a C++ library inside its own process. So you need to build llama.cpp from source and grab the `.lib` files it produces.

**Clone and checkout the tested version:**

```
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp
git checkout b8067
```

> `b8067` is the version everything was built and tested against. Newer versions might work fine, but no guarantees.

**Figure out your CUDA architecture:**

This tells the compiler which GPU instruction set to target. If you pick the wrong one, CUDA won't run on your card.

| Architecture | GPUs                       | Value |
| ------------ | -------------------------- | ----- |
| Turing       | RTX 2060, 2070, 2080       | `75`  |
| Ampere       | RTX 3060, 3070, 3080       | `86`  |
| Ada Lovelace | RTX 4060, 4070, 4080, 4090 | `89`  |

Not sure which one you have? Run `nvidia-smi` — it shows your GPU name right at the top.

**Configure and build:**

```
mkdir build
cd build
cmake .. -G "Visual Studio 17 2022" -A x64 -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75 -DBUILD_SHARED_LIBS=OFF
cmake --build . --config Release
```

> If you're on Visual Studio 2019, change the generator to `"Visual Studio 16 2019"` instead.
>
> Replace `75` with whatever matches your GPU from the table above.

This takes a few minutes. When it's done, you'll have a bunch of `.lib` files inside `build/Release/` (or scattered in subfolders). Those are what Revan needs.

---

## Step 2 — Copy the llama.cpp output into Revan

Revan expects the compiled libraries and headers in a `bin_lib/` folder at the project root. You need to create that folder and copy everything into it.

**Create the folder structure:**

```
cd path/to/revan-oss
mkdir bin_lib\lib
mkdir bin_lib\include
```

**Copy the libraries** (6 `.lib` files from your llama.cpp build):

```
copy path\to\llama.cpp\build\Release\llama.lib bin_lib\lib\
copy path\to\llama.cpp\build\Release\common.lib bin_lib\lib\
copy path\to\llama.cpp\build\Release\ggml.lib bin_lib\lib\
copy path\to\llama.cpp\build\Release\ggml-base.lib bin_lib\lib\
copy path\to\llama.cpp\build\Release\ggml-cpu.lib bin_lib\lib\
copy path\to\llama.cpp\build\Release\ggml-cuda.lib bin_lib\lib\
```

**Copy the headers** (2 `.h` files the engine needs):

```
copy path\to\llama.cpp\include\llama.h bin_lib\include\
copy path\to\llama.cpp\include\ggml.h bin_lib\include\
```

> Replace `path\to\llama.cpp` with wherever you actually cloned it. For example, if you cloned to `C:\Dev\llama.cpp`, it would be `C:\Dev\llama.cpp\build\Release\llama.lib`.
>
> If the `.lib` files aren't directly in `build/Release/`, check subfolders like `build/src/Release/` or `build/ggml/src/Release/` — llama.cpp moves things around between versions.

---

## Step 3 — Build the Revan engine

This compiles the C++ decision engine (`src/revan.cpp`) into a standalone `.exe` that loads your model and listens for requests on a Named Pipe.

**Before you build** — if your GPU is NOT an RTX 2060/2070/2080 (Turing), open `CMakeLists.txt` and change this line:

```cmake
set(CMAKE_CUDA_ARCHITECTURES 75)
```

Set it to `86` for RTX 30-series or `89` for RTX 40-series (same values from the table in Step 1).

**Build it:**

```
cd path/to/revan-oss
mkdir build
cd build
cmake .. -G "Visual Studio 17 2022" -A x64
cmake --build . --config Release
```

> Again, use `"Visual Studio 16 2019"` if that's what you have.

When it finishes, you'll get `build/Release/revan.exe`. That's the engine.

If CMake complains about missing headers or libraries, double-check that `bin_lib/include/` has `llama.h` and `ggml.h`, and `bin_lib/lib/` has all 6 `.lib` files from Step 2.

---

## Step 4 — Build the hub

The hub is the Rust program that manages everything — it spawns agent processes, routes their requests to the engine, and monitors your GPU memory.

```
cd path/to/revan-oss/hub
cargo build --release
```

Cargo downloads and compiles all dependencies automatically. Nothing to configure. When it's done you'll get `hub/target/release/revan-hub.exe`.

---

## Step 5 — Download a model

Revan works with any GGUF-format model. It was built and benchmarked with **OLMo 3.1 32B Instruct Q4_K_M** (~18 GB download):

https://huggingface.co/allenai/OLMo-2-0325-32B-Instruct-GGUF

If you want something smaller to get going quick, **Llama 3.2 3B** (~1.9 GB) works too:

https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF

Download the `.gguf` file and put it wherever makes sense. You'll point the config at it in the next step.

---

## Step 6 — Configure

**Engine config** — copy the example and fill in your model path:

```
cd path/to/revan-oss
copy revan.toml.example revan.toml
```

Open `revan.toml` and set it up:

```toml
[model]
path = "C:/Models/your-model-Q4_K_M.gguf"
n_gpu_layers = 23
ctx_size = 2048
n_threads = 6

[pipe]
name = \\.\pipe\revan

[inference]
n_predict_max = 15
temperature_default = 0.1
```

**What each setting does:**

- **`path`** — Full path to your `.gguf` model file. Use forward slashes.
- **`n_gpu_layers`** — How many of the model's layers to put on your GPU. More layers = faster, but uses more VRAM. Here's a rough guide for the 32B model (64 layers total):
  - 6 GB VRAM → `15` to `18` layers
  - 8 GB VRAM → `20` to `25` layers
  - 12 GB VRAM → `38` to `45` layers
  - 16+ GB VRAM → `64` (everything on GPU)
  - For 3B or 7B models, just set it to `99` — the whole model fits easily.
- **`ctx_size`** — Context window in tokens. `2048` is plenty since Revan prompts are small (150-250 tokens).
- **`n_threads`** — Number of CPU threads for inference. Set this to your **physical** core count, not the logical/hyperthread count. Example: a 6-core/12-thread CPU → set `6`.
- **`name`** (under `[pipe]`) — The Windows Named Pipe path. Leave this as-is unless you have a reason to change it.
- **`n_predict_max`** — Hard ceiling on output tokens. `15` is the design limit. Don't raise this unless you know what you're doing.
- **`temperature_default`** — How random the model's answers are. `0.1` keeps things deterministic, which is what you want for decisions.

**Hub config** — `hub.toml` ships ready to go. You don't need to touch it unless you changed the pipe name in `revan.toml`.

---

## Step 7 — Run it

You need three terminals open. One for the engine, one for the hub, and one to actually do stuff.

**Terminal 1 — Start the engine:**

```
cd path/to/revan-oss
build\Release\revan.exe
```

You should see it load the model and print something like "listening on pipe". That means it's ready.

**Terminal 2 — Start the hub:**

```
cd path/to/revan-oss
hub\target\release\revan-hub.exe
```

The hub connects to the engine's pipe and starts its interactive prompt.

**Terminal 3 — Run the benchmarks:**

```
cd path/to/revan-oss
python tools/bench_revan.py
```

This fires off yes/no, tool routing, and scoring tests against the engine. You'll see per-request timing and throughput numbers. If everything's wired up right, yes/no answers come back in ~0.4 seconds.

---

## Hub commands

Once the hub is running, you get an interactive prompt. Type `help` to see everything:

| Command                            | What it does                                       |
| ---------------------------------- | -------------------------------------------------- |
| `help`                             | Shows all available commands                       |
| `status`                           | Lists running agents and current VRAM usage        |
| `health`                           | Pings the engine to check if it's alive            |
| `spawn <name>`                     | Starts an agent defined in `hub.toml`              |
| `stop <name>`                      | Kills a running agent                              |
| `dispatch <agent> <action> <json>` | Sends a task to an agent and waits for the result  |
| `quit`                             | Shuts down the hub, engine, and all agents cleanly |

**Quick example session:**

```
> spawn file_scanner
spawned: file_scanner

> dispatch file_scanner classify {"path":"C:/some/file.toml"}
dispatched: abc-123
waiting for result...
result: status=ok, data={"category":"config","gen_ms":362}
```

---

## Extra tools

**KV cache validation** — sends the same system prompt multiple times and checks that the second+ requests are faster (cache hit). You should see 48-60% time savings:

```
python tools/test_kv_cache.py
```

**Quantization A/B testing** — compare accuracy and speed between two different model quantizations:

```
python tools/quant_ab_test.py --tag q4km
```

Then swap your model in `revan.toml`, restart the engine, and run:

```
python tools/quant_ab_test.py --tag iq4xs
python tools/quant_ab_test.py --compare
```

---

## Troubleshooting

| Problem                          | What's happening                             | Fix                                                                                                                            |
| -------------------------------- | -------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| `Cannot connect (error 2)`       | Engine isn't running                         | Start `revan.exe` first, then the hub                                                                                          |
| `Cannot connect (error 231)`     | Pipe is busy (another request is mid-flight) | Wait a sec and try again                                                                                                       |
| Garbage output from inference    | Model expects a different prompt format      | Revan uses ChatML (`<\|im_start\|>`). OLMo uses this natively. If you're using a different model, check what template it needs |
| Slow prompt processing           | Not enough layers on GPU                     | Bump `n_gpu_layers` in `revan.toml`, watch VRAM with `nvidia-smi`                                                              |
| CUDA out of memory               | Too many layers on GPU                       | Lower `n_gpu_layers` in `revan.toml`                                                                                           |
| Hub can't find `hub.toml`        | Wrong working directory                      | Run the hub from the project root (`revan-oss/`)                                                                               |
| Agent spawns but nothing happens | Agent crashed silently                       | Check stderr output in the hub's terminal — agent errors get forwarded there                                                   |
| CMake can't find CUDA            | `nvcc` not in PATH                           | Make sure the CUDA Toolkit bin folder is on your system PATH                                                                   |

---

## Platform

This is Windows-only. The engine uses Named Pipes for IPC, the hub uses Job Objects for process cleanup, and VRAM monitoring uses NVML — all Windows APIs.

Porting to Linux would mean replacing Named Pipes with Unix domain sockets and Job Objects with process groups. NVML itself works cross-platform, so that part's fine.
