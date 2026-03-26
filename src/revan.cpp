// revan.cpp — Silent Decision Engine
// Single-file C++ inference engine linking llama.cpp as a static library.
// Stateless: 150-250 token prompts in, 1-15 token decisions out.
// IPC: Windows Named Pipes with length-prefixed binary protocol.
// No HTTP, no server process, no dependencies beyond llama.cpp + Win32.

// ==================== INCLUDES ====================

#include "llama.h"
#include "ggml.h"

#define NOMINMAX
#include <windows.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <chrono>
#include <atomic>
#include <fstream>
#include <sstream>
#include <filesystem>

namespace fs = std::filesystem;

// ==================== PROTOCOL CONSTANTS ====================

static constexpr uint16_t PROTO_VERSION = 1;

// Request types
static constexpr uint16_t REQ_INFERENCE    = 0x0001;
static constexpr uint16_t REQ_HEALTH_CHECK = 0x0002;
static constexpr uint16_t REQ_SHUTDOWN     = 0x0003;
static constexpr uint16_t REQ_INFERENCE_V2 = 0x0004;  // grammar + logprobs
static constexpr uint16_t REQ_MODEL_SWAP   = 0x0005;  // hot model swap

// Response status codes
static constexpr uint16_t STATUS_OK                    = 0x0000;
static constexpr uint16_t STATUS_ERROR_MODEL_NOT_LOADED = 0x0001;
static constexpr uint16_t STATUS_ERROR_PROMPT_TOO_LONG  = 0x0002;
static constexpr uint16_t STATUS_ERROR_INFERENCE_FAILED = 0x0003;
static constexpr uint16_t STATUS_ERROR_TIMEOUT          = 0x0004;

// Header sizes
static constexpr int REQUEST_HEADER_SIZE  = 20;  // 20 bytes (with system prompt length)
static constexpr int RESPONSE_HEADER_SIZE = 24;  // 24 bytes fixed

// ==================== HARDCODED PARAMETERS ====================
// These never change — proven by benchmark and confirmed in REVAN-MASTER.md

static constexpr int   BATCH_SIZE         = 2048;
static constexpr int   UBATCH_SIZE        = 512;
static constexpr int   TOP_K              = 40;
static constexpr float TOP_P              = 0.95f;
static constexpr float MIN_P              = 0.05f;
static constexpr float REP_PENALTY        = 1.0f;   // off
static constexpr int   DECODE_WATCHDOG_MS = 10000;   // 10s per decode call
static constexpr int   WALL_TIMEOUT_MS    = 120000;  // 120s total request
static constexpr int   PIPE_BUFFER_SIZE   = 65536;   // 64 KB

// ==================== CONFIG STRUCT ====================

struct RevanConfig {
    // [model]
    std::string model_path;
    int n_gpu_layers      = 23;
    int ctx_size          = 2048;
    int n_threads         = 6;

    // [pipe]
    std::string pipe_name = "\\\\.\\pipe\\revan";

    // [inference]
    int   n_predict_max       = 15;
    float temperature_default = 0.1f;

    // resolved at startup
    std::string root_dir;
};

// ==================== GLOBALS ====================

static std::atomic<bool> g_shutdown{false};
static std::atomic<bool> g_abort_inference{false};
static llama_model*      g_model = nullptr;
static llama_context*    g_ctx   = nullptr;
static std::string       g_pipe_name;  // set once in run_pipe_server, read by ctrl_handler

// ==================== SYSTEM PROMPT KV CACHE ====================
// Caches the KV state after evaluating a system prompt so repeat requests
// with the same system prompt skip prompt eval entirely.

static constexpr int MAX_SYS_CACHE_SLOTS = 3;

struct CachedSystemPrompt {
    uint64_t             hash;            // std::hash of system_prompt string
    std::string          system_prompt;   // full text for collision check
    std::vector<uint8_t> kv_state;        // serialized KV cache (seq 0)
    int                  n_tokens;        // token count of system prefix
    uint64_t             last_used;       // LRU counter
};

static std::vector<CachedSystemPrompt> g_sys_cache;
static uint64_t g_cache_counter = 0;

static CachedSystemPrompt* find_sys_cache(uint64_t hash, const std::string& sys) {
    for (auto& entry : g_sys_cache) {
        if (entry.hash == hash && entry.system_prompt == sys) {
            entry.last_used = ++g_cache_counter;
            return &entry;
        }
    }
    return nullptr;
}

static void store_sys_cache(uint64_t hash, const std::string& sys,
                            const std::vector<uint8_t>& kv_state, int n_tokens) {
    if ((int)g_sys_cache.size() >= MAX_SYS_CACHE_SLOTS) {
        // Evict LRU entry
        auto lru = std::min_element(g_sys_cache.begin(), g_sys_cache.end(),
            [](const CachedSystemPrompt& a, const CachedSystemPrompt& b) {
                return a.last_used < b.last_used;
            });
        *lru = { hash, sys, kv_state, n_tokens, ++g_cache_counter };
    } else {
        g_sys_cache.push_back({ hash, sys, kv_state, n_tokens, ++g_cache_counter });
    }
}

// ==================== LOGGING ====================
// stderr only, structured [REVAN] prefix, no log files

static void log_info(const char* fmt, ...) {
    va_list args;
    va_start(args, fmt);
    fprintf(stderr, "[REVAN] ");
    vfprintf(stderr, fmt, args);
    fprintf(stderr, "\n");
    va_end(args);
}

static void log_err(const char* fmt, ...) {
    va_list args;
    va_start(args, fmt);
    fprintf(stderr, "[REVAN] ERROR: ");
    vfprintf(stderr, fmt, args);
    fprintf(stderr, "\n");
    va_end(args);
}

// llama.cpp log callback — suppress INFO, only show WARN and above
static void llama_log_callback(enum ggml_log_level level, const char* text, void* /*user_data*/) {
    if (level >= GGML_LOG_LEVEL_WARN) {
        fprintf(stderr, "%s", text);
    }
}

// ==================== UTILITY ====================

static inline int64_t elapsed_ms(std::chrono::steady_clock::time_point start) {
    auto now = std::chrono::steady_clock::now();
    return std::chrono::duration_cast<std::chrono::milliseconds>(now - start).count();
}

// Read a little-endian uint16 from buffer
static inline uint16_t read_u16(const uint8_t* buf) {
    return (uint16_t)buf[0] | ((uint16_t)buf[1] << 8);
}

// Read a little-endian uint32 from buffer
static inline uint32_t read_u32(const uint8_t* buf) {
    return (uint32_t)buf[0] | ((uint32_t)buf[1] << 8) |
           ((uint32_t)buf[2] << 16) | ((uint32_t)buf[3] << 24);
}

// Write a little-endian uint16 to buffer
static inline void write_u16(uint8_t* buf, uint16_t val) {
    buf[0] = (uint8_t)(val & 0xFF);
    buf[1] = (uint8_t)((val >> 8) & 0xFF);
}

// Write a little-endian uint32 to buffer
static inline void write_u32(uint8_t* buf, uint32_t val) {
    buf[0] = (uint8_t)(val & 0xFF);
    buf[1] = (uint8_t)((val >> 8) & 0xFF);
    buf[2] = (uint8_t)((val >> 16) & 0xFF);
    buf[3] = (uint8_t)((val >> 24) & 0xFF);
}

// Write a little-endian int32 to buffer (memcpy for strict aliasing safety)
static inline void write_i32(uint8_t* buf, int32_t val) {
    memcpy(buf, &val, 4);
}

// Write a little-endian float32 to buffer (memcpy for strict aliasing safety)
static inline void write_f32(uint8_t* buf, float val) {
    memcpy(buf, &val, 4);
}

// ==================== V2 LOGPROB TYPES ====================

// Single token + its log-probability
struct TokenLogprob {
    int32_t token_id;
    float   logprob;   // natural log probability (ln)
};

// Per-generated-token entry: chosen token + top-N candidates
struct TokenLogprobEntry {
    TokenLogprob chosen;
    std::vector<TokenLogprob> top_candidates;
};

// ==================== TOML PARSER (minimal) ====================
// Only handles the 3 sections we need. No arrays, no nested tables.

static bool parse_config(const std::string& path, RevanConfig& cfg) {
    std::ifstream file(path);
    if (!file.is_open()) {
        log_err("Cannot open config: %s", path.c_str());
        return false;
    }

    std::string line;
    std::string section;

    while (std::getline(file, line)) {
        // strip leading/trailing whitespace
        size_t start = line.find_first_not_of(" \t\r\n");
        if (start == std::string::npos) continue;
        line = line.substr(start);
        size_t end = line.find_last_not_of(" \t\r\n");
        if (end != std::string::npos) line = line.substr(0, end + 1);

        // skip comments and empty
        if (line.empty() || line[0] == '#') continue;

        // section header
        if (line[0] == '[') {
            size_t close = line.find(']');
            if (close != std::string::npos) {
                section = line.substr(1, close - 1);
            }
            continue;
        }

        // key = value
        size_t eq = line.find('=');
        if (eq == std::string::npos) continue;

        std::string key = line.substr(0, eq);
        std::string val = line.substr(eq + 1);

        // trim key and value
        auto trim = [](std::string& s) {
            size_t a = s.find_first_not_of(" \t\"");
            size_t b = s.find_last_not_of(" \t\"");
            if (a == std::string::npos) { s.clear(); return; }
            s = s.substr(a, b - a + 1);
        };
        trim(key);
        trim(val);

        // assign to config — wrap conversions to catch malformed values
        try {
            if (section == "model") {
                if (key == "path")             cfg.model_path = val;
                else if (key == "n_gpu_layers") cfg.n_gpu_layers = std::stoi(val);
                else if (key == "ctx_size")     cfg.ctx_size = std::stoi(val);
                else if (key == "n_threads")    cfg.n_threads = std::stoi(val);
            } else if (section == "pipe") {
                if (key == "name") cfg.pipe_name = val;
            } else if (section == "inference") {
                if (key == "n_predict_max")       cfg.n_predict_max = std::stoi(val);
                else if (key == "temperature_default") cfg.temperature_default = std::stof(val);
            }
        } catch (const std::exception& e) {
            log_err("Bad value for [%s] %s = '%s': %s", section.c_str(), key.c_str(), val.c_str(), e.what());
            return false;
        }
    }

    return true;
}

// ==================== CHATML WRAPPING ====================
// Revan wraps ChatML internally (Option A). Caller sends system + user separately.
// Format: <|im_start|>system\n{sys}<|im_end|>\n<|im_start|>user\n{usr}<|im_end|>\n<|im_start|>assistant\n

static std::string build_chatml(const std::string& system_prompt, const std::string& user_prompt) {
    std::string result;
    result.reserve(system_prompt.size() + user_prompt.size() + 80);

    result += "<|im_start|>system\n";
    result += system_prompt;
    result += "<|im_end|>\n";
    result += "<|im_start|>user\n";
    result += user_prompt;
    result += "<|im_end|>\n";
    result += "<|im_start|>assistant\n";

    return result;
}

// Split ChatML builders for KV cache (system and user tokenized separately)
static std::string build_chatml_system(const std::string& system_prompt) {
    std::string result;
    result.reserve(system_prompt.size() + 40);
    result += "<|im_start|>system\n";
    result += system_prompt;
    result += "<|im_end|>\n";
    return result;
}

static std::string build_chatml_user(const std::string& user_prompt) {
    std::string result;
    result.reserve(user_prompt.size() + 50);
    result += "<|im_start|>user\n";
    result += user_prompt;
    result += "<|im_end|>\n";
    result += "<|im_start|>assistant\n";
    return result;
}

// Create a batch with explicit positions (for KV cache restore path)
static llama_batch make_positioned_batch(const llama_token* tokens, int n_tokens, int start_pos) {
    llama_batch batch = llama_batch_init(n_tokens, 0, 1);
    for (int i = 0; i < n_tokens; i++) {
        batch.token[i]      = tokens[i];
        batch.pos[i]        = start_pos + i;
        batch.n_seq_id[i]   = 1;
        batch.seq_id[i][0]  = 0;
        batch.logits[i]     = (i == n_tokens - 1) ? 1 : 0;
    }
    batch.n_tokens = n_tokens;
    return batch;
}

// ==================== RESPONSE BUILDER ====================

static std::vector<uint8_t> build_response(
    uint16_t status,
    uint16_t tokens_generated,
    uint16_t tokens_in_prompt,
    uint32_t prompt_eval_ms,
    uint32_t gen_ms,
    const std::string& output_text
) {
    uint32_t msg_len = RESPONSE_HEADER_SIZE + (uint32_t)output_text.size();
    std::vector<uint8_t> buf(msg_len);

    write_u32(&buf[0],  msg_len);
    write_u16(&buf[4],  PROTO_VERSION);
    write_u16(&buf[6],  status);
    write_u16(&buf[8],  tokens_generated);
    write_u16(&buf[10], tokens_in_prompt);
    write_u32(&buf[12], prompt_eval_ms);
    write_u32(&buf[16], gen_ms);
    write_u32(&buf[20], 0);  // reserved

    if (!output_text.empty()) {
        memcpy(&buf[24], output_text.data(), output_text.size());
    }

    return buf;
}

// ==================== V2 RESPONSE BUILDER ====================
// Header [0:24] same as V1, but bytes [20:24] repurposed:
//   [20:22] u16 output_text_length
//   [22]    u8  top_n (echoes request)
//   [23]    u8  reserved
// After header: output_text, then logprobs section

static std::vector<uint8_t> build_response_v2(
    uint16_t status,
    uint16_t tokens_generated,
    uint16_t tokens_in_prompt,
    uint32_t prompt_eval_ms,
    uint32_t gen_ms,
    const std::string& output_text,
    uint8_t top_n,
    const std::vector<TokenLogprobEntry>& logprobs
) {
    uint16_t text_len = (uint16_t)output_text.size();

    // Logprobs section: per token = 8 bytes (chosen) + 8 * top_n bytes (candidates)
    size_t per_token_size = 8 + 8 * (size_t)top_n;
    size_t logprobs_size = logprobs.size() * per_token_size;

    uint32_t msg_len = RESPONSE_HEADER_SIZE + text_len + (uint32_t)logprobs_size;
    std::vector<uint8_t> buf(msg_len);

    // Header (24 bytes)
    write_u32(&buf[0],  msg_len);
    write_u16(&buf[4],  PROTO_VERSION);
    write_u16(&buf[6],  status);
    write_u16(&buf[8],  tokens_generated);
    write_u16(&buf[10], tokens_in_prompt);
    write_u32(&buf[12], prompt_eval_ms);
    write_u32(&buf[16], gen_ms);
    // V2 repurposed bytes [20:24]
    write_u16(&buf[20], text_len);
    buf[22] = top_n;
    buf[23] = 0;  // reserved

    // Output text
    if (text_len > 0) {
        memcpy(&buf[24], output_text.data(), text_len);
    }

    // Logprobs section (binary, little-endian)
    size_t offset = 24 + text_len;
    for (const auto& entry : logprobs) {
        write_i32(&buf[offset],     entry.chosen.token_id);
        write_f32(&buf[offset + 4], entry.chosen.logprob);
        offset += 8;

        for (uint8_t c = 0; c < top_n; c++) {
            if (c < entry.top_candidates.size()) {
                write_i32(&buf[offset],     entry.top_candidates[c].token_id);
                write_f32(&buf[offset + 4], entry.top_candidates[c].logprob);
            } else {
                // Pad with zeros if fewer candidates than top_n
                write_i32(&buf[offset],     0);
                write_f32(&buf[offset + 4], -100.0f);
            }
            offset += 8;
        }
    }

    return buf;
}

// ==================== PIPE HELPERS ====================

static bool send_response(HANDLE pipe, const std::vector<uint8_t>& data) {
    DWORD written = 0;
    BOOL ok = WriteFile(pipe, data.data(), (DWORD)data.size(), &written, NULL);
    return ok && written == (DWORD)data.size();
}

// ==================== CTRL+C HANDLER ====================

static BOOL WINAPI ctrl_handler(DWORD event) {
    if (event == CTRL_C_EVENT || event == CTRL_CLOSE_EVENT) {
        log_info("Shutdown signal received.");
        g_shutdown.store(true);
        g_abort_inference.store(true);

        // Unblock ConnectNamedPipe by making a dummy connection
        if (!g_pipe_name.empty()) {
            HANDLE dummy = CreateFileA(g_pipe_name.c_str(),
                GENERIC_READ | GENERIC_WRITE, 0, NULL, OPEN_EXISTING, 0, NULL);
            if (dummy != INVALID_HANDLE_VALUE) {
                CloseHandle(dummy);
            }
        }
        return TRUE;
    }
    return FALSE;
}

// ==================== ABORT CALLBACK ====================
// Layer 3: works on CPU layers only (41 of 64 at ngl=23)

static bool abort_callback(void* data) {
    return ((std::atomic<bool>*)data)->load(std::memory_order_relaxed);
}

// ==================== MODEL LOADING ====================

static bool load_model(const RevanConfig& cfg) {
    // Resolve model path (relative to root dir)
    fs::path model_path = fs::path(cfg.root_dir) / cfg.model_path;
    std::string model_path_str = model_path.string();

    // Check file exists
    if (!fs::exists(model_path)) {
        log_err("Model file not found: %s", model_path_str.c_str());
        return false;
    }

    log_info("Loading model: %s", model_path_str.c_str());
    auto load_start = std::chrono::steady_clock::now();

    // Model params
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = cfg.n_gpu_layers;

    g_model = llama_model_load_from_file(model_path_str.c_str(), model_params);
    if (!g_model) {
        log_err("Failed to load model (nullptr returned). File may be corrupt or VRAM insufficient.");
        return false;
    }

    // Context params — all proven benchmark settings
    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx       = cfg.ctx_size;
    ctx_params.n_threads   = cfg.n_threads;
    ctx_params.n_threads_batch = cfg.n_threads;
    ctx_params.n_batch     = BATCH_SIZE;
    ctx_params.n_ubatch    = UBATCH_SIZE;
    ctx_params.type_k      = GGML_TYPE_Q4_0;         // q4_0 KV cache
    ctx_params.type_v      = GGML_TYPE_Q4_0;
    ctx_params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;  // flash attention ON
    ctx_params.swa_full        = false;              // reduced SWA cache (saves VRAM)
    ctx_params.abort_callback      = abort_callback;  // Layer 3
    ctx_params.abort_callback_data = &g_abort_inference;

    g_ctx = llama_init_from_model(g_model, ctx_params);
    if (!g_ctx) {
        log_err("Failed to create context. Possibly VRAM or memory issue.");
        llama_model_free(g_model);
        g_model = nullptr;
        return false;
    }

    float load_secs = elapsed_ms(load_start) / 1000.0f;
    log_info("Model loaded (%.1fs)", load_secs);

    return true;
}

// ==================== WARM-UP ====================
// One dummy inference to prime CUDA kernels and memory pools

static void warmup(const RevanConfig& cfg) {
    log_info("Warm-up inference...");
    auto start = std::chrono::steady_clock::now();

    const llama_vocab* vocab = llama_model_get_vocab(g_model);

    // Warmup prompt (minimal ChatML)
    const char* warmup_text = "<|im_start|>system\nReady.<|im_end|>\n<|im_start|>user\nPing.<|im_end|>\n<|im_start|>assistant\n";

    // Tokenize
    std::vector<llama_token> tokens(256);
    int n_tokens = llama_tokenize(vocab, warmup_text, (int)strlen(warmup_text),
                                  tokens.data(), (int)tokens.size(), true, true);
    if (n_tokens < 0) {
        log_err("Warm-up tokenize failed, skipping");
        return;
    }
    tokens.resize(n_tokens);

    // Normal decode (NOT llama_set_warmup — that forces ALL 64 layers which is too slow
    // for partial offload at ngl=23). A normal decode primes the CUDA kernels we actually use.
    llama_batch batch = llama_batch_get_one(tokens.data(), n_tokens);
    int result = llama_decode(g_ctx, batch);

    if (result != 0) {
        log_err("Warm-up decode returned %d, continuing anyway", result);
    }

    // Clear KV cache after warmup
    llama_memory_t mem = llama_get_memory(g_ctx);
    if (mem) {
        llama_memory_clear(mem, false);
    }

    float ms = (float)elapsed_ms(start);
    log_info("Warm-up done (%.1fs)", ms / 1000.0f);
}

// ==================== INFERENCE ====================

struct InferenceResult {
    uint16_t    status;
    std::string output;
    uint16_t    tokens_generated;
    uint16_t    tokens_in_prompt;
    uint32_t    prompt_eval_ms;
    uint32_t    gen_ms;
    bool        fatal_cuda = false;  // sticky CUDA error — caller must exit(2)
    std::vector<TokenLogprobEntry> logprobs;  // V2: per-token logprob data (empty for V1)
};

static InferenceResult run_inference(
    const std::string& system_prompt,
    const std::string& user_prompt,
    int max_tokens,
    float temperature,
    const std::string& grammar = "",
    uint8_t top_n_logprobs = 0
) {
    InferenceResult res = {};
    res.status = STATUS_OK;

    const llama_vocab* vocab = llama_model_get_vocab(g_model);

    // Get memory handle for cleanup
    llama_memory_t mem = llama_get_memory(g_ctx);

    // Reset abort flag
    g_abort_inference.store(false);

    int ctx_size = llama_n_ctx(g_ctx);
    int decode_result = 0;

    auto prompt_start = std::chrono::steady_clock::now();
    auto request_start = prompt_start;  // Layer 4: wall clock

    // ---- Helper: check decode result and set error status ----
    // Returns true if decode failed (caller should return res)
    auto handle_decode_error = [&](int result, const char* context) -> bool {
        if (result == 0) return false;
        log_err("Decode failed (code %d) in %s", result, context);
        if (result < -1) {
            cudaError_t cuda_err = cudaGetLastError();
            if (cuda_err == cudaErrorIllegalAddress || cuda_err == cudaErrorLaunchFailure) {
                log_err("FATAL: Sticky CUDA error %d, must exit", (int)cuda_err);
                res.fatal_cuda = true;
            }
        }
        res.status = (result == 2) ? STATUS_ERROR_TIMEOUT : STATUS_ERROR_INFERENCE_FAILED;
        return true;
    };

    // ---- System Prompt KV Cache Path ----
    // If system_prompt is non-empty, try to reuse cached KV state.
    // Cache key: hash of raw system_prompt string.

    bool use_uncached = system_prompt.empty();

    if (!use_uncached) {
        uint64_t sys_hash = std::hash<std::string>{}(system_prompt);
        CachedSystemPrompt* cached = find_sys_cache(sys_hash, system_prompt);

        // Build user suffix (same for both hit and miss paths)
        std::string user_chatml = build_chatml_user(user_prompt);
        std::vector<llama_token> usr_tokens(ctx_size);
        int n_usr = llama_tokenize(vocab, user_chatml.c_str(), (int)user_chatml.size(),
                                   usr_tokens.data(), (int)usr_tokens.size(), false, true);
        if (n_usr < 0) {
            log_err("User tokenize failed (returned %d)", n_usr);
            res.status = STATUS_ERROR_INFERENCE_FAILED;
            return res;
        }
        usr_tokens.resize(n_usr);

        if (cached) {
            // ---- CACHE HIT: restore KV state, decode only user tokens ----
            int n_total = cached->n_tokens + n_usr;

            if (n_total + max_tokens > ctx_size) {
                log_err("Prompt too long: %d tokens + %d max_gen > %d ctx",
                        n_total, max_tokens, ctx_size);
                res.status = STATUS_ERROR_PROMPT_TOO_LONG;
                return res;
            }

            // Clear KV and restore cached system prompt state
            if (mem) llama_memory_clear(mem, false);
            size_t restored = llama_state_seq_set_data(g_ctx, cached->kv_state.data(),
                                                       cached->kv_state.size(), 0);
            if (restored == 0) {
                log_err("KV cache restore failed, falling through to uncached path");
                if (mem) llama_memory_clear(mem, false);
                use_uncached = true;
            } else {
                // Decode user tokens at positions after the cached system tokens
                llama_batch usr_batch = make_positioned_batch(usr_tokens.data(), n_usr, cached->n_tokens);
                decode_result = llama_decode(g_ctx, usr_batch);
                llama_batch_free(usr_batch);
                llama_synchronize(g_ctx);

                res.prompt_eval_ms = (uint32_t)elapsed_ms(prompt_start);
                res.tokens_in_prompt = (uint16_t)n_total;

                if (handle_decode_error(decode_result, "cache hit user decode")) {
                    if (mem) llama_memory_clear(mem, false);
                    return res;
                }

                log_info("SYS_CACHE HIT hash=%llu sys=%dtok usr=%dtok",
                         (unsigned long long)sys_hash, cached->n_tokens, n_usr);
            }

        } else {
            // ---- CACHE MISS: decode system, snapshot, then decode user ----
            std::string sys_chatml = build_chatml_system(system_prompt);
            std::vector<llama_token> sys_tokens(ctx_size);
            int n_sys = llama_tokenize(vocab, sys_chatml.c_str(), (int)sys_chatml.size(),
                                       sys_tokens.data(), (int)sys_tokens.size(), true, true);
            if (n_sys < 0) {
                log_err("System tokenize failed (returned %d)", n_sys);
                res.status = STATUS_ERROR_INFERENCE_FAILED;
                return res;
            }
            sys_tokens.resize(n_sys);

            int n_total = n_sys + n_usr;

            if (n_total + max_tokens > ctx_size) {
                log_err("Prompt too long: %d tokens + %d max_gen > %d ctx",
                        n_total, max_tokens, ctx_size);
                res.status = STATUS_ERROR_PROMPT_TOO_LONG;
                return res;
            }

            // Clear KV and decode system tokens
            if (mem) llama_memory_clear(mem, false);
            llama_batch sys_batch = make_positioned_batch(sys_tokens.data(), n_sys, 0);
            decode_result = llama_decode(g_ctx, sys_batch);
            llama_batch_free(sys_batch);
            llama_synchronize(g_ctx);

            if (handle_decode_error(decode_result, "cache miss system decode")) {
                res.prompt_eval_ms = (uint32_t)elapsed_ms(prompt_start);
                res.tokens_in_prompt = (uint16_t)n_total;
                return res;
            }

            // Snapshot KV state for this system prompt
            size_t state_size = llama_state_seq_get_size(g_ctx, 0);
            if (state_size > 0) {
                std::vector<uint8_t> state_buf(state_size);
                size_t written = llama_state_seq_get_data(g_ctx, state_buf.data(), state_buf.size(), 0);
                if (written > 0) {
                    state_buf.resize(written);
                    store_sys_cache(sys_hash, system_prompt, state_buf, n_sys);
                    log_info("SYS_CACHE STORE hash=%llu sys=%dtok state=%zuB",
                             (unsigned long long)sys_hash, n_sys, written);
                }
            }

            // Decode user tokens
            llama_batch usr_batch = make_positioned_batch(usr_tokens.data(), n_usr, n_sys);
            decode_result = llama_decode(g_ctx, usr_batch);
            llama_batch_free(usr_batch);
            llama_synchronize(g_ctx);

            res.prompt_eval_ms = (uint32_t)elapsed_ms(prompt_start);
            res.tokens_in_prompt = (uint16_t)n_total;

            if (handle_decode_error(decode_result, "cache miss user decode")) {
                return res;
            }

            log_info("SYS_CACHE MISS hash=%llu sys=%dtok usr=%dtok",
                     (unsigned long long)sys_hash, n_sys, n_usr);
        }
    }

    // ---- Uncached path (empty system prompt or cache restore failure) ----
    if (use_uncached) {
        std::string chatml = build_chatml(system_prompt, user_prompt);
        std::vector<llama_token> tokens(ctx_size);
        int n_tokens = llama_tokenize(vocab, chatml.c_str(), (int)chatml.size(),
                                      tokens.data(), (int)tokens.size(), true, true);
        if (n_tokens < 0) {
            log_err("Tokenize failed (returned %d)", n_tokens);
            res.status = STATUS_ERROR_INFERENCE_FAILED;
            return res;
        }
        tokens.resize(n_tokens);
        res.tokens_in_prompt = (uint16_t)n_tokens;

        if (n_tokens + max_tokens > ctx_size) {
            log_err("Prompt too long: %d tokens + %d max_gen > %d ctx", n_tokens, max_tokens, ctx_size);
            res.status = STATUS_ERROR_PROMPT_TOO_LONG;
            return res;
        }

        if (mem) llama_memory_clear(mem, false);

        llama_batch batch = llama_batch_get_one(tokens.data(), n_tokens);
        decode_result = llama_decode(g_ctx, batch);
        llama_synchronize(g_ctx);

        res.prompt_eval_ms = (uint32_t)elapsed_ms(prompt_start);

        if (handle_decode_error(decode_result, "uncached prompt eval")) {
            return res;
        }
    }

    // ---- Set up sampler chain ----
    // Order: top_k -> top_p -> min_p -> temperature -> [grammar] -> dist
    llama_sampler_chain_params chain_params = llama_sampler_chain_default_params();
    chain_params.no_perf = true;  // skip perf timing in sampler

    llama_sampler* smpl = llama_sampler_chain_init(chain_params);
    llama_sampler_chain_add(smpl, llama_sampler_init_top_k(TOP_K));
    llama_sampler_chain_add(smpl, llama_sampler_init_top_p(TOP_P, 1));
    llama_sampler_chain_add(smpl, llama_sampler_init_min_p(MIN_P, 1));
    llama_sampler_chain_add(smpl, llama_sampler_init_temp(temperature));

    // V2: grammar sampler — masks invalid tokens before dist picks
    if (!grammar.empty()) {
        llama_sampler* grmr = llama_sampler_init_grammar(vocab, grammar.c_str(), "root");
        if (!grmr) {
            log_err("Failed to parse GBNF grammar");
            llama_sampler_free(smpl);
            if (mem) llama_memory_clear(mem, false);
            res.status = STATUS_ERROR_INFERENCE_FAILED;
            return res;
        }
        llama_sampler_chain_add(smpl, grmr);
    }

    llama_sampler_chain_add(smpl, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));

    // V2: pre-allocate candidates array for logprob extraction
    int32_t n_vocab = llama_vocab_n_tokens(vocab);
    std::vector<llama_token_data> candidates;
    if (top_n_logprobs > 0) {
        candidates.resize(n_vocab);
        res.logprobs.reserve(max_tokens);
    }

    // ---- Generation loop ----
    auto gen_start = std::chrono::steady_clock::now();
    std::string output;
    int generated = 0;
    bool hit_fatal_cuda = false;

    for (int i = 0; i < max_tokens; i++) {
        llama_token token;

        if (top_n_logprobs > 0) {
            // ---- V2 path: manual sampling with logprob extraction ----
            // Expand llama_sampler_sample() manually to capture logits

            // 1. Get raw logits for last position
            float* logits = llama_get_logits_ith(g_ctx, -1);

            // 2. Build candidate array from logits
            for (int32_t t = 0; t < n_vocab; t++) {
                candidates[t] = { t, logits[t], 0.0f };
            }
            llama_token_data_array cur_p = { candidates.data(), (size_t)n_vocab, -1, false };

            // 3. Apply all samplers (top_k, top_p, min_p, temp, grammar, dist)
            llama_sampler_apply(smpl, &cur_p);

            // 4. Read chosen token
            token = cur_p.data[cur_p.selected].id;

            // 5. Compute softmax over remaining candidates for probabilities
            // Find max logit for numerical stability
            float max_logit = -1e30f;
            for (size_t c = 0; c < cur_p.size; c++) {
                if (cur_p.data[c].logit > max_logit) max_logit = cur_p.data[c].logit;
            }
            float sum_exp = 0.0f;
            for (size_t c = 0; c < cur_p.size; c++) {
                cur_p.data[c].p = expf(cur_p.data[c].logit - max_logit);
                sum_exp += cur_p.data[c].p;
            }
            for (size_t c = 0; c < cur_p.size; c++) {
                cur_p.data[c].p /= sum_exp;
            }

            // 6. Build logprob entry — chosen token + top-N
            TokenLogprobEntry entry;
            float chosen_prob = cur_p.data[cur_p.selected].p;
            entry.chosen = { token, (chosen_prob > 0.0f) ? logf(chosen_prob) : -100.0f };

            // partial_sort to get top-N by probability
            size_t top_n = std::min((size_t)top_n_logprobs, cur_p.size);
            std::partial_sort(cur_p.data, cur_p.data + top_n, cur_p.data + cur_p.size,
                [](const llama_token_data& a, const llama_token_data& b) {
                    return a.p > b.p;
                });

            entry.top_candidates.reserve(top_n);
            for (size_t c = 0; c < top_n; c++) {
                float p = cur_p.data[c].p;
                entry.top_candidates.push_back({
                    cur_p.data[c].id,
                    (p > 0.0f) ? logf(p) : -100.0f
                });
            }
            res.logprobs.push_back(std::move(entry));

            // 7. Accept token to advance grammar state
            llama_sampler_accept(smpl, token);

        } else {
            // ---- V1 fast path: one-liner, no overhead ----
            token = llama_sampler_sample(smpl, g_ctx, -1);
        }

        // Check for end-of-generation
        if (llama_vocab_is_eog(vocab, token)) {
            break;
        }

        // Detokenize
        char piece[256];
        int piece_len = llama_token_to_piece(vocab, token, piece, sizeof(piece), 0, false);
        if (piece_len > 0) {
            output.append(piece, piece_len);
        }
        generated++;

        // Check wall clock timeout (Layer 4)
        if (elapsed_ms(request_start) > WALL_TIMEOUT_MS) {
            log_err("Wall clock timeout (%ds)", WALL_TIMEOUT_MS / 1000);
            g_abort_inference.store(true);
            res.status = STATUS_ERROR_TIMEOUT;
            break;
        }

        // Decode this token for next iteration (if not last)
        if (i < max_tokens - 1) {
            auto decode_start = std::chrono::steady_clock::now();

            llama_batch next_batch = llama_batch_get_one(&token, 1);
            decode_result = llama_decode(g_ctx, next_batch);

            // Per-decode watchdog (Layer 2)
            int64_t decode_ms = elapsed_ms(decode_start);
            if (decode_ms > DECODE_WATCHDOG_MS) {
                log_err("Decode watchdog: single call took %lldms (limit %dms)", decode_ms, DECODE_WATCHDOG_MS);
                g_abort_inference.store(true);
                res.status = STATUS_ERROR_TIMEOUT;
                break;
            }

            if (decode_result != 0) {
                if (decode_result == 2) {
                    // Aborted by callback (Layer 3)
                    res.status = STATUS_ERROR_TIMEOUT;
                    break;
                }
                // Check for sticky CUDA error
                if (decode_result < -1) {
                    cudaError_t cuda_err = cudaGetLastError();
                    if (cuda_err == cudaErrorIllegalAddress || cuda_err == cudaErrorLaunchFailure) {
                        log_err("FATAL: Sticky CUDA error %d during generation", (int)cuda_err);
                        hit_fatal_cuda = true;
                        res.status = STATUS_ERROR_INFERENCE_FAILED;
                        break;
                    }
                }
                log_err("Decode failed during generation (code %d)", decode_result);
                res.status = STATUS_ERROR_INFERENCE_FAILED;
                break;
            }
        }
    }

    res.gen_ms = (uint32_t)elapsed_ms(gen_start);
    res.output = output;
    res.tokens_generated = (uint16_t)generated;

    // Clean up sampler
    llama_sampler_free(smpl);

    // Clear KV cache — stateless
    if (mem) {
        llama_memory_clear(mem, false);
    }

    // Propagate fatal CUDA flag to caller
    res.fatal_cuda = hit_fatal_cuda;

    return res;
}

// ==================== REQUEST HANDLER ====================

struct ParsedRequest {
    uint16_t version;
    uint16_t request_type;
    uint16_t max_tokens;
    float    temperature;
    std::string system_prompt;
    std::string user_prompt;
    std::string grammar;            // V2: GBNF grammar string (empty = no constraint)
    uint8_t     top_n_logprobs = 0; // V2: 0 = no logprobs, 1-10 = top-N
    bool valid;
};

static ParsedRequest parse_request(const uint8_t* data, DWORD size) {
    ParsedRequest req = {};
    req.valid = false;

    if (size < REQUEST_HEADER_SIZE) {
        log_err("Request too short: %lu bytes (need %d)", size, REQUEST_HEADER_SIZE);
        return req;
    }

    uint32_t msg_len    = read_u32(&data[0]);
    req.version         = read_u16(&data[4]);
    req.request_type    = read_u16(&data[6]);
    req.max_tokens      = read_u16(&data[8]);
    uint16_t temp_x100  = read_u16(&data[10]);
    // bytes 12-15: reserved
    uint32_t sys_len    = read_u32(&data[16]);

    req.temperature = temp_x100 / 100.0f;

    // Version check
    if (req.version != PROTO_VERSION) {
        log_err("Unknown protocol version: %d (expected %d)", req.version, PROTO_VERSION);
        return req;
    }

    // For non-inference requests (except model swap), we don't need prompts
    if (req.request_type == REQ_HEALTH_CHECK || req.request_type == REQ_SHUTDOWN) {
        req.valid = true;
        return req;
    }

    // Model swap: payload is model path in the system_prompt field
    if (req.request_type == REQ_MODEL_SWAP) {
        if (REQUEST_HEADER_SIZE + sys_len > size) {
            log_err("Model swap path length %u exceeds message size %lu", sys_len, size);
            return req;
        }
        if (sys_len > 0) {
            req.system_prompt.assign((const char*)&data[REQUEST_HEADER_SIZE], sys_len);
        }
        req.valid = true;
        return req;
    }

    // V2 request: extract grammar_length and top_n_logprobs from header
    uint16_t grammar_len = 0;
    if (req.request_type == REQ_INFERENCE_V2) {
        grammar_len         = read_u16(&data[12]);  // bytes [12:14]
        req.top_n_logprobs  = data[14];             // byte [14]
        // byte [15] reserved
    }

    // Extract system prompt and user prompt from payload
    if (REQUEST_HEADER_SIZE + sys_len > size) {
        log_err("System prompt length %u exceeds message size %lu", sys_len, size);
        return req;
    }

    if (sys_len > 0) {
        req.system_prompt.assign((const char*)&data[REQUEST_HEADER_SIZE], sys_len);
    }

    // V2: payload order is system_prompt | grammar | user_prompt
    uint32_t grammar_start = REQUEST_HEADER_SIZE + sys_len;
    if (req.request_type == REQ_INFERENCE_V2 && grammar_len > 0) {
        if (grammar_start + grammar_len > size) {
            log_err("Grammar length %u exceeds message size %lu", grammar_len, size);
            return req;
        }
        req.grammar.assign((const char*)&data[grammar_start], grammar_len);
    }

    uint32_t user_start = grammar_start + grammar_len;
    uint32_t user_len   = size - user_start;
    if (user_len > 0) {
        req.user_prompt.assign((const char*)&data[user_start], user_len);
    }

    req.valid = true;
    return req;
}

// ==================== PIPE SERVER ====================

static int run_pipe_server(const RevanConfig& cfg) {
    // Store pipe name globally so ctrl_handler can unblock ConnectNamedPipe
    g_pipe_name = cfg.pipe_name;

    // Create named pipe
    HANDLE pipe = CreateNamedPipeA(
        cfg.pipe_name.c_str(),
        PIPE_ACCESS_DUPLEX,
        PIPE_TYPE_MESSAGE | PIPE_READMODE_MESSAGE | PIPE_WAIT,
        1,                    // max instances = 1 (single inference thread)
        PIPE_BUFFER_SIZE,     // output buffer
        PIPE_BUFFER_SIZE,     // input buffer
        0,                    // default timeout
        NULL                  // default security
    );

    if (pipe == INVALID_HANDLE_VALUE) {
        log_err("Failed to create pipe '%s' (error %lu)", cfg.pipe_name.c_str(), GetLastError());
        return 1;
    }

    log_info("Pipe ready: %s", cfg.pipe_name.c_str());

    std::vector<uint8_t> read_buf(PIPE_BUFFER_SIZE);

    while (!g_shutdown.load()) {
        // Wait for client connection
        BOOL connected = ConnectNamedPipe(pipe, NULL);
        if (!connected && GetLastError() != ERROR_PIPE_CONNECTED) {
            if (g_shutdown.load()) break;
            log_err("ConnectNamedPipe failed (error %lu)", GetLastError());
            continue;
        }

        // Read request
        DWORD bytes_read = 0;
        BOOL read_ok = ReadFile(pipe, read_buf.data(), (DWORD)read_buf.size(), &bytes_read, NULL);

        if (!read_ok || bytes_read == 0) {
            DWORD err = GetLastError();
            if (err == ERROR_BROKEN_PIPE) {
                // Client disconnected before sending — not an error
            } else if (!g_shutdown.load()) {
                log_err("ReadFile failed (error %lu)", err);
            }
            FlushFileBuffers(pipe);
            DisconnectNamedPipe(pipe);
            continue;
        }

        // Parse request
        ParsedRequest req = parse_request(read_buf.data(), bytes_read);
        if (!req.valid) {
            auto resp = build_response(STATUS_ERROR_INFERENCE_FAILED, 0, 0, 0, 0, "");
            send_response(pipe, resp);
            FlushFileBuffers(pipe);
            DisconnectNamedPipe(pipe);
            continue;
        }

        // Handle request type
        if (req.request_type == REQ_SHUTDOWN) {
            log_info("Shutdown requested. Cleaning up...");
            auto resp = build_response(STATUS_OK, 0, 0, 0, 0, "");
            send_response(pipe, resp);
            FlushFileBuffers(pipe);
            DisconnectNamedPipe(pipe);
            g_shutdown.store(true);
            break;
        }

        if (req.request_type == REQ_HEALTH_CHECK) {
            uint16_t status = (g_model && g_ctx) ? STATUS_OK : STATUS_ERROR_MODEL_NOT_LOADED;
            auto resp = build_response(status, 0, 0, 0, 0, "");
            send_response(pipe, resp);
            FlushFileBuffers(pipe);
            DisconnectNamedPipe(pipe);
            continue;
        }

        if (req.request_type == REQ_MODEL_SWAP) {
            log_info("Model swap requested: %s", req.system_prompt.c_str());

            // Invalidate system prompt KV cache (model is changing)
            g_sys_cache.clear();
            g_cache_counter = 0;

            // Free existing context and model
            if (g_ctx) { llama_free(g_ctx); g_ctx = nullptr; }
            if (g_model) { llama_model_free(g_model); g_model = nullptr; }

            // Update config and reload
            // NOTE: cfg is const ref, so we make a mutable copy for the swap
            RevanConfig swap_cfg = cfg;
            swap_cfg.model_path = req.system_prompt;

            bool ok = load_model(swap_cfg);
            if (ok) warmup(swap_cfg);

            uint16_t status = ok ? STATUS_OK : STATUS_ERROR_MODEL_NOT_LOADED;
            std::string msg = ok ? "model loaded" : "load failed";
            auto resp = build_response(status, 0, 0, 0, 0, msg);
            send_response(pipe, resp);
            FlushFileBuffers(pipe);
            DisconnectNamedPipe(pipe);

            if (!ok) {
                log_err("Model swap failed, engine has no model loaded");
            }
            continue;
        }

        if (req.request_type == REQ_INFERENCE || req.request_type == REQ_INFERENCE_V2) {
            // Clamp max_tokens to configured max
            int max_tok = (req.max_tokens > 0 && req.max_tokens <= cfg.n_predict_max)
                        ? req.max_tokens : cfg.n_predict_max;
            float temp = (req.temperature >= 0.0f) ? req.temperature : cfg.temperature_default;

            bool is_v2 = (req.request_type == REQ_INFERENCE_V2);

            if (is_v2) {
                log_info("REQ_V2 sys=%zuB usr=%zuB grammar=%zuB top_n=%d max_tokens=%d temp=%.2f",
                         req.system_prompt.size(), req.user_prompt.size(),
                         req.grammar.size(), req.top_n_logprobs, max_tok, temp);
            } else {
                log_info("REQ  sys=%zuB usr=%zuB max_tokens=%d temp=%.2f",
                         req.system_prompt.size(), req.user_prompt.size(), max_tok, temp);
            }

            // Run inference — V1 passes empty grammar + 0 logprobs (no overhead)
            InferenceResult result = run_inference(
                req.system_prompt, req.user_prompt, max_tok, temp,
                req.grammar, req.top_n_logprobs
            );

            // Log result
            if (result.status == STATUS_OK) {
                log_info("DONE output=\"%s\" gen=%dtok pp=%ums gen=%ums total=%ums",
                         result.output.c_str(), result.tokens_generated,
                         result.prompt_eval_ms, result.gen_ms,
                         result.prompt_eval_ms + result.gen_ms);
            } else {
                log_info("FAIL status=0x%04X total=%ums",
                         result.status, result.prompt_eval_ms + result.gen_ms);
            }

            // Send response — V2 gets logprobs, V1 gets standard response
            std::vector<uint8_t> resp;
            if (is_v2) {
                resp = build_response_v2(
                    result.status,
                    result.tokens_generated,
                    result.tokens_in_prompt,
                    result.prompt_eval_ms,
                    result.gen_ms,
                    result.output,
                    req.top_n_logprobs,
                    result.logprobs
                );
            } else {
                resp = build_response(
                    result.status,
                    result.tokens_generated,
                    result.tokens_in_prompt,
                    result.prompt_eval_ms,
                    result.gen_ms,
                    result.output
                );
            }
            send_response(pipe, resp);
            FlushFileBuffers(pipe);
            DisconnectNamedPipe(pipe);

            // Exit on fatal CUDA error (exit code 2)
            if (result.fatal_cuda) {
                log_err("Exiting due to sticky CUDA error");
                CloseHandle(pipe);
                return 2;
            }

            continue;
        }

        // Unknown request type
        log_err("Unknown request type: 0x%04X", req.request_type);
        auto resp = build_response(STATUS_ERROR_INFERENCE_FAILED, 0, 0, 0, 0, "");
        send_response(pipe, resp);
        FlushFileBuffers(pipe);
        DisconnectNamedPipe(pipe);
    }

    CloseHandle(pipe);
    return 0;
}

// ==================== MAIN ====================

int main(int argc, char* argv[]) {
    // ---- Parse command line ----
    std::string config_path;

    for (int i = 1; i < argc; i++) {
        if ((strcmp(argv[i], "--config") == 0 || strcmp(argv[i], "-c") == 0) && i + 1 < argc) {
            config_path = argv[++i];
        }
    }

    // If no --config given, look for revan.toml in current directory, then exe directory
    if (config_path.empty()) {
        if (fs::exists("revan.toml")) {
            config_path = "revan.toml";
        } else {
            // Try exe directory
            fs::path exe_dir = fs::path(argv[0]).parent_path();
            fs::path candidate = exe_dir / "revan.toml";
            if (fs::exists(candidate)) {
                config_path = candidate.string();
            }
        }
    }

    if (config_path.empty()) {
        log_err("No config file found. Use --config <path> or place revan.toml in the current directory.");
        return 1;
    }

    // Resolve root directory — the directory containing revan.toml
    fs::path config_abs = fs::absolute(config_path);
    if (!fs::exists(config_abs)) {
        log_err("Config file not found: %s", config_abs.string().c_str());
        return 1;
    }

    // ---- Load config ----
    RevanConfig cfg;
    cfg.root_dir = config_abs.parent_path().string();

    if (!parse_config(config_abs.string(), cfg)) {
        return 1;
    }

    log_info("Config loaded from: %s", config_abs.string().c_str());
    log_info("Root directory: %s", cfg.root_dir.c_str());

    // ---- Set up llama.cpp logging ----
    llama_log_set(llama_log_callback, nullptr);

    // ---- Initialize backend ----
    llama_backend_init();

    // ---- Register Ctrl+C handler ----
    SetConsoleCtrlHandler(ctrl_handler, TRUE);

    // ---- Load model (startup fail-fast) ----
    if (!load_model(cfg)) {
        llama_backend_free();
        return 1;
    }

    // ---- Warm-up ----
    warmup(cfg);

    // ---- Run pipe server ----
    int exit_code = run_pipe_server(cfg);

    // ---- Cleanup ----
    log_info("Freeing model and context...");
    if (g_ctx) {
        llama_free(g_ctx);
        g_ctx = nullptr;
    }
    if (g_model) {
        llama_model_free(g_model);
        g_model = nullptr;
    }
    llama_backend_free();

    log_info("Shutdown complete (exit code %d)", exit_code);
    return exit_code;
}
