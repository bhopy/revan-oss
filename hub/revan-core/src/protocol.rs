// protocol.rs — Binary protocol for Revan Named Pipe IPC
// Exact port of revan.cpp binary format (little-endian throughout).
//
// Request layout (20-byte header):
//   [0:4]   u32 msg_len (total message size including header)
//   [4:6]   u16 version (1)
//   [6:8]   u16 request_type
//   [8:10]  u16 max_tokens
//   [10:12] u16 temperature * 100
//   [12:16] u32 reserved (0)
//   [16:20] u32 system_prompt_length
//   [20..20+sys_len]  system prompt (UTF-8)
//   [20+sys_len..end] user prompt (UTF-8)
//
// Response layout (24-byte header):
//   [0:4]   u32 msg_len
//   [4:6]   u16 version (1)
//   [6:8]   u16 status
//   [8:10]  u16 tokens_generated
//   [10:12] u16 tokens_in_prompt
//   [12:16] u32 prompt_eval_ms
//   [16:20] u32 gen_ms
//   [20:24] u32 reserved
//   [24..end] output_text (UTF-8)

use thiserror::Error;

// Protocol version — must match revan.cpp
pub const PROTO_VERSION: u16 = 1;

// Request header size in bytes
pub const REQUEST_HEADER_SIZE: usize = 20;

// Response header size in bytes
pub const RESPONSE_HEADER_SIZE: usize = 24;

// Request types
pub const REQ_INFERENCE: u16 = 0x0001;
pub const REQ_HEALTH_CHECK: u16 = 0x0002;
pub const REQ_SHUTDOWN: u16 = 0x0003;
pub const REQ_INFERENCE_V2: u16 = 0x0004;  // grammar + logprobs
pub const REQ_MODEL_SWAP: u16 = 0x0005;   // hot model swap

// Response status codes
pub const STATUS_OK: u16 = 0x0000;
pub const STATUS_ERROR_MODEL_NOT_LOADED: u16 = 0x0001;
pub const STATUS_ERROR_PROMPT_TOO_LONG: u16 = 0x0002;
pub const STATUS_ERROR_INFERENCE_FAILED: u16 = 0x0003;
pub const STATUS_ERROR_TIMEOUT: u16 = 0x0004;

// ==================== V2 LOGPROB TYPES ====================

/// Single token + its log-probability
#[derive(Debug, Clone)]
pub struct TokenLogprob {
    pub token_id: i32,
    pub logprob: f32,  // natural log probability (ln)
}

/// Per-generated-token entry: chosen token + top-N candidates
#[derive(Debug, Clone)]
pub struct TokenLogprobEntry {
    pub chosen: TokenLogprob,
    pub top_candidates: Vec<TokenLogprob>,
}

// ==================== REQUEST / RESPONSE ====================

#[derive(Debug, Clone)]
pub struct RevanRequest {
    pub system_prompt: String,
    pub user_prompt: String,
    pub max_tokens: u16,
    /// Temperature as float (0.0 - 2.0). Encoded as temp*100 on wire.
    pub temperature: f32,
    /// V2: GBNF grammar string. None = no constraint (V1 behavior).
    pub grammar: Option<String>,
    /// V2: Number of top-N logprobs to return. None/0 = no logprobs (V1 behavior).
    pub top_n_logprobs: u8,
}

#[derive(Debug, Clone)]
pub struct RevanResponse {
    pub status: u16,
    pub output: String,
    pub tokens_generated: u16,
    pub tokens_in_prompt: u16,
    pub prompt_eval_ms: u32,
    pub gen_ms: u32,
    /// V2: per-token logprob data. Empty for V1 responses.
    pub logprobs: Vec<TokenLogprobEntry>,
}

#[derive(Debug, Error)]
pub enum ProtocolError {
    #[error("response too short: {0} bytes (need {RESPONSE_HEADER_SIZE})")]
    ResponseTooShort(usize),
    #[error("protocol version mismatch: got {0}, expected {PROTO_VERSION}")]
    VersionMismatch(u16),
    #[error("invalid UTF-8 in response output")]
    InvalidUtf8,
}

impl RevanResponse {
    /// Human-readable status name
    pub fn status_name(&self) -> &'static str {
        match self.status {
            STATUS_OK => "OK",
            STATUS_ERROR_MODEL_NOT_LOADED => "ERROR_MODEL_NOT_LOADED",
            STATUS_ERROR_PROMPT_TOO_LONG => "ERROR_PROMPT_TOO_LONG",
            STATUS_ERROR_INFERENCE_FAILED => "ERROR_INFERENCE_FAILED",
            STATUS_ERROR_TIMEOUT => "ERROR_TIMEOUT",
            _ => "UNKNOWN",
        }
    }

    /// True if status is OK
    pub fn is_ok(&self) -> bool {
        self.status == STATUS_OK
    }

    /// Total round-trip time in ms (prompt eval + generation)
    pub fn total_ms(&self) -> u32 {
        self.prompt_eval_ms + self.gen_ms
    }
}

// ==================== ENCODING ====================

/// Encode an inference request into the binary wire format.
/// Auto-selects V2 when grammar or logprobs are requested.
pub fn encode_request(req: &RevanRequest) -> Vec<u8> {
    let has_v2_features = req.grammar.is_some() || req.top_n_logprobs > 0;
    if has_v2_features {
        return encode_request_v2(req);
    }

    let sys_bytes = req.system_prompt.as_bytes();
    let usr_bytes = req.user_prompt.as_bytes();
    let payload_len = sys_bytes.len() + usr_bytes.len();
    let msg_len = REQUEST_HEADER_SIZE + payload_len;

    let mut buf = Vec::with_capacity(msg_len);

    // Header (20 bytes)
    buf.extend_from_slice(&(msg_len as u32).to_le_bytes());        // [0:4]  msg_len
    buf.extend_from_slice(&PROTO_VERSION.to_le_bytes());            // [4:6]  version
    buf.extend_from_slice(&REQ_INFERENCE.to_le_bytes());            // [6:8]  request type
    buf.extend_from_slice(&req.max_tokens.to_le_bytes());           // [8:10] max_tokens
    let temp_x100 = (req.temperature * 100.0) as u16;
    buf.extend_from_slice(&temp_x100.to_le_bytes());                // [10:12] temperature*100
    buf.extend_from_slice(&0u32.to_le_bytes());                     // [12:16] reserved
    buf.extend_from_slice(&(sys_bytes.len() as u32).to_le_bytes()); // [16:20] sys_prompt_len

    // Payload
    buf.extend_from_slice(sys_bytes);
    buf.extend_from_slice(usr_bytes);

    buf
}

/// Encode a V2 inference request with grammar + logprobs support.
/// Header bytes [12:14] = grammar_length, [14] = top_n_logprobs, [15] = reserved.
/// Payload: system_prompt | grammar | user_prompt.
pub fn encode_request_v2(req: &RevanRequest) -> Vec<u8> {
    let sys_bytes = req.system_prompt.as_bytes();
    let usr_bytes = req.user_prompt.as_bytes();
    let grammar_bytes = req.grammar.as_deref().unwrap_or("").as_bytes();
    let payload_len = sys_bytes.len() + grammar_bytes.len() + usr_bytes.len();
    let msg_len = REQUEST_HEADER_SIZE + payload_len;

    let mut buf = Vec::with_capacity(msg_len);

    // Header (20 bytes)
    buf.extend_from_slice(&(msg_len as u32).to_le_bytes());        // [0:4]  msg_len
    buf.extend_from_slice(&PROTO_VERSION.to_le_bytes());            // [4:6]  version
    buf.extend_from_slice(&REQ_INFERENCE_V2.to_le_bytes());         // [6:8]  request type
    buf.extend_from_slice(&req.max_tokens.to_le_bytes());           // [8:10] max_tokens
    let temp_x100 = (req.temperature * 100.0) as u16;
    buf.extend_from_slice(&temp_x100.to_le_bytes());                // [10:12] temperature*100
    buf.extend_from_slice(&(grammar_bytes.len() as u16).to_le_bytes()); // [12:14] grammar_length
    buf.push(req.top_n_logprobs);                                   // [14]   top_n_logprobs
    buf.push(0);                                                    // [15]   reserved
    buf.extend_from_slice(&(sys_bytes.len() as u32).to_le_bytes()); // [16:20] sys_prompt_len

    // Payload: system_prompt | grammar | user_prompt
    buf.extend_from_slice(sys_bytes);
    buf.extend_from_slice(grammar_bytes);
    buf.extend_from_slice(usr_bytes);

    buf
}

/// Encode a health check request (no payload).
pub fn encode_health_check() -> Vec<u8> {
    let msg_len = REQUEST_HEADER_SIZE as u32;
    let mut buf = Vec::with_capacity(REQUEST_HEADER_SIZE);

    buf.extend_from_slice(&msg_len.to_le_bytes());       // [0:4]
    buf.extend_from_slice(&PROTO_VERSION.to_le_bytes());  // [4:6]
    buf.extend_from_slice(&REQ_HEALTH_CHECK.to_le_bytes()); // [6:8]
    buf.extend_from_slice(&0u16.to_le_bytes());           // [8:10]  max_tokens (unused)
    buf.extend_from_slice(&0u16.to_le_bytes());           // [10:12] temperature (unused)
    buf.extend_from_slice(&0u32.to_le_bytes());           // [12:16] reserved
    buf.extend_from_slice(&0u32.to_le_bytes());           // [16:20] sys_prompt_len (unused)

    buf
}

/// Encode a shutdown request (no payload).
pub fn encode_shutdown() -> Vec<u8> {
    let msg_len = REQUEST_HEADER_SIZE as u32;
    let mut buf = Vec::with_capacity(REQUEST_HEADER_SIZE);

    buf.extend_from_slice(&msg_len.to_le_bytes());
    buf.extend_from_slice(&PROTO_VERSION.to_le_bytes());
    buf.extend_from_slice(&REQ_SHUTDOWN.to_le_bytes());
    buf.extend_from_slice(&0u16.to_le_bytes());
    buf.extend_from_slice(&0u16.to_le_bytes());
    buf.extend_from_slice(&0u32.to_le_bytes());
    buf.extend_from_slice(&0u32.to_le_bytes());

    buf
}

/// Encode a model swap request. The model path is carried in the system_prompt field.
pub fn encode_model_swap(model_path: &str) -> Vec<u8> {
    let path_bytes = model_path.as_bytes();
    let msg_len = (REQUEST_HEADER_SIZE + path_bytes.len()) as u32;
    let mut buf = Vec::with_capacity(msg_len as usize);

    buf.extend_from_slice(&msg_len.to_le_bytes());                  // [0:4]  msg_len
    buf.extend_from_slice(&PROTO_VERSION.to_le_bytes());             // [4:6]  version
    buf.extend_from_slice(&REQ_MODEL_SWAP.to_le_bytes());            // [6:8]  request type
    buf.extend_from_slice(&0u16.to_le_bytes());                      // [8:10] unused
    buf.extend_from_slice(&0u16.to_le_bytes());                      // [10:12] unused
    buf.extend_from_slice(&0u32.to_le_bytes());                      // [12:16] reserved
    buf.extend_from_slice(&(path_bytes.len() as u32).to_le_bytes()); // [16:20] path_len

    buf.extend_from_slice(path_bytes);

    buf
}

// ==================== DECODING ====================

/// Decode a V1 binary response from Revan.
pub fn decode_response(data: &[u8]) -> Result<RevanResponse, ProtocolError> {
    if data.len() < RESPONSE_HEADER_SIZE {
        return Err(ProtocolError::ResponseTooShort(data.len()));
    }

    // Parse 24-byte header (all little-endian)
    let _msg_len = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
    let version = u16::from_le_bytes([data[4], data[5]]);
    let status = u16::from_le_bytes([data[6], data[7]]);
    let tokens_generated = u16::from_le_bytes([data[8], data[9]]);
    let tokens_in_prompt = u16::from_le_bytes([data[10], data[11]]);
    let prompt_eval_ms = u32::from_le_bytes([data[12], data[13], data[14], data[15]]);
    let gen_ms = u32::from_le_bytes([data[16], data[17], data[18], data[19]]);
    // bytes 20-23: reserved

    if version != PROTO_VERSION {
        return Err(ProtocolError::VersionMismatch(version));
    }

    // Output text is everything after the 24-byte header
    let output = if data.len() > RESPONSE_HEADER_SIZE {
        std::str::from_utf8(&data[RESPONSE_HEADER_SIZE..])
            .map_err(|_| ProtocolError::InvalidUtf8)?
            .to_string()
    } else {
        String::new()
    };

    Ok(RevanResponse {
        status,
        output,
        tokens_generated,
        tokens_in_prompt,
        prompt_eval_ms,
        gen_ms,
        logprobs: Vec::new(),
    })
}

/// Decode a V2 binary response from Revan (with logprobs section).
/// Header bytes [20:22] = output_text_length, [22] = top_n, [23] = reserved.
pub fn decode_response_v2(data: &[u8]) -> Result<RevanResponse, ProtocolError> {
    if data.len() < RESPONSE_HEADER_SIZE {
        return Err(ProtocolError::ResponseTooShort(data.len()));
    }

    let _msg_len = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
    let version = u16::from_le_bytes([data[4], data[5]]);
    let status = u16::from_le_bytes([data[6], data[7]]);
    let tokens_generated = u16::from_le_bytes([data[8], data[9]]);
    let tokens_in_prompt = u16::from_le_bytes([data[10], data[11]]);
    let prompt_eval_ms = u32::from_le_bytes([data[12], data[13], data[14], data[15]]);
    let gen_ms = u32::from_le_bytes([data[16], data[17], data[18], data[19]]);
    let output_text_len = u16::from_le_bytes([data[20], data[21]]) as usize;
    let top_n = data[22] as usize;

    if version != PROTO_VERSION {
        return Err(ProtocolError::VersionMismatch(version));
    }

    // Extract output text
    let text_start = RESPONSE_HEADER_SIZE;
    let text_end = text_start + output_text_len;
    let output = if output_text_len > 0 && data.len() >= text_end {
        std::str::from_utf8(&data[text_start..text_end])
            .map_err(|_| ProtocolError::InvalidUtf8)?
            .to_string()
    } else {
        String::new()
    };

    // Parse logprobs section
    let mut logprobs = Vec::new();
    if top_n > 0 && tokens_generated > 0 {
        let per_token_size = 8 + 8 * top_n;  // 8 bytes chosen + 8*top_n candidates
        let logprobs_start = text_end;
        logprobs.reserve(tokens_generated as usize);

        for t in 0..tokens_generated as usize {
            let base = logprobs_start + t * per_token_size;
            if base + per_token_size > data.len() {
                break;  // truncated data — return what we have
            }

            let chosen_id = i32::from_le_bytes([data[base], data[base+1], data[base+2], data[base+3]]);
            let chosen_lp = f32::from_le_bytes([data[base+4], data[base+5], data[base+6], data[base+7]]);

            let mut top_candidates = Vec::with_capacity(top_n);
            for c in 0..top_n {
                let off = base + 8 + c * 8;
                let cid = i32::from_le_bytes([data[off], data[off+1], data[off+2], data[off+3]]);
                let clp = f32::from_le_bytes([data[off+4], data[off+5], data[off+6], data[off+7]]);
                top_candidates.push(TokenLogprob { token_id: cid, logprob: clp });
            }

            logprobs.push(TokenLogprobEntry {
                chosen: TokenLogprob { token_id: chosen_id, logprob: chosen_lp },
                top_candidates,
            });
        }
    }

    Ok(RevanResponse {
        status,
        output,
        tokens_generated,
        tokens_in_prompt,
        prompt_eval_ms,
        gen_ms,
        logprobs,
    })
}

// ==================== TESTS ====================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encode_decode_inference_roundtrip() {
        // Encode a V1 request (no grammar, no logprobs) and verify structure
        let req = RevanRequest {
            system_prompt: "Classify.".to_string(),
            user_prompt: "Hello".to_string(),
            max_tokens: 5,
            temperature: 0.1,
            grammar: None,
            top_n_logprobs: 0,
        };
        let encoded = encode_request(&req);

        // Verify header
        let msg_len = u32::from_le_bytes([encoded[0], encoded[1], encoded[2], encoded[3]]);
        assert_eq!(msg_len as usize, encoded.len());

        let version = u16::from_le_bytes([encoded[4], encoded[5]]);
        assert_eq!(version, PROTO_VERSION);

        let req_type = u16::from_le_bytes([encoded[6], encoded[7]]);
        assert_eq!(req_type, REQ_INFERENCE);

        let max_tok = u16::from_le_bytes([encoded[8], encoded[9]]);
        assert_eq!(max_tok, 5);

        let temp = u16::from_le_bytes([encoded[10], encoded[11]]);
        assert_eq!(temp, 10); // 0.1 * 100

        let sys_len = u32::from_le_bytes([encoded[16], encoded[17], encoded[18], encoded[19]]);
        assert_eq!(sys_len, 9); // "Classify." = 9 bytes

        // Verify payload
        let sys = &encoded[20..20 + sys_len as usize];
        assert_eq!(sys, b"Classify.");
        let usr = &encoded[20 + sys_len as usize..];
        assert_eq!(usr, b"Hello");
    }

    #[test]
    fn test_encode_health_check() {
        let encoded = encode_health_check();
        assert_eq!(encoded.len(), REQUEST_HEADER_SIZE);

        let req_type = u16::from_le_bytes([encoded[6], encoded[7]]);
        assert_eq!(req_type, REQ_HEALTH_CHECK);
    }

    #[test]
    fn test_encode_shutdown() {
        let encoded = encode_shutdown();
        assert_eq!(encoded.len(), REQUEST_HEADER_SIZE);

        let req_type = u16::from_le_bytes([encoded[6], encoded[7]]);
        assert_eq!(req_type, REQ_SHUTDOWN);
    }

    #[test]
    fn test_decode_ok_response() {
        // Build a response matching revan.cpp build_response()
        let output = "calculator";
        let msg_len = (RESPONSE_HEADER_SIZE + output.len()) as u32;

        let mut data = Vec::new();
        data.extend_from_slice(&msg_len.to_le_bytes());     // [0:4]
        data.extend_from_slice(&PROTO_VERSION.to_le_bytes()); // [4:6]
        data.extend_from_slice(&STATUS_OK.to_le_bytes());    // [6:8]
        data.extend_from_slice(&1u16.to_le_bytes());         // [8:10]  tokens_gen
        data.extend_from_slice(&63u16.to_le_bytes());        // [10:12] tokens_prompt
        data.extend_from_slice(&1200u32.to_le_bytes());      // [12:16] pp_ms
        data.extend_from_slice(&400u32.to_le_bytes());       // [16:20] gen_ms
        data.extend_from_slice(&0u32.to_le_bytes());         // [20:24] reserved
        data.extend_from_slice(output.as_bytes());           // [24:]

        let resp = decode_response(&data).unwrap();
        assert_eq!(resp.status, STATUS_OK);
        assert_eq!(resp.output, "calculator");
        assert_eq!(resp.tokens_generated, 1);
        assert_eq!(resp.tokens_in_prompt, 63);
        assert_eq!(resp.prompt_eval_ms, 1200);
        assert_eq!(resp.gen_ms, 400);
        assert_eq!(resp.total_ms(), 1600);
        assert!(resp.is_ok());
    }

    #[test]
    fn test_decode_error_response() {
        let msg_len = RESPONSE_HEADER_SIZE as u32;
        let mut data = Vec::new();
        data.extend_from_slice(&msg_len.to_le_bytes());
        data.extend_from_slice(&PROTO_VERSION.to_le_bytes());
        data.extend_from_slice(&STATUS_ERROR_TIMEOUT.to_le_bytes());
        data.extend_from_slice(&0u16.to_le_bytes());
        data.extend_from_slice(&0u16.to_le_bytes());
        data.extend_from_slice(&0u32.to_le_bytes());
        data.extend_from_slice(&0u32.to_le_bytes());
        data.extend_from_slice(&0u32.to_le_bytes());

        let resp = decode_response(&data).unwrap();
        assert_eq!(resp.status, STATUS_ERROR_TIMEOUT);
        assert!(!resp.is_ok());
        assert_eq!(resp.status_name(), "ERROR_TIMEOUT");
    }

    #[test]
    fn test_decode_too_short() {
        let data = vec![0u8; 10];
        assert!(decode_response(&data).is_err());
    }

    #[test]
    fn test_decode_version_mismatch() {
        let mut data = vec![0u8; RESPONSE_HEADER_SIZE];
        data[0] = RESPONSE_HEADER_SIZE as u8;
        // Set version to 99
        data[4] = 99;
        data[5] = 0;

        assert!(matches!(
            decode_response(&data),
            Err(ProtocolError::VersionMismatch(99))
        ));
    }

    // ==================== V2 TESTS ====================

    #[test]
    fn test_encode_v2_request_with_grammar() {
        let req = RevanRequest {
            system_prompt: "Classify.".to_string(),
            user_prompt: "Is sky blue?".to_string(),
            max_tokens: 3,
            temperature: 0.1,
            grammar: Some(r#"root ::= " yes" | " no""#.to_string()),
            top_n_logprobs: 5,
        };
        let encoded = encode_request(&req);

        // Should auto-select V2
        let req_type = u16::from_le_bytes([encoded[6], encoded[7]]);
        assert_eq!(req_type, REQ_INFERENCE_V2);

        // Grammar length at [12:14]
        let grammar_len = u16::from_le_bytes([encoded[12], encoded[13]]);
        assert_eq!(grammar_len as usize, req.grammar.as_ref().unwrap().len());

        // top_n at [14]
        assert_eq!(encoded[14], 5);

        // Payload: sys_prompt | grammar | user_prompt
        let sys_len = u32::from_le_bytes([encoded[16], encoded[17], encoded[18], encoded[19]]) as usize;
        assert_eq!(sys_len, 9); // "Classify."

        let sys = &encoded[20..20+sys_len];
        assert_eq!(sys, b"Classify.");

        let grammar = &encoded[20+sys_len..20+sys_len+grammar_len as usize];
        assert_eq!(grammar, req.grammar.as_ref().unwrap().as_bytes());

        let user = &encoded[20+sys_len+grammar_len as usize..];
        assert_eq!(user, b"Is sky blue?");
    }

    #[test]
    fn test_v1_request_no_grammar() {
        // Request with no grammar/logprobs should encode as V1
        let req = RevanRequest {
            system_prompt: "Test.".to_string(),
            user_prompt: "Hello".to_string(),
            max_tokens: 5,
            temperature: 0.1,
            grammar: None,
            top_n_logprobs: 0,
        };
        let encoded = encode_request(&req);
        let req_type = u16::from_le_bytes([encoded[6], encoded[7]]);
        assert_eq!(req_type, REQ_INFERENCE); // V1, not V2
    }

    #[test]
    fn test_decode_response_v2_with_logprobs() {
        // Build a fake V2 response with 2 tokens, top_n=2
        let output = " yes";
        let top_n: u8 = 2;
        let tokens_gen: u16 = 2;
        let text_len = output.len() as u16;
        let per_token = 8 + 8 * top_n as usize;
        let logprobs_size = tokens_gen as usize * per_token;
        let msg_len = (RESPONSE_HEADER_SIZE + text_len as usize + logprobs_size) as u32;

        let mut data = Vec::new();
        data.extend_from_slice(&msg_len.to_le_bytes());       // [0:4]
        data.extend_from_slice(&PROTO_VERSION.to_le_bytes()); // [4:6]
        data.extend_from_slice(&STATUS_OK.to_le_bytes());     // [6:8]
        data.extend_from_slice(&tokens_gen.to_le_bytes());    // [8:10]
        data.extend_from_slice(&50u16.to_le_bytes());         // [10:12] tokens_prompt
        data.extend_from_slice(&100u32.to_le_bytes());        // [12:16] pp_ms
        data.extend_from_slice(&50u32.to_le_bytes());         // [16:20] gen_ms
        data.extend_from_slice(&text_len.to_le_bytes());      // [20:22] output_text_len
        data.push(top_n);                                     // [22] top_n
        data.push(0);                                         // [23] reserved

        // Output text
        data.extend_from_slice(output.as_bytes());

        // Logprobs: 2 tokens, each with chosen + 2 candidates
        // Token 1: chosen=42 lp=-0.1, candidates: 42/-0.1, 99/-2.3
        data.extend_from_slice(&42i32.to_le_bytes());
        data.extend_from_slice(&(-0.1f32).to_le_bytes());
        data.extend_from_slice(&42i32.to_le_bytes());
        data.extend_from_slice(&(-0.1f32).to_le_bytes());
        data.extend_from_slice(&99i32.to_le_bytes());
        data.extend_from_slice(&(-2.3f32).to_le_bytes());

        // Token 2: chosen=77 lp=-0.05, candidates: 77/-0.05, 88/-3.0
        data.extend_from_slice(&77i32.to_le_bytes());
        data.extend_from_slice(&(-0.05f32).to_le_bytes());
        data.extend_from_slice(&77i32.to_le_bytes());
        data.extend_from_slice(&(-0.05f32).to_le_bytes());
        data.extend_from_slice(&88i32.to_le_bytes());
        data.extend_from_slice(&(-3.0f32).to_le_bytes());

        let resp = decode_response_v2(&data).unwrap();
        assert_eq!(resp.status, STATUS_OK);
        assert_eq!(resp.output, " yes");
        assert_eq!(resp.tokens_generated, 2);
        assert_eq!(resp.logprobs.len(), 2);

        // Check first token logprob
        assert_eq!(resp.logprobs[0].chosen.token_id, 42);
        assert!((resp.logprobs[0].chosen.logprob - (-0.1)).abs() < 0.001);
        assert_eq!(resp.logprobs[0].top_candidates.len(), 2);
        assert_eq!(resp.logprobs[0].top_candidates[1].token_id, 99);

        // Check second token logprob
        assert_eq!(resp.logprobs[1].chosen.token_id, 77);
        assert_eq!(resp.logprobs[1].top_candidates.len(), 2);
    }

    #[test]
    fn test_encode_model_swap() {
        let path = "models/Llama-3.2-3B-Q4_K_M.gguf";
        let encoded = encode_model_swap(path);

        // Verify header
        let msg_len = u32::from_le_bytes([encoded[0], encoded[1], encoded[2], encoded[3]]);
        assert_eq!(msg_len as usize, encoded.len());

        let version = u16::from_le_bytes([encoded[4], encoded[5]]);
        assert_eq!(version, PROTO_VERSION);

        let req_type = u16::from_le_bytes([encoded[6], encoded[7]]);
        assert_eq!(req_type, REQ_MODEL_SWAP);

        let path_len = u32::from_le_bytes([encoded[16], encoded[17], encoded[18], encoded[19]]);
        assert_eq!(path_len as usize, path.len());

        // Verify payload
        let payload = &encoded[20..];
        assert_eq!(payload, path.as_bytes());
    }
}
