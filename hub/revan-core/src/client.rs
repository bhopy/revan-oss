// client.rs — Synchronous Win32 Named Pipe client for Revan
//
// Uses the `windows` crate for CreateFileW + SetNamedPipeHandleState +
// WriteFile + ReadFile. All calls are blocking (Revan takes 2-3s per
// request anyway — async buys nothing here).
//
// Designed to be wrapped in tokio::task::spawn_blocking by the hub.

use crate::protocol::{
    RevanRequest, RevanResponse,
    encode_request, encode_health_check, encode_shutdown,
    decode_response, decode_response_v2,
    ProtocolError,
};
use thiserror::Error;
use tracing::debug;

#[cfg(windows)]
use windows::{
    core::HSTRING,
    Win32::Foundation::CloseHandle,
    Win32::Storage::FileSystem::{
        CreateFileW, ReadFile, WriteFile,
        FILE_SHARE_NONE, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL,
    },
    Win32::System::Pipes::{SetNamedPipeHandleState, PIPE_READMODE_MESSAGE},
};

// Pipe buffer size — matches revan.cpp PIPE_BUFFER_SIZE
const PIPE_BUFFER_SIZE: usize = 65536;

#[derive(Debug, Error)]
pub enum ClientError {
    #[error("failed to connect to pipe '{0}': {1}")]
    ConnectFailed(String, String),
    #[error("failed to set pipe message mode: {0}")]
    SetModeFailed(String),
    #[error("write failed: {0}")]
    WriteFailed(String),
    #[error("read failed: {0}")]
    ReadFailed(String),
    #[error("protocol error: {0}")]
    Protocol(#[from] ProtocolError),
    #[error("revan returned error: {0} (status 0x{1:04X})")]
    RevanError(String, u16),
    #[error("not supported on this platform")]
    UnsupportedPlatform,
}

/// Synchronous client for the Revan C++ inference engine.
/// Connects via Windows Named Pipe in message mode.
#[derive(Debug, Clone)]
pub struct RevanClient {
    pipe_name: String,
}

impl RevanClient {
    /// Create a new client targeting the given pipe name.
    /// Does NOT connect yet — each call opens a fresh connection.
    pub fn new(pipe_name: &str) -> Self {
        Self {
            pipe_name: pipe_name.to_string(),
        }
    }

    /// Send an inference request to Revan (blocking).
    /// Opens pipe, writes request, reads response, closes pipe.
    /// Auto-detects V2 when grammar or logprobs are present.
    pub fn send(&self, req: &RevanRequest) -> Result<RevanResponse, ClientError> {
        let is_v2 = req.grammar.is_some() || req.top_n_logprobs > 0;
        let data = encode_request(req);
        let raw_resp = self.send_raw_bytes(&data)?;
        if is_v2 {
            decode_response_v2(&raw_resp).map_err(ClientError::Protocol)
        } else {
            decode_response(&raw_resp).map_err(ClientError::Protocol)
        }
    }

    /// Health check — returns true if Revan is alive and model is loaded.
    pub fn health_check(&self) -> Result<bool, ClientError> {
        let data = encode_health_check();
        let raw = self.send_raw_bytes(&data)?;
        let resp = decode_response(&raw).map_err(ClientError::Protocol)?;
        Ok(resp.is_ok())
    }

    /// Send shutdown command to Revan.
    pub fn shutdown(&self) -> Result<(), ClientError> {
        let data = encode_shutdown();
        let raw = self.send_raw_bytes(&data)?;
        let _resp = decode_response(&raw).map_err(ClientError::Protocol)?;
        Ok(())
    }

    /// Low-level: send raw bytes and receive raw response bytes.
    #[cfg(windows)]
    fn send_raw_bytes(&self, request_bytes: &[u8]) -> Result<Vec<u8>, ClientError> {
        unsafe {
            // Connect to named pipe
            let pipe_hstring = HSTRING::from(&self.pipe_name);
            let handle = CreateFileW(
                &pipe_hstring,
                (0x80000000u32 | 0x40000000u32).into(), // GENERIC_READ | GENERIC_WRITE
                FILE_SHARE_NONE,
                None,
                OPEN_EXISTING,
                FILE_ATTRIBUTE_NORMAL,
                None,
            )
            .map_err(|e: windows::core::Error| ClientError::ConnectFailed(self.pipe_name.clone(), e.to_string()))?;

            // Set pipe to message-read mode (critical for atomic messages)
            let mode = PIPE_READMODE_MESSAGE;
            let result = SetNamedPipeHandleState(
                handle,
                Some(&mode as *const _ as *const _),
                None,
                None,
            );
            if let Err(e) = result {
                let _ = CloseHandle(handle);
                return Err(ClientError::SetModeFailed(e.to_string()));
            }

            // Write request
            let mut bytes_written = 0u32;
            if let Err(e) = WriteFile(
                handle,
                Some(request_bytes),
                Some(&mut bytes_written),
                None,
            ) {
                let _ = CloseHandle(handle);
                return Err(ClientError::WriteFailed(e.to_string()));
            }

            debug!(bytes = bytes_written, "wrote request to pipe");

            // Read response
            let mut buf = vec![0u8; PIPE_BUFFER_SIZE];
            let mut bytes_read = 0u32;
            if let Err(e) = ReadFile(
                handle,
                Some(&mut buf),
                Some(&mut bytes_read),
                None,
            ) {
                let _ = CloseHandle(handle);
                return Err(ClientError::ReadFailed(e.to_string()));
            }

            let _ = CloseHandle(handle);

            debug!(bytes = bytes_read, "read response from pipe");

            Ok(buf[..bytes_read as usize].to_vec())
        }
    }

    #[cfg(not(windows))]
    fn send_raw_bytes(&self, _request_bytes: &[u8]) -> Result<Vec<u8>, ClientError> {
        Err(ClientError::UnsupportedPlatform)
    }
}
