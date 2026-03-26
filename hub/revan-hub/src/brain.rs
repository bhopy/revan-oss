// brain.rs — Serialized access to Revan C++ engine via mpsc channel
//
// A single mpsc channel queues all Revan requests. A dedicated tokio task
// consumes from the channel, calls client.send() via spawn_blocking
// (since it's a blocking Win32 pipe call), and returns results through
// a oneshot channel. This naturally serializes all requests FIFO.

use anyhow::{Context, Result};
use revan_core::client::RevanClient;
use revan_core::protocol::{RevanRequest, RevanResponse};
use tokio::sync::{mpsc, oneshot};
use tracing::{debug, error, info};

// Pending request: the request + a channel to send the result back on
type PendingRequest = (RevanRequest, oneshot::Sender<Result<RevanResponse>>);

/// Handle to send requests to the Revan brain.
/// Clone-friendly — all clones share the same underlying channel.
#[derive(Clone)]
pub struct Brain {
    tx: mpsc::Sender<PendingRequest>,
}

impl Brain {
    /// Create a new Brain and spawn the background worker task.
    /// `pipe_name` is the Revan Named Pipe (e.g. `\\.\pipe\revan`).
    /// `queue_size` is how many requests can be buffered (16 is good).
    pub fn new(pipe_name: &str, queue_size: usize) -> Self {
        let (tx, rx) = mpsc::channel::<PendingRequest>(queue_size);
        let client = RevanClient::new(pipe_name);

        // Spawn the worker that processes requests sequentially
        tokio::spawn(brain_worker(client, rx));

        info!(pipe = pipe_name, queue_size, "brain worker started");
        Self { tx }
    }

    /// Send an inference request to Revan and await the response.
    /// Queues behind any pending requests (FIFO).
    pub async fn think(&self, req: RevanRequest) -> Result<RevanResponse> {
        let (resp_tx, resp_rx) = oneshot::channel();

        self.tx
            .send((req, resp_tx))
            .await
            .map_err(|_| anyhow::anyhow!("brain worker has shut down"))?;

        resp_rx
            .await
            .context("brain worker dropped the response channel")?
    }
}

/// Background worker: reads requests from channel, calls Revan via spawn_blocking.
async fn brain_worker(client: RevanClient, mut rx: mpsc::Receiver<PendingRequest>) {
    info!("brain worker ready, waiting for requests");

    while let Some((req, resp_tx)) = rx.recv().await {
        debug!(
            sys_len = req.system_prompt.len(),
            usr_len = req.user_prompt.len(),
            max_tokens = req.max_tokens,
            "processing request"
        );

        // Clone client for the blocking task (it's just a pipe name string)
        let client = client.clone();

        // Run the blocking pipe call on the blocking thread pool
        let result = tokio::task::spawn_blocking(move || client.send(&req)).await;

        // Unwrap the JoinHandle result, then the client result
        let response = match result {
            Ok(Ok(resp)) => {
                debug!(
                    status = resp.status_name(),
                    output = resp.output.as_str(),
                    gen_ms = resp.gen_ms,
                    "revan responded"
                );
                Ok(resp)
            }
            Ok(Err(e)) => {
                error!(error = %e, "revan pipe error");
                Err(anyhow::anyhow!("revan pipe error: {e}"))
            }
            Err(e) => {
                error!(error = %e, "spawn_blocking panicked");
                Err(anyhow::anyhow!("blocking task panicked: {e}"))
            }
        };

        // Send result back (ignore error if caller dropped their receiver)
        let _ = resp_tx.send(response);
    }

    info!("brain worker shutting down (channel closed)");
}

/// Standalone health check without going through the brain queue.
/// Useful at startup before the brain is fully initialized.
pub async fn standalone_health_check(pipe_name: &str) -> Result<bool> {
    let client = RevanClient::new(pipe_name);
    let result = tokio::task::spawn_blocking(move || client.health_check())
        .await
        .context("health check task panicked")?;
    Ok(result?)
}

/// Send a standalone shutdown command to Revan.
pub async fn standalone_shutdown(pipe_name: &str) -> Result<()> {
    let client = RevanClient::new(pipe_name);
    let result = tokio::task::spawn_blocking(move || client.shutdown())
        .await
        .context("shutdown task panicked")?;
    Ok(result?)
}
