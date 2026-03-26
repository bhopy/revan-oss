// agents.rs — Agent lifecycle management with Windows Job Objects
//
// Spawns child processes with piped stdin/stdout for JSON Lines IPC.
// All agents are assigned to a Windows Job Object so they die when the hub dies.
// Routes agent "think" requests to the Brain, sends "thought" responses back.

use anyhow::{Context, Result};
use revan_core::agent::{AgentToHub, HubToAgent};
use revan_core::protocol::RevanRequest;
use crate::brain::Brain;
use std::collections::HashMap;
use std::process::Stdio;
use std::sync::Arc;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, Command};
use tokio::sync::{mpsc, oneshot, Mutex};
use tracing::{debug, error, info, warn};
use uuid::Uuid;

/// Result of a completed agent task.
#[derive(Debug, Clone)]
pub struct AgentResult {
    pub status: String,
    pub data: serde_json::Value,
}

/// Shared map of pending task results: task_id -> oneshot sender.
type PendingResults = Arc<Mutex<HashMap<String, oneshot::Sender<AgentResult>>>>;

#[cfg(windows)]
use windows::Win32::Foundation::CloseHandle;
#[cfg(windows)]
use windows::Win32::System::JobObjects::{
    AssignProcessToJobObject, CreateJobObjectW, SetInformationJobObject,
    JobObjectExtendedLimitInformation, JOBOBJECT_EXTENDED_LIMIT_INFORMATION,
    JOB_OBJECT_LIMIT,
};
#[cfg(windows)]
use windows::Win32::System::Threading::{
    OpenProcess, PROCESS_SET_QUOTA, PROCESS_TERMINATE,
};

// Job object flag: kill all processes when the job handle closes
const JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE: u32 = 0x00002000;

// CREATE_NO_WINDOW prevents console flash when spawning child processes
#[cfg(windows)]
const CREATE_NO_WINDOW: u32 = 0x08000000;

/// Handle to a running agent process.
pub struct AgentHandle {
    pub name: String,
    /// Channel to send messages to the agent's stdin
    pub stdin_tx: mpsc::Sender<HubToAgent>,
    /// The child process (kept alive by ownership)
    _child: Child,
}

/// Manages agent lifecycles and routes messages.
pub struct AgentManager {
    brain: Brain,
    agents: HashMap<String, AgentHandle>,
    pending_results: PendingResults,
    #[cfg(windows)]
    job_handle: Option<windows::Win32::Foundation::HANDLE>,
}

impl AgentManager {
    /// Create a new agent manager with a Job Object for process cleanup.
    pub fn new(brain: Brain) -> Self {
        #[cfg(windows)]
        let job_handle = create_job_object();

        Self {
            brain,
            agents: HashMap::new(),
            pending_results: Arc::new(Mutex::new(HashMap::new())),
            #[cfg(windows)]
            job_handle,
        }
    }

    /// Spawn a new agent process.
    /// `name` is a human-readable identifier.
    /// `command` is the executable (e.g. "python").
    /// `args` are command-line arguments.
    /// `env` are additional environment variables.
    pub async fn spawn(
        &mut self,
        name: &str,
        command: &str,
        args: &[&str],
        env: &[(&str, &str)],
    ) -> Result<()> {
        info!(name, command, ?args, "spawning agent");

        let mut cmd = Command::new(command);
        cmd.args(args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit()); // agent stderr goes to hub's stderr

        // Set environment variables
        for (k, v) in env {
            cmd.env(k, v);
        }

        // Windows: CREATE_NO_WINDOW to prevent console flash
        #[cfg(windows)]
        cmd.creation_flags(CREATE_NO_WINDOW);

        let mut child = cmd.spawn().context("failed to spawn agent process")?;

        // Assign to Job Object (so it dies when hub dies)
        #[cfg(windows)]
        {
            if let Some(job) = self.job_handle {
                assign_to_job(job, &child);
            }
        }

        // Take ownership of stdin/stdout
        let stdin = child.stdin.take().context("no stdin on child")?;
        let stdout = child.stdout.take().context("no stdout on child")?;

        // Create channel for sending messages to this agent
        let (stdin_tx, mut stdin_rx) = mpsc::channel::<HubToAgent>(32);

        // Spawn stdin writer task
        let agent_name = name.to_string();
        tokio::spawn(async move {
            let mut stdin = stdin;
            while let Some(msg) = stdin_rx.recv().await {
                let mut line = serde_json::to_string(&msg).unwrap_or_default();
                line.push('\n');
                if let Err(e) = stdin.write_all(line.as_bytes()).await {
                    error!(agent = agent_name.as_str(), error = %e, "stdin write failed");
                    break;
                }
                let _ = stdin.flush().await;
            }
            debug!(agent = agent_name.as_str(), "stdin writer done");
        });

        // Spawn stdout reader task (routes messages to hub)
        let brain = self.brain.clone();
        let reader_name = name.to_string();
        let reply_tx = stdin_tx.clone();
        let pending = self.pending_results.clone();
        tokio::spawn(async move {
            let reader = BufReader::new(stdout);
            let mut lines = reader.lines();

            while let Ok(Some(line)) = lines.next_line().await {
                if line.trim().is_empty() {
                    continue;
                }

                // Parse JSON line from agent
                let msg: AgentToHub = match serde_json::from_str(&line) {
                    Ok(m) => m,
                    Err(e) => {
                        warn!(
                            agent = reader_name.as_str(),
                            line = line.as_str(),
                            error = %e,
                            "invalid JSON from agent"
                        );
                        continue;
                    }
                };

                match msg {
                    AgentToHub::Think { id, system, user, max_tokens, grammar, top_n_logprobs } => {
                        // Route to Revan brain
                        debug!(agent = reader_name.as_str(), id = id.as_str(), "routing think request to brain");
                        let req = RevanRequest {
                            system_prompt: system,
                            user_prompt: user,
                            max_tokens,
                            temperature: 0.1,
                            grammar,
                            top_n_logprobs: top_n_logprobs.unwrap_or(0),
                        };

                        match brain.think(req).await {
                            Ok(resp) => {
                                // Convert protocol logprobs to agent logprobs
                                let logprobs = if resp.logprobs.is_empty() {
                                    None
                                } else {
                                    Some(resp.logprobs.iter().map(|e| {
                                        revan_core::agent::AgentLogprobEntry {
                                            chosen: revan_core::agent::AgentTokenLogprob {
                                                token_id: e.chosen.token_id,
                                                logprob: e.chosen.logprob,
                                            },
                                            top_candidates: e.top_candidates.iter().map(|c| {
                                                revan_core::agent::AgentTokenLogprob {
                                                    token_id: c.token_id,
                                                    logprob: c.logprob,
                                                }
                                            }).collect(),
                                        }
                                    }).collect())
                                };
                                let thought = HubToAgent::Thought {
                                    id,
                                    output: resp.output,
                                    gen_ms: resp.gen_ms,
                                    logprobs,
                                };
                                let _ = reply_tx.send(thought).await;
                            }
                            Err(e) => {
                                error!(agent = reader_name.as_str(), error = %e, "brain think failed");
                                let thought = HubToAgent::Thought {
                                    id,
                                    output: String::new(),
                                    gen_ms: 0,
                                    logprobs: None,
                                };
                                let _ = reply_tx.send(thought).await;
                            }
                        }
                    }

                    AgentToHub::Result { id, status, data } => {
                        info!(
                            agent = reader_name.as_str(),
                            id = id.as_str(),
                            status = status.as_str(),
                            "agent returned result"
                        );
                        // Route result to whoever dispatched the task
                        let mut map = pending.lock().await;
                        if let Some(tx) = map.remove(&id) {
                            let _ = tx.send(AgentResult { status, data });
                        } else {
                            debug!(id = id.as_str(), "no pending receiver for result (fire-and-forget dispatch)");
                        }
                    }

                    AgentToHub::Progress { id, message } => {
                        debug!(
                            agent = reader_name.as_str(),
                            id = id.as_str(),
                            message = message.as_str(),
                            "agent progress"
                        );
                    }

                    AgentToHub::Error { id, message } => {
                        error!(
                            agent = reader_name.as_str(),
                            id = id.as_str(),
                            message = message.as_str(),
                            "agent error"
                        );
                    }
                }
            }

            info!(agent = reader_name.as_str(), "stdout reader done (agent exited or pipe closed)");
        });

        let handle = AgentHandle {
            name: name.to_string(),
            stdin_tx,
            _child: child,
        };
        self.agents.insert(name.to_string(), handle);

        info!(name, "agent spawned successfully");
        Ok(())
    }

    /// Dispatch a task to a named agent and return a receiver for the result.
    /// The receiver resolves when the agent sends a Result message with the matching task ID.
    pub async fn dispatch(
        &self,
        agent_name: &str,
        action: &str,
        payload: serde_json::Value,
    ) -> Result<(String, oneshot::Receiver<AgentResult>)> {
        let handle = self.agents.get(agent_name)
            .context(format!("no agent named '{agent_name}'"))?;

        let id = Uuid::new_v4().to_string();
        let (result_tx, result_rx) = oneshot::channel();

        // Register pending result before sending task (no race)
        self.pending_results.lock().await.insert(id.clone(), result_tx);

        let msg = HubToAgent::Task {
            id: id.clone(),
            action: action.to_string(),
            payload,
        };

        if let Err(e) = handle.stdin_tx.send(msg).await {
            // Clean up pending entry on send failure
            self.pending_results.lock().await.remove(&id);
            return Err(anyhow::anyhow!("agent '{agent_name}' stdin channel closed: {e}"));
        }

        info!(agent = agent_name, task_id = id.as_str(), action, "task dispatched");
        Ok((id, result_rx))
    }

    /// Dispatch a task without waiting for the result (fire-and-forget).
    pub async fn dispatch_fire_and_forget(
        &self,
        agent_name: &str,
        action: &str,
        payload: serde_json::Value,
    ) -> Result<String> {
        let handle = self.agents.get(agent_name)
            .context(format!("no agent named '{agent_name}'"))?;

        let id = Uuid::new_v4().to_string();

        let msg = HubToAgent::Task {
            id: id.clone(),
            action: action.to_string(),
            payload,
        };

        handle.stdin_tx.send(msg).await
            .map_err(|_| anyhow::anyhow!("agent '{agent_name}' stdin channel closed"))?;

        info!(agent = agent_name, task_id = id.as_str(), action, "task dispatched (fire-and-forget)");
        Ok(id)
    }

    /// Shutdown and remove a single agent by name.
    pub async fn shutdown_agent(&mut self, name: &str) -> Result<()> {
        let handle = self.agents.remove(name)
            .context(format!("no agent named '{name}'"))?;

        let _ = handle.stdin_tx.send(HubToAgent::Shutdown).await;
        tokio::time::sleep(std::time::Duration::from_secs(1)).await;
        info!(agent = name, "agent stopped");
        Ok(())
    }

    /// Send shutdown to all agents and wait briefly for them to exit.
    pub async fn shutdown_all(&mut self) {
        info!(count = self.agents.len(), "shutting down all agents");

        for (name, handle) in &self.agents {
            debug!(agent = name.as_str(), "sending shutdown");
            let _ = handle.stdin_tx.send(HubToAgent::Shutdown).await;
        }

        // Give agents a moment to exit gracefully
        tokio::time::sleep(std::time::Duration::from_secs(2)).await;

        // Job Object will kill any stragglers when hub exits
        self.agents.clear();
        info!("all agents shut down");
    }

    /// List active agent names.
    pub fn list(&self) -> Vec<&str> {
        self.agents.keys().map(|s| s.as_str()).collect()
    }
}

impl Drop for AgentManager {
    fn drop(&mut self) {
        #[cfg(windows)]
        {
            if let Some(handle) = self.job_handle {
                // Closing the job handle kills all assigned processes
                unsafe { let _ = CloseHandle(handle); }
            }
        }
    }
}

// ==================== WINDOWS JOB OBJECT ====================

/// Create a Windows Job Object with KILL_ON_JOB_CLOSE.
/// When the hub process exits (or the handle is closed), all child processes die.
#[cfg(windows)]
fn create_job_object() -> Option<windows::Win32::Foundation::HANDLE> {
    unsafe {
        let job = match CreateJobObjectW(None, None) {
            Ok(j) => j,
            Err(e) => {
                error!(error = %e, "failed to create Job Object");
                return None;
            }
        };

        // Set KILL_ON_JOB_CLOSE flag
        let mut info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION::default();
        info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT(JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE);

        let result = SetInformationJobObject(
            job,
            JobObjectExtendedLimitInformation,
            &info as *const _ as *const std::ffi::c_void,
            std::mem::size_of::<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>() as u32,
        );

        if result.is_err() {
            error!(error = ?result.err(), "failed to set Job Object limits");
            let _ = CloseHandle(job);
            return None;
        }

        info!("job object created (kill-on-close enabled)");
        Some(job)
    }
}

/// Assign a child process to the Job Object using its PID.
#[cfg(windows)]
fn assign_to_job(job: windows::Win32::Foundation::HANDLE, child: &Child) {
    let pid = match child.id() {
        Some(pid) => pid,
        None => {
            warn!("child has no PID, cannot assign to job object");
            return;
        }
    };

    unsafe {
        // Open process handle with required access rights for job assignment
        let proc_handle = match OpenProcess(PROCESS_SET_QUOTA | PROCESS_TERMINATE, false, pid) {
            Ok(h) => h,
            Err(e) => {
                warn!(pid, error = %e, "failed to open process for job assignment");
                return;
            }
        };

        if let Err(e) = AssignProcessToJobObject(job, proc_handle) {
            warn!(pid, error = %e, "failed to assign process to job object");
        } else {
            debug!(pid, "process assigned to job object");
        }

        let _ = CloseHandle(proc_handle);
    }
}
