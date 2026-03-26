// main.rs — Revan Hub entry point
//
// Loads hub.toml config, initializes tracing, creates Brain + AgentManager +
// VramMonitor, then runs an interactive stdin command loop.

mod brain;
mod agents;
mod monitor;

use anyhow::{Context, Result};
use serde::Deserialize;
use tokio::io::{AsyncBufReadExt, BufReader};
use tracing::{error, info, warn};

// ==================== CONFIG ====================

#[derive(Debug, Deserialize)]
struct HubConfig {
    revan: RevanSection,
    #[serde(default)]
    monitor: MonitorSection,
    #[serde(default)]
    agents: std::collections::HashMap<String, AgentSection>,
}

#[derive(Debug, Deserialize)]
struct RevanSection {
    pipe: String,
    #[serde(default = "default_health_interval")]
    health_check_interval_secs: u64,
    #[serde(default = "default_request_timeout")]
    request_timeout_secs: u64,
}

#[derive(Debug, Deserialize, Default)]
struct MonitorSection {
    #[serde(default = "default_vram_poll")]
    vram_poll_interval_secs: u64,
    #[serde(default = "default_vram_threshold")]
    vram_warn_threshold_percent: u32,
}

#[derive(Debug, Deserialize, Clone)]
struct AgentSection {
    command: String,
    #[serde(default)]
    args: Vec<String>,
    #[serde(default)]
    description: String,
    #[serde(default)]
    auto_start: bool,
    #[serde(default)]
    env: std::collections::HashMap<String, String>,
}

fn default_health_interval() -> u64 { 30 }
fn default_request_timeout() -> u64 { 10 }
fn default_vram_poll() -> u64 { 5 }
fn default_vram_threshold() -> u32 { 95 }

// ==================== MAIN ====================

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize tracing (structured logging)
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "revan_hub=info,revan_core=debug".into()),
        )
        .with_target(false)
        .init();

    info!("Revan Hub starting...");

    // Load config
    let config_path = find_config()?;
    let config_text = std::fs::read_to_string(&config_path)
        .context(format!("failed to read config: {config_path}"))?;
    let config: HubConfig = toml::from_str(&config_text)
        .context("failed to parse hub.toml")?;

    info!(
        pipe = config.revan.pipe.as_str(),
        agents = config.agents.len(),
        "config loaded from {config_path}"
    );

    // Health check — non-fatal, hub works without Revan running
    info!("checking Revan health...");
    match brain::standalone_health_check(&config.revan.pipe).await {
        Ok(true) => info!("Revan is alive and model is loaded"),
        Ok(false) => warn!("Revan responded but model is NOT loaded"),
        Err(e) => {
            warn!(error = %e, "cannot reach Revan — hub will start without brain");
            info!("hint: start Revan with: build/Release/revan.exe (from the project root)");
        }
    }

    // Initialize VRAM monitor
    let vram_monitor = monitor::VramMonitor::new(config.monitor.vram_warn_threshold_percent);
    if let Some(status) = vram_monitor.status() {
        info!(
            used = status.used_mib,
            total = status.total_mib,
            free = status.free_mib,
            usage = format!("{:.1}%", status.usage_percent * 100.0).as_str(),
            "initial VRAM status"
        );
    }
    vram_monitor.spawn_watchdog(config.monitor.vram_poll_interval_secs);

    // Initialize Brain (Revan access queue)
    let brain = brain::Brain::new(&config.revan.pipe, 16);

    // Initialize Agent Manager
    let mut agent_mgr = agents::AgentManager::new(brain.clone());

    // Auto-start agents
    for (name, agent_cfg) in &config.agents {
        if agent_cfg.auto_start {
            let args: Vec<&str> = agent_cfg.args.iter().map(|s| s.as_str()).collect();
            let env: Vec<(&str, &str)> = agent_cfg.env.iter()
                .map(|(k, v)| (k.as_str(), v.as_str()))
                .collect();

            if let Err(e) = agent_mgr.spawn(name, &agent_cfg.command, &args, &env).await {
                error!(agent = name.as_str(), error = %e, "failed to auto-start agent");
            }
        }
    }

    // Interactive command loop
    println!("Revan Hub ready. Type 'help' for commands.");
    let stdin = BufReader::new(tokio::io::stdin());
    let mut lines = stdin.lines();

    loop {
        // Print prompt
        eprint!("> ");

        tokio::select! {
            line = lines.next_line() => {
                match line {
                    Ok(Some(line)) => {
                        let line = line.trim().to_string();
                        if line.is_empty() { continue; }

                        let parts: Vec<&str> = line.splitn(4, ' ').collect();
                        match parts[0] {
                            "help" => print_help(),
                            "status" => cmd_status(&agent_mgr, &vram_monitor),
                            "health" => cmd_health(&config.revan.pipe).await,
                            "spawn" => {
                                if parts.len() < 2 {
                                    println!("usage: spawn <agent_name>");
                                } else {
                                    cmd_spawn(&mut agent_mgr, &config.agents, parts[1]).await;
                                }
                            }
                            "stop" => {
                                if parts.len() < 2 {
                                    println!("usage: stop <agent_name>");
                                } else {
                                    cmd_stop(&mut agent_mgr, parts[1]).await;
                                }
                            }
                            "dispatch" => {
                                if parts.len() < 4 {
                                    println!("usage: dispatch <agent> <action> <json_payload>");
                                } else {
                                    cmd_dispatch(&agent_mgr, parts[1], parts[2], parts[3]).await;
                                }
                            }
                            "quit" | "exit" => {
                                println!("shutting down...");
                                break;
                            }
                            other => println!("unknown command: {other} (type 'help')"),
                        }
                    }
                    Ok(None) => break, // EOF
                    Err(e) => {
                        error!(error = %e, "stdin read error");
                        break;
                    }
                }
            }
            _ = tokio::signal::ctrl_c() => {
                println!("\nCtrl+C received");
                break;
            }
        }
    }

    // Graceful shutdown
    agent_mgr.shutdown_all().await;

    info!("Revan Hub stopped.");
    Ok(())
}

// ==================== COMMANDS ====================

fn print_help() {
    println!("Commands:");
    println!("  help                              — show this help");
    println!("  status                            — running agents + VRAM");
    println!("  health                            — check Revan brain");
    println!("  spawn <name>                      — start agent from hub.toml");
    println!("  stop <name>                       — stop a running agent");
    println!("  dispatch <agent> <action> <json>  — send task, await result");
    println!("  quit                              — shutdown and exit");
}

fn cmd_status(agent_mgr: &agents::AgentManager, vram: &monitor::VramMonitor) {
    let agents = agent_mgr.list();
    if agents.is_empty() {
        println!("agents: (none running)");
    } else {
        println!("agents: {}", agents.join(", "));
    }
    match vram.status() {
        Some(s) => println!("vram:   {}/{} MiB ({:.1}%)", s.used_mib, s.total_mib, s.usage_percent * 100.0),
        None => println!("vram:   (monitoring unavailable)"),
    }
}

async fn cmd_health(pipe_name: &str) {
    match brain::standalone_health_check(pipe_name).await {
        Ok(true) => println!("revan: OK (model loaded)"),
        Ok(false) => println!("revan: connected but model NOT loaded"),
        Err(e) => println!("revan: DOWN ({e})"),
    }
}

async fn cmd_spawn(
    agent_mgr: &mut agents::AgentManager,
    agent_configs: &std::collections::HashMap<String, AgentSection>,
    name: &str,
) {
    let cfg = match agent_configs.get(name) {
        Some(c) => c,
        None => {
            let available: Vec<&str> = agent_configs.keys().map(|s| s.as_str()).collect();
            println!("no agent '{name}' in config. available: {}", available.join(", "));
            return;
        }
    };

    let args: Vec<&str> = cfg.args.iter().map(|s| s.as_str()).collect();
    let env: Vec<(&str, &str)> = cfg.env.iter().map(|(k, v)| (k.as_str(), v.as_str())).collect();

    match agent_mgr.spawn(name, &cfg.command, &args, &env).await {
        Ok(()) => println!("spawned: {name}"),
        Err(e) => println!("failed to spawn {name}: {e}"),
    }
}

async fn cmd_stop(agent_mgr: &mut agents::AgentManager, name: &str) {
    match agent_mgr.shutdown_agent(name).await {
        Ok(()) => println!("stopped: {name}"),
        Err(e) => println!("failed to stop {name}: {e}"),
    }
}

async fn cmd_dispatch(agent_mgr: &agents::AgentManager, agent: &str, action: &str, json_str: &str) {
    let payload: serde_json::Value = match serde_json::from_str(json_str) {
        Ok(v) => v,
        Err(e) => {
            println!("invalid JSON: {e}");
            return;
        }
    };

    match agent_mgr.dispatch(agent, action, payload).await {
        Ok((task_id, result_rx)) => {
            println!("dispatched: {task_id}");
            println!("waiting for result...");
            match tokio::time::timeout(
                std::time::Duration::from_secs(30),
                result_rx,
            ).await {
                Ok(Ok(result)) => {
                    println!("result: status={}, data={}", result.status, result.data);
                }
                Ok(Err(_)) => println!("error: result channel dropped (agent crashed?)"),
                Err(_) => println!("error: timeout (30s)"),
            }
        }
        Err(e) => println!("dispatch failed: {e}"),
    }
}

// ==================== CONFIG LOOKUP ====================

/// Find hub.toml — check current dir, then relative to exe.
fn find_config() -> Result<String> {
    let candidates = [
        "hub.toml".to_string(),
    ];

    for path in &candidates {
        if std::path::Path::new(path).exists() {
            return Ok(path.clone());
        }
    }

    // Try relative to executable
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            let candidate = dir.join("hub.toml");
            if candidate.exists() {
                return Ok(candidate.to_string_lossy().to_string());
            }
            if let Some(parent) = dir.parent() {
                let candidate = parent.join("hub.toml");
                if candidate.exists() {
                    return Ok(candidate.to_string_lossy().to_string());
                }
            }
        }
    }

    anyhow::bail!("hub.toml not found. Place it in the project root or the current directory.")
}
