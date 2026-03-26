// agent.rs — JSON protocol types for hub <-> agent communication
//
// Agents communicate over stdin/stdout using JSON Lines (one JSON object per line).
// This module defines the message types for both directions.

use serde::{Deserialize, Serialize};

// ==================== LOGPROB TYPES (for agent JSON protocol) ====================

/// Logprob entry serialized in agent JSON messages
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentTokenLogprob {
    pub token_id: i32,
    pub logprob: f32,
}

/// Per-token logprob entry for agent protocol
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentLogprobEntry {
    pub chosen: AgentTokenLogprob,
    pub top_candidates: Vec<AgentTokenLogprob>,
}

// ==================== HUB -> AGENT ====================

/// Messages the hub sends to agents via their stdin.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum HubToAgent {
    /// Dispatch a task to the agent
    #[serde(rename = "task")]
    Task {
        id: String,
        action: String,
        #[serde(default)]
        payload: serde_json::Value,
    },

    /// Revan's response to a "think" request from the agent
    #[serde(rename = "thought")]
    Thought {
        id: String,
        output: String,
        gen_ms: u32,
        /// V2: per-token logprob data (absent for V1 responses)
        #[serde(default, skip_serializing_if = "Option::is_none")]
        logprobs: Option<Vec<AgentLogprobEntry>>,
    },

    /// Tell agent to shut down gracefully
    #[serde(rename = "shutdown")]
    Shutdown,
}

// ==================== AGENT -> HUB ====================

/// Messages agents send to the hub via their stdout.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum AgentToHub {
    /// Task completed with result
    #[serde(rename = "result")]
    Result {
        id: String,
        status: String,
        #[serde(default)]
        data: serde_json::Value,
    },

    /// Agent asks hub to query Revan (the brain)
    #[serde(rename = "think")]
    Think {
        id: String,
        system: String,
        user: String,
        #[serde(default = "default_max_tokens")]
        max_tokens: u16,
        /// V2: GBNF grammar constraint (absent = no constraint)
        #[serde(default, skip_serializing_if = "Option::is_none")]
        grammar: Option<String>,
        /// V2: number of top-N logprobs to return (absent/0 = none)
        #[serde(default, skip_serializing_if = "Option::is_none")]
        top_n_logprobs: Option<u8>,
    },

    /// Progress update (informational, hub logs it)
    #[serde(rename = "progress")]
    Progress {
        id: String,
        message: String,
    },

    /// Agent-side error
    #[serde(rename = "error")]
    Error {
        id: String,
        message: String,
    },
}

fn default_max_tokens() -> u16 {
    15
}

// ==================== TESTS ====================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hub_to_agent_task_serialize() {
        let msg = HubToAgent::Task {
            id: "abc-123".into(),
            action: "classify".into(),
            payload: serde_json::json!({"path": "/tmp/test.txt"}),
        };
        let json = serde_json::to_string(&msg).unwrap();
        assert!(json.contains("\"type\":\"task\""));
        assert!(json.contains("\"action\":\"classify\""));
    }

    #[test]
    fn test_agent_to_hub_think_deserialize() {
        // V1 think (no grammar/logprobs) — backward compatible
        let json = r#"{"type":"think","id":"x1","system":"Classify.","user":"hello","max_tokens":5}"#;
        let msg: AgentToHub = serde_json::from_str(json).unwrap();
        match msg {
            AgentToHub::Think { id, system, user, max_tokens, grammar, top_n_logprobs } => {
                assert_eq!(id, "x1");
                assert_eq!(system, "Classify.");
                assert_eq!(user, "hello");
                assert_eq!(max_tokens, 5);
                assert!(grammar.is_none());
                assert!(top_n_logprobs.is_none());
            }
            _ => panic!("expected Think variant"),
        }
    }

    #[test]
    fn test_agent_to_hub_think_v2_deserialize() {
        // V2 think with grammar + logprobs
        let json = r#"{"type":"think","id":"x2","system":"Route.","user":"calc 2+2","max_tokens":3,"grammar":"root ::= \" yes\" | \" no\"","top_n_logprobs":5}"#;
        let msg: AgentToHub = serde_json::from_str(json).unwrap();
        match msg {
            AgentToHub::Think { grammar, top_n_logprobs, .. } => {
                assert_eq!(grammar.unwrap(), "root ::= \" yes\" | \" no\"");
                assert_eq!(top_n_logprobs.unwrap(), 5);
            }
            _ => panic!("expected Think variant"),
        }
    }

    #[test]
    fn test_agent_to_hub_result_deserialize() {
        let json = r#"{"type":"result","id":"x1","status":"ok","data":{"category":"config"}}"#;
        let msg: AgentToHub = serde_json::from_str(json).unwrap();
        match msg {
            AgentToHub::Result { id, status, data } => {
                assert_eq!(id, "x1");
                assert_eq!(status, "ok");
                assert_eq!(data["category"], "config");
            }
            _ => panic!("expected Result variant"),
        }
    }

    #[test]
    fn test_shutdown_roundtrip() {
        let msg = HubToAgent::Shutdown;
        let json = serde_json::to_string(&msg).unwrap();
        assert_eq!(json, r#"{"type":"shutdown"}"#);
    }

    #[test]
    fn test_default_max_tokens() {
        let json = r#"{"type":"think","id":"x1","system":"test","user":"hello"}"#;
        let msg: AgentToHub = serde_json::from_str(json).unwrap();
        match msg {
            AgentToHub::Think { max_tokens, grammar, top_n_logprobs, .. } => {
                assert_eq!(max_tokens, 15);
                assert!(grammar.is_none());
                assert!(top_n_logprobs.is_none());
            }
            _ => panic!("expected Think"),
        }
    }

    #[test]
    fn test_thought_with_logprobs_roundtrip() {
        let msg = HubToAgent::Thought {
            id: "t1".into(),
            output: " yes".into(),
            gen_ms: 42,
            logprobs: Some(vec![AgentLogprobEntry {
                chosen: AgentTokenLogprob { token_id: 100, logprob: -0.1 },
                top_candidates: vec![
                    AgentTokenLogprob { token_id: 100, logprob: -0.1 },
                    AgentTokenLogprob { token_id: 200, logprob: -2.3 },
                ],
            }]),
        };
        let json = serde_json::to_string(&msg).unwrap();
        assert!(json.contains("\"logprobs\""));

        // Deserialize back
        let parsed: HubToAgent = serde_json::from_str(&json).unwrap();
        match parsed {
            HubToAgent::Thought { logprobs, .. } => {
                let lps = logprobs.unwrap();
                assert_eq!(lps.len(), 1);
                assert_eq!(lps[0].chosen.token_id, 100);
            }
            _ => panic!("expected Thought"),
        }
    }

    #[test]
    fn test_thought_without_logprobs_backward_compat() {
        // V1 thought message (no logprobs field) should deserialize fine
        let json = r#"{"type":"thought","id":"t1","output":" yes","gen_ms":42}"#;
        let msg: HubToAgent = serde_json::from_str(json).unwrap();
        match msg {
            HubToAgent::Thought { logprobs, .. } => {
                assert!(logprobs.is_none());
            }
            _ => panic!("expected Thought"),
        }
    }
}
