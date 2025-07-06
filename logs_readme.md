# SWE Arena Logging Systems

## Core IDs Design

In FastChat Arena's battle mode, there are three key identifiers:

- **`chat_session_id`**: Session-level unique identifier, shared by both models in the same battle
- **`conv_id`**: Conversation-level unique identifier, unique to each model instance
- **`sandbox_id`**: Remote e2b sandbox environment identifier, unique to each sandbox instance where code executes

| Aspect | Conversation Logs | Sandbox Logs |
|--------|-------------------|--------------|
| **Filename Based On** | `chat_session_id` | `conv_id` |
| **File Granularity** | Per Session (cross-model) | Per Model Per Round |
| **Write Mode** | Append mode | Overwrite mode |
| **Content Scope** | Complete battle records | Single sandbox operation |

## Architecture Diagram

```mermaid
graph TD
    A[Arena Battle Session] --> B[chat_session_id: abc123]
    
    B --> C[Model A State]
    B --> D[Model B State]
    
    C --> C1[conv_id: def456]
    D --> D1[conv_id: ghi789]
    
    subgraph "Conversation Logs"
        E[conv-log-abc123.json]
        E --> E1[Model A Records]
        E --> E2[Model B Records]
        E --> E3[Voting Records]
    end
    
    subgraph "Sandbox Logs"
        F[sandbox-logs-def456-1-1.json]
        G[sandbox-logs-ghi789-1-1.json]
        H[sandbox-logs-def456-2-1.json]
    end
    
    C1 --> F
    C1 --> H
    D1 --> G
    
    style E fill:#ffeb3b
    style F fill:#4caf50
    style G fill:#4caf50
    style H fill:#4caf50
```

## File Naming Rules

### Conversation Logs
```
logs/{date}/conv_logs/{chat_mode}/conv-log-{chat_session_id}.json
```
**Example**: `logs/2025_01_15/conv_logs/battle_anony/conv-log-abc123.json`

The same `conv_logs` file will contain the outputs of both models in a battle session.

### Sandbox Logs

- **Purpose**: Record precise code execution states
- **Characteristics**: Each model's each sandbox operation has an independent file
- **Use Cases**: Code debugging, execution state tracking, version control

```
logs/{date}/sandbox_logs/sandbox-logs-{conv_id}-{chat_round}-{sandbox_run_round}.json
```
**Example**: `logs/2025_01_15/sandbox_logs/sandbox-logs-def456-1-1.json`
- Meaning: on 2025_01_15, the conv id `def456`, the first response, the first time to run.
- If users send another followup prompt, it will be `logs/2025_01_15/sandbox_logs/sandbox-logs-def456-2-1.json`.
- If users run the sandbox again to re-initialize, it will be `logs/2025_01_15/sandbox_logs/sandbox-logs-def456-2-2.json`.

## Relationship Mapping

Both log types are fully correlated through ID fields:

```python
# Find all records for a session
conv_file = f"conv-log-{chat_session_id}.json"

# Find sandbox records for a specific model
sandbox_files = glob(f"sandbox-logs-{conv_id}-*-*.json")
```
