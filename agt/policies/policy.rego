package full_rag.governance

# Microsoft Agent Governance Toolkit - Rego policies for the Full-RAG assistant
#
# These policies evaluate tool calls before they execute (pre_tool_call
# intervention points). Return `true` to allow, `false` to deny.

# Chroma vector store queries - always allowed for read-only retrieval
allow_chroma_query := true

# LLM generation - allowed unless the input is flagged as unsafe
allow_llm_generate := true

# Mem0 read operations - always allowed
allow_mem0_read := true

# Mem0 write operations - always allowed
allow_mem0_write := true

# PDF ingest - allowed
allow_pdf_ingest := true

# Deny list: tool names that should never be executed
denied_tools := {"delete_collection", "drop_database", "execute_raw_sql"}

# Override: deny if tool is in the denied list
default allow_tool := false

allow_tool {
    input.tool_name
    not denied_tools[input.tool_name]
}

# Violation reasons
violation_reason[reason] {
    denied_tools[input.tool_name]
    reason := sprintf("Tool '%s' is denied by governance policy", [input.tool_name])
}