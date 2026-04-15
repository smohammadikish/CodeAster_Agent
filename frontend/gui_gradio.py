from __future__ import annotations

import json
import sqlite3
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import gradio as gr
except ImportError:  # pragma: no cover - optional runtime dependency
    gr = None

from langchain_core.messages import AIMessage, HumanMessage

from chat.state import ParameterSheet
from main_agent import (
    build_agent,
    build_catalog_family_guidance,
    build_confirmation_summary,
    build_pipeline_kwargs_from_sheet,
    build_post_run_idle_message,
    build_post_run_message,
    build_post_success_message,
    build_state_aware_input,
    build_state_extractor_llm,
    extract_run_id_and_folder,
    get_ambiguous_dimension_message,
    get_missing_fields_for_execution,
    get_sheet_scenario,
    get_sheet_signature,
    is_confirmation_input,
    is_exit_input,
    is_negative_response,
    is_successful_run_output,
    is_verification_input,
    run_full_pipeline,
    run_verification_for_run,
)
from storage import DB_PATH, RUNS_ROOT, get_registered_run_paths, init_db


APP_CSS = """
:root {
  --bg: #edf1f4;
  --panel: #fbfcfd;
  --panel-2: #f4f7f9;
  --border: #c6d0d8;
  --ink: #16222d;
  --muted: #566674;
  --blue: #1e5b87;
  --green: #2d6a4f;
  --amber: #9a6700;
  --red: #a63d40;
  --slate: #41505c;
}

body, .gradio-container {
  background:
    radial-gradient(circle at top left, rgba(30,91,135,0.08), transparent 28%),
    linear-gradient(180deg, #f4f7f9 0%, #e9eef2 100%);
  color: var(--ink);
}

.gradio-container {
  max-width: 100% !important;
  padding: 10px 14px !important;
  font-size: 13px !important;
  line-height: 1.25 !important;
}

h1, h2, h3, h4, h5, h6,
label, .prose, .md, .tiny, .section-label, .run-title {
  font-family: "Segoe UI", "Aptos", "Trebuchet MS", sans-serif !important;
}

button, input, textarea, select {
  font-size: 13px !important;
}

.app-shell {
  border: 1px solid var(--border);
  background: linear-gradient(180deg, rgba(255,255,255,0.92), rgba(248,250,252,0.95));
  box-shadow: 0 14px 40px rgba(18, 32, 45, 0.08);
}

.panel-card {
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 14px 16px;
}

.status-strip {
  display: flex;
  gap: 10px;
  flex-wrap: wrap;
  align-items: center;
}

.status-pill {
  display: inline-block;
  padding: 6px 10px;
  border-radius: 999px;
  border: 1px solid var(--border);
  background: var(--panel-2);
  color: var(--slate);
  font-size: 12px;
  letter-spacing: 0.04em;
  text-transform: uppercase;
  font-weight: 700;
}

.status-pill.active {
  color: white;
  background: var(--blue);
  border-color: var(--blue);
}

.status-pill.completed {
  color: white;
  background: var(--green);
  border-color: var(--green);
}

.status-pill.pending {
  color: white;
  background: var(--amber);
  border-color: var(--amber);
}

.status-pill.failed {
  color: white;
  background: var(--red);
  border-color: var(--red);
}

.spec-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 13px;
}

.spec-table th,
.spec-table td {
  border-bottom: 1px solid var(--border);
  padding: 8px 6px;
  text-align: left;
  vertical-align: top;
}

.spec-table th {
  width: 34%;
  color: var(--muted);
  font-weight: 600;
}

.section-label {
  font-size: 11px;
  font-weight: 700;
  color: var(--muted);
  text-transform: uppercase;
  letter-spacing: 0.08em;
  margin-bottom: 8px;
}

.tiny {
  font-size: 12px;
  color: var(--muted);
}

.run-title {
  font-size: 16px;
  font-weight: 700;
  margin-bottom: 4px;
}

.gr-chatbot {
  min-height: 420px !important;
}

.gr-group, .gr-box, .gr-form, .gr-block {
  border-radius: 10px !important;
}

@media (max-width: 1100px) {
  .gradio-container {
    padding: 8px !important;
  }
}

@media (max-width: 900px) {
  .gradio-container {
    font-size: 12px !important;
  }
}
"""


@dataclass
class FrontendSession:
    sheet: ParameterSheet = field(default_factory=ParameterSheet)
    chat_messages: list[dict[str, str]] = field(default_factory=list)
    lc_history: list[Any] = field(default_factory=list)
    simulation_executed: bool = False
    last_executed_signature: str | None = None
    last_run_id: str | None = None
    verification_prompt_pending: bool = False
    agent_executor: Any | None = None
    extractor_llm: Any | None = None

    def ensure_models(self) -> None:
        if self.agent_executor is None:
            self.agent_executor = build_agent()
        if self.extractor_llm is None:
            self.extractor_llm = build_state_extractor_llm()

    def append_exchange(self, user_text: str, assistant_text: str) -> None:
        self.chat_messages.append({"role": "user", "content": user_text})
        self.chat_messages.append({"role": "assistant", "content": assistant_text})
        self.lc_history.append(HumanMessage(content=user_text))
        self.lc_history.append(AIMessage(content=assistant_text))


def _safe_json_load(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _read_text(path: Path, max_chars: int = 20000) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:
        return f"Could not read {path}.\n{exc}"
    if len(text) > max_chars:
        return text[:max_chars] + "\n\n... truncated ..."
    return text


def _format_json(data: Any) -> str:
    return json.dumps(data, indent=2, ensure_ascii=False) if data is not None else "{}"


def _detect_phase(session: FrontendSession) -> str:
    missing = get_missing_fields_for_execution(session.sheet)
    if session.simulation_executed and session.verification_prompt_pending:
        return "verification"
    if session.simulation_executed:
        return "interpretation"
    if not session.sheet.to_dict():
        return "elicitation"
    if not missing:
        return "confirmation"
    return "elicitation"


def _format_value(value: Any) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _phase_badges(active_phase: str) -> str:
    phases = [
        ("elicitation", "Elicitation"),
        ("confirmation", "Confirmation"),
        ("execution", "Execution"),
        ("interpretation", "Interpretation"),
        ("verification", "Verification"),
    ]
    active_index = next((i for i, (name, _) in enumerate(phases) if name == active_phase), 0)
    pills: list[str] = []
    for idx, (key, label) in enumerate(phases):
        klass = "active" if key == active_phase else "completed" if idx < active_index else "status-pill"
        if klass == "status-pill":
            pills.append(f'<span class="status-pill">{label}</span>')
        else:
            pills.append(f'<span class="status-pill {klass}">{label}</span>')
    return "".join(pills)


def _render_workflow_html(session: FrontendSession) -> str:
    phase = _detect_phase(session)
    run_id = session.last_run_id or "No active run"
    missing = get_missing_fields_for_execution(session.sheet)
    readiness = "Ready for explicit confirmation" if not missing else "Collecting required inputs"
    if session.simulation_executed:
        readiness = "Run completed. Inspect artifacts or request verification."
    return f"""
    <div class="panel-card">
      <div class="section-label">Workflow Status</div>
      <div class="status-strip">{_phase_badges(phase)}</div>
      <div style="margin-top:12px">
        <div class="run-title">{phase.title()}</div>
        <div><strong>Current run:</strong> {run_id}</div>
        <div class="tiny" style="margin-top:6px">{readiness}</div>
      </div>
    </div>
    """


def _render_parameter_sheet_html(session: FrontendSession) -> str:
    values = session.sheet.to_dict()
    resolved_state = session.sheet.resolved_state()
    resolved = resolved_state["resolved"]
    missing = resolved_state["missing"]
    notes = resolved_state["notes"]
    rows = []
    for key in (
        "scenario",
        "length_m",
        "length_unit",
        "n_elems",
        "FY",
        "force_unit",
        "HY",
        "hy_unit",
        "HZ",
        "hz_unit",
        "material_hint",
        "section_hint",
        "section_shape",
        "E",
        "NU",
        "RHO",
        "A_mm2",
        "IY_mm4",
        "IZ_mm4",
        "JX_mm4",
        "AY_mm2",
        "AZ_mm2",
    ):
        if key in values:
            rows.append(f"<tr><th>{key}</th><td>{_format_value(values[key])}</td></tr>")
    if not rows:
        rows.append("<tr><th>Sheet</th><td>No parameters collected yet.</td></tr>")

    material = resolved.get("resolved_material")
    section = resolved.get("resolved_section")
    material_html = "N/A"
    section_html = "N/A"
    if material:
        material_html = f"{material.get('name', 'unknown')}<br><span class='tiny'>{material.get('reference', '')}</span>"
    if section:
        name = section.get("Profile") or section.get("Section") or section.get("shape") or "section"
        section_html = f"{name}<br><span class='tiny'>{section.get('reference', 'generated section')}</span>"

    notes_html = "<br>".join(notes) if notes else "N/A"
    missing_html = ", ".join(missing) if missing else "none"

    return f"""
    <div class="panel-card">
      <div class="section-label">Parameter Sheet</div>
      <table class="spec-table">
        {''.join(rows)}
        <tr><th>Resolved material</th><td>{material_html}</td></tr>
        <tr><th>Resolved section</th><td>{section_html}</td></tr>
        <tr><th>Missing required</th><td>{missing_html}</td></tr>
        <tr><th>Resolution notes</th><td>{notes_html}</td></tr>
      </table>
    </div>
    """


def _query_runs() -> list[dict[str, Any]]:
    init_db()
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT run_id, status, created_at, completed_at, params_json, run_dir, error_message
            FROM runs
            ORDER BY created_at DESC
            """
        ).fetchall()
    items: list[dict[str, Any]] = []
    for row in rows:
        try:
            params = json.loads(row["params_json"] or "{}")
        except Exception:
            params = {}
        run_dir = Path(row["run_dir"])
        items.append(
            {
                "run_id": row["run_id"],
                "status": row["status"],
                "created_at": row["created_at"],
                "completed_at": row["completed_at"] or "",
                "scenario": params.get("scenario", ""),
                "length_m": params.get("length_m", ""),
                "n_elems": params.get("n_elems", ""),
                "FY": params.get("FY", ""),
                "verification": "yes" if (run_dir / "verification").exists() else "no",
                "run_dir": str(run_dir),
                "error_message": row["error_message"] or "",
            }
        )
    return items


def _runs_dataframe_rows(runs: list[dict[str, Any]]) -> list[list[Any]]:
    return [
        [
            run["run_id"],
            run["status"],
            run["scenario"],
            run["length_m"],
            run["n_elems"],
            run["FY"],
            run["verification"],
            run["created_at"],
        ]
        for run in runs
    ]


def _artifact_stage_from_path(run_root: Path, artifact_path: Path) -> str:
    try:
        relative_parts = artifact_path.relative_to(run_root).parts
    except Exception:
        return "other"
    return relative_parts[0] if relative_parts else "other"


def _artifact_kind(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".png", ".jpg", ".jpeg", ".bmp", ".gif"}:
        return "image"
    if suffix in {".json"}:
        return "json"
    if suffix in {".txt", ".log", ".comm", ".geo", ".export", ".mess", ".resu", ".out", ".csv", ".md", ".tcl"}:
        return "text"
    return "binary"


def _collect_artifacts(run_id: str) -> list[dict[str, Any]]:
    run_paths = get_registered_run_paths(run_id)
    run_root = run_paths.root if run_paths else RUNS_ROOT / run_id
    if not run_root.exists():
        return []

    artifacts: dict[str, dict[str, Any]] = {}
    for path in sorted(run_root.rglob("*")):
        if not path.is_file():
            continue
        stage = _artifact_stage_from_path(run_root, path)
        artifacts[str(path.resolve()).lower()] = {
            "label": f"{stage} / {path.name}",
            "path": str(path.resolve()),
            "stage": stage,
            "kind": _artifact_kind(path),
        }
    return sorted(artifacts.values(), key=lambda item: (item["stage"], item["label"].lower()))


def _query_tool_events(run_id: str) -> list[dict[str, Any]]:
    init_db()
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT tool_name, status, started_at, completed_at, message, details_json
            FROM tool_events
            WHERE run_id = ?
            ORDER BY started_at ASC, id ASC
            """,
            (run_id,),
        ).fetchall()
    return [dict(row) for row in rows]


def _render_tool_timeline_html(run_id: str | None) -> str:
    if not run_id:
        return '<div class="panel-card"><div class="section-label">Execution Timeline</div><div class="tiny">No run selected.</div></div>'
    events = _query_tool_events(run_id)
    if not events:
        return '<div class="panel-card"><div class="section-label">Execution Timeline</div><div class="tiny">No tool events recorded for this run yet.</div></div>'

    items = []
    for event in events:
        status = str(event["status"]).lower()
        status_class = "completed" if status == "completed" else "failed" if status == "failed" else "pending"
        details = f"<div class='tiny'>{event['message']}</div>" if event.get("message") else ""
        items.append(
            f"""
            <tr>
              <th>{event['tool_name']}</th>
              <td>
                <span class="status-pill {status_class}">{event['status']}</span><br>
                <span class="tiny">{event['started_at']} -> {event['completed_at']}</span>
                {details}
              </td>
            </tr>
            """
        )
    return f"""
    <div class="panel-card">
      <div class="section-label">Execution Timeline</div>
      <table class="spec-table">{''.join(items)}</table>
    </div>
    """


def _render_run_overview_html(run_id: str | None) -> str:
    if not run_id:
        return '<div class="panel-card"><div class="section-label">Run Overview</div><div class="tiny">Select a run to inspect its artifacts and metadata.</div></div>'
    run_paths = get_registered_run_paths(run_id)
    if run_paths is None or not run_paths.root.exists():
        return f'<div class="panel-card"><div class="section-label">Run Overview</div><div class="tiny">Run {run_id} is not available on disk.</div></div>'

    meta_path = run_paths.meta / "run.json"
    meta = _safe_json_load(meta_path) or {}
    artifacts = _collect_artifacts(run_id)
    counts: dict[str, int] = {}
    for artifact in artifacts:
        counts[artifact["stage"]] = counts.get(artifact["stage"], 0) + 1

    rows = [
        ("Run ID", run_id),
        ("Root", str(run_paths.root.resolve())),
        ("Status", meta.get("status", "unknown")),
        ("Inputs", counts.get("inputs", 0)),
        ("Outputs", counts.get("outputs", 0)),
        ("Visualizations", counts.get("viz", 0)),
        ("Logs", counts.get("logs", 0)),
        ("Verification files", counts.get("verification", 0)),
    ]
    if meta.get("units"):
        rows.append(("Units", _format_json(meta["units"])))

    rendered_rows = "".join(f"<tr><th>{label}</th><td>{_format_value(value)}</td></tr>" for label, value in rows)
    return f"""
    <div class="panel-card">
      <div class="section-label">Run Overview</div>
      <table class="spec-table">{rendered_rows}</table>
    </div>
    """


def _gallery_items(run_id: str | None) -> list[tuple[str, str]]:
    if not run_id:
        return []
    items = []
    for artifact in _collect_artifacts(run_id):
        if artifact["stage"] == "viz" and artifact["kind"] == "image":
            items.append((artifact["path"], artifact["label"]))
    return items


def _verification_available(run_id: str | None) -> bool:
    if not run_id:
        return False
    run_paths = get_registered_run_paths(run_id)
    return bool(run_paths and (run_paths.root / "verification").exists())


def _artifact_choices(run_id: str | None) -> list[tuple[str, str]]:
    if not run_id:
        return []
    return [(artifact["label"], artifact["path"]) for artifact in _collect_artifacts(run_id)]


def _compare_choices(run_id: str | None) -> list[tuple[str, str]]:
    if not run_id:
        return []
    options = []
    for artifact in _collect_artifacts(run_id):
        if artifact["kind"] == "image":
            options.append((artifact["label"], artifact["path"]))
    return options


def _artifact_preview(path_str: str | None) -> tuple[str, Any, dict[str, Any], str]:
    if not path_str:
        return "No artifact selected.", None, {}, ""

    path = Path(path_str)
    if not path.exists():
        return f"{path.name}\n\nFile is missing on disk.", None, {}, ""

    kind = _artifact_kind(path)
    title = f"{path.name}\n\n`{path}`"
    if kind == "image":
        return title, str(path), {}, ""
    if kind == "json":
        data = _safe_json_load(path)
        return title, None, data or {}, ""
    if kind == "text":
        return title, None, {}, _read_text(path)
    return title, None, {}, f"Binary artifact.\nPath: {path}"


def _compose_run_panel_updates(selected_run_id: str | None) -> tuple[Any, ...]:
    runs = _query_runs()
    run_choices = [(run["run_id"], run["run_id"]) for run in runs]
    selected = selected_run_id or (runs[0]["run_id"] if runs else None)
    artifact_choices = _artifact_choices(selected)
    compare_choices = _compare_choices(selected)
    verification_visible = _verification_available(selected)
    preview_choice = artifact_choices[0][1] if artifact_choices else None
    preview_title, preview_image, preview_json, preview_text = _artifact_preview(preview_choice)

    return (
        gr.update(choices=run_choices, value=selected),
        _runs_dataframe_rows(runs),
        _render_run_overview_html(selected),
        _render_tool_timeline_html(selected),
        _gallery_items(selected),
        gr.update(choices=artifact_choices, value=preview_choice),
        preview_title,
        gr.update(value=preview_image, visible=bool(preview_image)),
        gr.update(value=preview_json, visible=bool(preview_json)),
        gr.update(value=preview_text, visible=bool(preview_text)),
        gr.update(visible=verification_visible),
        gr.update(choices=compare_choices, value=None),
        gr.update(choices=compare_choices, value=None),
        gr.update(choices=compare_choices, value=None),
        gr.update(choices=compare_choices, value=None),
    )


def _selected_run_for_ui(session: FrontendSession, requested_run_id: str | None) -> str | None:
    return requested_run_id or session.last_run_id


def _process_message(user_text: str, session: FrontendSession | None, selected_run_id: str | None):
    session = session or FrontendSession()
    user_text = (user_text or "").strip()
    if not user_text:
        selected = _selected_run_for_ui(session, selected_run_id)
        run_updates = _compose_run_panel_updates(selected)
        return (
            session.chat_messages,
            session,
            _render_workflow_html(session),
            _render_parameter_sheet_html(session),
            "",
            *run_updates,
        )

    if is_exit_input(user_text):
        assistant_text = "Workflow stopped. Start a new session or inspect previous runs."
        session.append_exchange(user_text, assistant_text)
        selected = _selected_run_for_ui(session, selected_run_id)
        run_updates = _compose_run_panel_updates(selected)
        return (
            session.chat_messages,
            session,
            _render_workflow_html(session),
            _render_parameter_sheet_html(session),
            "",
            *run_updates,
        )

    ambiguous_message = get_ambiguous_dimension_message(user_text)
    if ambiguous_message:
        session.append_exchange(user_text, ambiguous_message)
        selected = _selected_run_for_ui(session, selected_run_id)
        run_updates = _compose_run_panel_updates(selected)
        return (
            session.chat_messages,
            session,
            _render_workflow_html(session),
            _render_parameter_sheet_html(session),
            "",
            *run_updates,
        )

    try:
        session.ensure_models()
        turn_update = session.sheet.update_from_text(user_text, llm=session.extractor_llm)
        state_input = build_state_aware_input(user_text, session.sheet, turn_update=turn_update)
        parsed_update = turn_update.get("applied_updates", {})
        current_signature = get_sheet_signature(session.sheet)
        if session.last_executed_signature is not None and current_signature != session.last_executed_signature:
            session.simulation_executed = False
            session.verification_prompt_pending = False

        if session.simulation_executed and session.verification_prompt_pending and is_confirmation_input(user_text):
            if not session.last_run_id:
                output_str = "I could not find a completed run to verify yet."
            else:
                output_str = (
                    run_verification_for_run(session.last_run_id)
                    + "\nYou can change parameters for another run, request verification again, or finish the process."
                )
            session.verification_prompt_pending = False
            session.append_exchange(user_text, output_str)
            selected = _selected_run_for_ui(session, session.last_run_id)
            run_updates = _compose_run_panel_updates(selected)
            return (
                session.chat_messages,
                session,
                _render_workflow_html(session),
                _render_parameter_sheet_html(session),
                "",
                *run_updates,
            )

        if session.simulation_executed and session.verification_prompt_pending and is_negative_response(user_text):
            output_str = "Okay. You can change parameters for another run, request verification later, or finish the process."
            session.verification_prompt_pending = False
            session.append_exchange(user_text, output_str)
            selected = _selected_run_for_ui(session, session.last_run_id)
            run_updates = _compose_run_panel_updates(selected)
            return (
                session.chat_messages,
                session,
                _render_workflow_html(session),
                _render_parameter_sheet_html(session),
                "",
                *run_updates,
            )

        if session.simulation_executed and is_verification_input(user_text):
            if not session.last_run_id:
                output_str = "I could not find a completed run to verify yet."
            else:
                output_str = (
                    run_verification_for_run(session.last_run_id)
                    + "\nYou can change parameters for another run, request verification again, or finish the process."
                )
            session.append_exchange(user_text, output_str)
            selected = _selected_run_for_ui(session, session.last_run_id)
            run_updates = _compose_run_panel_updates(selected)
            return (
                session.chat_messages,
                session,
                _render_workflow_html(session),
                _render_parameter_sheet_html(session),
                "",
                *run_updates,
            )

        if session.simulation_executed and not parsed_update:
            output_str = build_post_run_idle_message()
            session.append_exchange(user_text, output_str)
            selected = _selected_run_for_ui(session, session.last_run_id)
            run_updates = _compose_run_panel_updates(selected)
            return (
                session.chat_messages,
                session,
                _render_workflow_html(session),
                _render_parameter_sheet_html(session),
                "",
                *run_updates,
            )

        if session.simulation_executed and is_confirmation_input(user_text):
            output_str = build_post_run_message()
            session.append_exchange(user_text, output_str)
            selected = _selected_run_for_ui(session, session.last_run_id)
            run_updates = _compose_run_panel_updates(selected)
            return (
                session.chat_messages,
                session,
                _render_workflow_html(session),
                _render_parameter_sheet_html(session),
                "",
                *run_updates,
            )

        missing_fields = get_missing_fields_for_execution(session.sheet)
        if not missing_fields:
            if is_confirmation_input(user_text):
                run_output = str(run_full_pipeline.invoke(build_pipeline_kwargs_from_sheet(session.sheet))).strip()
                if is_successful_run_output(run_output):
                    session.last_run_id, _ = extract_run_id_and_folder(run_output)
                    output_str = build_post_success_message(run_output)
                    session.simulation_executed = True
                    session.last_executed_signature = get_sheet_signature(session.sheet)
                    session.verification_prompt_pending = True
                else:
                    output_str = run_output
                    session.simulation_executed = False
                    session.verification_prompt_pending = False
                session.append_exchange(user_text, output_str)
                selected = _selected_run_for_ui(session, session.last_run_id)
                run_updates = _compose_run_panel_updates(selected)
                return (
                    session.chat_messages,
                    session,
                    _render_workflow_html(session),
                    _render_parameter_sheet_html(session),
                    "",
                    *run_updates,
                )

            output_str = build_confirmation_summary(session.sheet)
            session.append_exchange(user_text, output_str)
            selected = _selected_run_for_ui(session, selected_run_id)
            run_updates = _compose_run_panel_updates(selected)
            return (
                session.chat_messages,
                session,
                _render_workflow_html(session),
                _render_parameter_sheet_html(session),
                "",
                *run_updates,
            )

        scenario = get_sheet_scenario(session.sheet)
        if scenario is None:
            output_str = (
                "Please choose the scenario first:\n"
                "- cantilever_tip_load\n"
                "- simply_supported_midpoint_load"
            )
            session.append_exchange(user_text, output_str)
            selected = _selected_run_for_ui(session, selected_run_id)
            run_updates = _compose_run_panel_updates(selected)
            return (
                session.chat_messages,
                session,
                _render_workflow_html(session),
                _render_parameter_sheet_html(session),
                "",
                *run_updates,
            )

        family_guidance = build_catalog_family_guidance(session.sheet)
        if family_guidance:
            session.append_exchange(user_text, family_guidance)
            selected = _selected_run_for_ui(session, selected_run_id)
            run_updates = _compose_run_panel_updates(selected)
            return (
                session.chat_messages,
                session,
                _render_workflow_html(session),
                _render_parameter_sheet_html(session),
                "",
                *run_updates,
            )

        result = session.agent_executor.invoke(
            {
                "input": state_input,
                "chat_history": session.lc_history,
            }
        )
        output_str = str(result.get("output", "")).strip() or "No response returned."
    except Exception as exc:
        output_str = (
            "The agent could not complete this step.\n"
            "Check the OpenAI configuration and local solver environment.\n\n"
            f"Details: {exc}"
        )

    session.append_exchange(user_text, output_str)
    selected = _selected_run_for_ui(session, selected_run_id)
    run_updates = _compose_run_panel_updates(selected)
    return (
        session.chat_messages,
        session,
        _render_workflow_html(session),
        _render_parameter_sheet_html(session),
        "",
        *run_updates,
    )


def _refresh_runs(session: FrontendSession | None, selected_run_id: str | None):
    session = session or FrontendSession()
    selected = _selected_run_for_ui(session, selected_run_id)
    run_updates = _compose_run_panel_updates(selected)
    return (
        session,
        _render_workflow_html(session),
        _render_parameter_sheet_html(session),
        *run_updates,
    )


def _select_run(run_id: str | None):
    return _compose_run_panel_updates(run_id)


def _preview_selected_artifact(path_str: str | None):
    title, image, payload, text = _artifact_preview(path_str)
    return (
        title,
        gr.update(value=image, visible=bool(image)),
        gr.update(value=payload, visible=bool(payload)),
        gr.update(value=text, visible=bool(text)),
    )


def _run_verification_from_button(session: FrontendSession | None, selected_run_id: str | None):
    session = session or FrontendSession()
    run_id = selected_run_id or session.last_run_id
    if not run_id:
        message = "Select a completed run before starting verification."
    else:
        message = run_verification_for_run(run_id)
        session.last_run_id = run_id
        if "Verification finished" in message:
            session.verification_prompt_pending = False
    session.chat_messages.append({"role": "assistant", "content": message})
    selected = _selected_run_for_ui(session, run_id)
    run_updates = _compose_run_panel_updates(selected)
    return (
        session.chat_messages,
        session,
        _render_workflow_html(session),
        _render_parameter_sheet_html(session),
        *run_updates,
    )


def _compare_images(path_a: str | None, path_b: str | None, path_c: str | None, path_d: str | None):
    paths = [path_a, path_b, path_c, path_d]
    updates = []
    for path_str in paths:
        label = ""
        value = None
        visible = False
        if path_str and Path(path_str).exists():
            value = path_str
            label = Path(path_str).name
            visible = True
        updates.extend(
            [
                gr.update(label=label or "Empty slot", value=value, visible=visible),
                gr.update(value=f"`{path_str}`" if value else "", visible=visible),
            ]
        )
    return tuple(updates)


def _new_session():
    session = FrontendSession()
    run_updates = _compose_run_panel_updates(None)
    return (
        [],
        session,
        _render_workflow_html(session),
        _render_parameter_sheet_html(session),
        "",
        *run_updates,
    )


def build_demo():
    if gr is None:
        raise RuntimeError(
            "Gradio is not installed. Install it with `python -m pip install gradio` and rerun `python frontend\\gui_gradio.py`."
        )

    with gr.Blocks(title="Beam Analysis Assistant", fill_width=True, fill_height=True) as demo:
        session_state = gr.State(FrontendSession())

        gr.HTML(
            """
            <div class="panel-card app-shell">
              <div class="section-label">Beam Analysis Assistant</div>
              <div class="run-title">Stateful Engineering Workflow</div>
              <div class="tiny">
                Chat-driven parameter collection, deterministic Code_Aster runs, artifact tracking, and optional verification inside each run.
              </div>
            </div>
            """
        )

        with gr.Row(equal_height=True):
            workflow_html = gr.HTML(_render_workflow_html(FrontendSession()), scale=7)
            with gr.Column(scale=3, min_width=320):
                refresh_btn = gr.Button("Refresh Runs", variant="secondary")
                new_session_btn = gr.Button("New Session", variant="secondary")
                confirm_btn = gr.Button("Confirm And Run", variant="primary")

        with gr.Row(equal_height=True):
            with gr.Column(scale=5, min_width=560):
                chatbox = gr.Chatbot(
                    value=[],
                    height=430,
                    label="Assistant Thread",
                )
                with gr.Row():
                    user_input = gr.Textbox(
                        label="Message",
                        placeholder="Describe the beam, refine parameters, ask for verification, or type yes to run.",
                        lines=2,
                        scale=9,
                    )
                    send_btn = gr.Button("Send", variant="primary", scale=1)
            with gr.Column(scale=4, min_width=420):
                parameter_html = gr.HTML(_render_parameter_sheet_html(FrontendSession()))
                selected_run = gr.Dropdown(label="Selected Run", choices=[], value=None)
                run_overview_html = gr.HTML(_render_run_overview_html(None))

        with gr.Tabs():
            with gr.Tab("Results"):
                gallery = gr.Gallery(
                    value=[],
                    label="Generated Figures",
                    columns=3,
                    height=260,
                    object_fit="contain",
                )
                tool_timeline_html = gr.HTML(_render_tool_timeline_html(None))

            with gr.Tab("Artifacts"):
                artifact_selector = gr.Dropdown(label="Artifact", choices=[], value=None)
                artifact_title = gr.Markdown("No artifact selected.")
                artifact_image = gr.Image(label="Image Preview", visible=False, interactive=False)
                artifact_json = gr.JSON(label="JSON Preview", visible=False)
                artifact_text = gr.Textbox(label="Text Preview", visible=False, lines=16, max_lines=22)

            with gr.Tab("Verification"):
                verification_panel = gr.Column(visible=False)
                with verification_panel:
                    gr.Markdown(
                        "Verification is tied to the selected run and writes into the run's `verification/` folder."
                    )
                    verification_btn = gr.Button("Run Verification", variant="secondary")

            with gr.Tab("Compare"):
                gr.Markdown("Select up to four image artifacts from the selected run.")
                with gr.Row():
                    compare_1 = gr.Dropdown(label="Figure A", choices=[], value=None)
                    compare_2 = gr.Dropdown(label="Figure B", choices=[], value=None)
                    compare_3 = gr.Dropdown(label="Figure C", choices=[], value=None)
                    compare_4 = gr.Dropdown(label="Figure D", choices=[], value=None)
                compare_btn = gr.Button("Update Compare View", variant="secondary")
                with gr.Row():
                    cmp_img_1 = gr.Image(label="Empty slot", visible=False, interactive=False)
                    cmp_img_2 = gr.Image(label="Empty slot", visible=False, interactive=False)
                with gr.Row():
                    cmp_lbl_1 = gr.Markdown(visible=False)
                    cmp_lbl_2 = gr.Markdown(visible=False)
                with gr.Row():
                    cmp_img_3 = gr.Image(label="Empty slot", visible=False, interactive=False)
                    cmp_img_4 = gr.Image(label="Empty slot", visible=False, interactive=False)
                with gr.Row():
                    cmp_lbl_3 = gr.Markdown(visible=False)
                    cmp_lbl_4 = gr.Markdown(visible=False)

            with gr.Tab("Run History"):
                runs_table = gr.Dataframe(
                    headers=["run_id", "status", "scenario", "length_m", "n_elems", "FY", "verification", "created_at"],
                    datatype=["str", "str", "str", "number", "number", "number", "str", "str"],
                    interactive=False,
                    wrap=True,
                    row_count=(1, "dynamic"),
                    column_count=(8, "fixed"),
                    value=[],
                )

        send_outputs = [
            chatbox,
            session_state,
            workflow_html,
            parameter_html,
            user_input,
            selected_run,
            runs_table,
            run_overview_html,
            tool_timeline_html,
            gallery,
            artifact_selector,
            artifact_title,
            artifact_image,
            artifact_json,
            artifact_text,
            verification_panel,
            compare_1,
            compare_2,
            compare_3,
            compare_4,
        ]

        send_btn.click(_process_message, inputs=[user_input, session_state, selected_run], outputs=send_outputs)
        user_input.submit(_process_message, inputs=[user_input, session_state, selected_run], outputs=send_outputs)
        confirm_btn.click(
            lambda session, run_id: _process_message("yes", session, run_id),
            inputs=[session_state, selected_run],
            outputs=send_outputs,
        )

        refresh_btn.click(
            _refresh_runs,
            inputs=[session_state, selected_run],
            outputs=[
                session_state,
                workflow_html,
                parameter_html,
                selected_run,
                runs_table,
                run_overview_html,
                tool_timeline_html,
                gallery,
                artifact_selector,
                artifact_title,
                artifact_image,
                artifact_json,
                artifact_text,
                verification_panel,
                compare_1,
                compare_2,
                compare_3,
                compare_4,
            ],
        )

        new_session_btn.click(_new_session, inputs=[], outputs=send_outputs)

        selected_run.change(
            _select_run,
            inputs=[selected_run],
            outputs=[
                selected_run,
                runs_table,
                run_overview_html,
                tool_timeline_html,
                gallery,
                artifact_selector,
                artifact_title,
                artifact_image,
                artifact_json,
                artifact_text,
                verification_panel,
                compare_1,
                compare_2,
                compare_3,
                compare_4,
            ],
        )

        artifact_selector.change(
            _preview_selected_artifact,
            inputs=[artifact_selector],
            outputs=[artifact_title, artifact_image, artifact_json, artifact_text],
        )

        verification_btn.click(
            _run_verification_from_button,
            inputs=[session_state, selected_run],
            outputs=[
                chatbox,
                session_state,
                workflow_html,
                parameter_html,
                selected_run,
                runs_table,
                run_overview_html,
                tool_timeline_html,
                gallery,
                artifact_selector,
                artifact_title,
                artifact_image,
                artifact_json,
                artifact_text,
                verification_panel,
                compare_1,
                compare_2,
                compare_3,
                compare_4,
            ],
        )

        compare_btn.click(
            _compare_images,
            inputs=[compare_1, compare_2, compare_3, compare_4],
            outputs=[cmp_img_1, cmp_lbl_1, cmp_img_2, cmp_lbl_2, cmp_img_3, cmp_lbl_3, cmp_img_4, cmp_lbl_4],
        )

        demo.load(_new_session, inputs=[], outputs=send_outputs)

    return demo


def main() -> None:
    demo = build_demo()
    demo.launch(css=APP_CSS)


if __name__ == "__main__":
    main()
