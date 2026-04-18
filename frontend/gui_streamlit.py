from __future__ import annotations

import json
import sqlite3
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import pandas as pd
    import streamlit as st
except ImportError as exc:  # pragma: no cover
    raise RuntimeError(
        "Streamlit (and pandas) are not installed. Run: python -m pip install streamlit pandas"
    ) from exc

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
    is_negative_response,
    is_successful_run_output,
    is_verification_input,
    run_full_pipeline,
    run_verification_for_run,
)
from storage import DB_PATH, RUNS_ROOT, get_registered_run_paths, init_db

CHART_DISPLAY_NAMES = {
    "beam_job_def.png": "Deflection Diagram",
    "beam_job_rot.png": "Rotation Diagram",
    "freebody.png": "Free Body Diagram",
    "mfz_contour.png": "Bending Moment Contour (Z-axis)",
    "moment_diagram.png": "Moment Diagram",
    "shear_diagram.png": "Shear Diagram",
}


def _create_workspace_id() -> str:
    stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    return f"chat_{stamp}_{uuid.uuid4().hex[:6]}"


def _default_workspace(counter: int) -> dict[str, Any]:
    return {
        "name": f"Chat {counter}",
        "run_ids": [],
        "sheet": ParameterSheet(),
        "chat_messages": [],
        "lc_history": [],
        "simulation_executed": False,
        "last_executed_signature": None,
        "last_run_id": None,
        "verification_prompt_pending": False,
        "pending_user_message": None,
    }


def _ensure_state() -> None:
    if "workspaces" not in st.session_state:
        ws_id = _create_workspace_id()
        st.session_state.workspace_counter = 1
        st.session_state.workspaces = {ws_id: _default_workspace(1)}
        st.session_state.active_workspace_id = ws_id

    if "run_labels" not in st.session_state:
        st.session_state.run_labels = {}

    if "sheet" not in st.session_state:
        _load_workspace_state(st.session_state.active_workspace_id)

    if "pending_assistant_typewriter_idx" not in st.session_state:
        st.session_state.pending_assistant_typewriter_idx = None

    if "chat_draft" not in st.session_state:
        st.session_state.chat_draft = ""

    if "sidebar_intro_done" not in st.session_state:
        st.session_state.sidebar_intro_done = False


@st.cache_resource
def _get_models():
    return build_agent(), build_state_extractor_llm()


def _append_user_message(user_text: str) -> None:
    st.session_state.chat_messages.append({"role": "user", "content": user_text})
    st.session_state.lc_history.append(HumanMessage(content=user_text))


def _append_assistant_message(assistant_text: str) -> None:
    st.session_state.chat_messages.append({"role": "assistant", "content": assistant_text})
    st.session_state.pending_assistant_typewriter_idx = len(st.session_state.chat_messages) - 1
    st.session_state.lc_history.append(AIMessage(content=assistant_text))


def _append_exchange(user_text: str, assistant_text: str) -> None:
    _append_user_message(user_text)
    _append_assistant_message(assistant_text)


def _stream_typewriter_text(text: str, delay_seconds: float = 0.02):
    for chunk in text:
        yield chunk
        time.sleep(delay_seconds)


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
    out = []
    for row in rows:
        try:
            params = json.loads(row["params_json"] or "{}")
        except Exception:
            params = {}
        run_dir = Path(row["run_dir"])
        out.append(
            {
                "run_id": row["run_id"],
                "status": row["status"],
                "scenario": params.get("scenario", ""),
                "created_at": row["created_at"],
                "params": params,
                "verification": "yes" if (run_dir / "verification").exists() else "no",
            }
        )
    return out


def _scenario_label(scenario: str) -> str:
    if scenario == "simply_supported_midpoint_load":
        return "simple_beam"
    if scenario == "cantilever_tip_load":
        return "cantilever"
    return "beam"


def _format_force_label(params: dict[str, Any]) -> str:
    force = params.get("FY")
    unit = str(params.get("force_unit") or "N").lower()
    try:
        value = abs(float(force))
    except Exception:
        return "load"
    if unit == "kn":
        return f"{value:g}kN"
    if unit == "n":
        return f"{(value / 1000.0):g}kN"
    return f"{value:g}{unit}"


def _run_display_name(run: dict[str, Any]) -> str:
    run_id = run["run_id"]
    if run_id in st.session_state.run_labels:
        return st.session_state.run_labels[run_id]

    params = run.get("params", {})
    scenario = _scenario_label(str(params.get("scenario") or ""))
    load = _format_force_label(params)
    material = str(params.get("material") or params.get("material_hint") or "material")
    section = str(params.get("section") or params.get("section_hint") or "section")
    compact = f"{scenario}/{load}/{material}/{section}"
    suffix = run_id[-6:] if len(run_id) >= 6 else run_id
    return f"{compact} ({suffix})"


def _active_workspace() -> dict[str, Any]:
    return st.session_state.workspaces[st.session_state.active_workspace_id]


def _save_active_workspace_state() -> None:
    ws = _active_workspace()
    ws["sheet"] = st.session_state.sheet
    ws["chat_messages"] = st.session_state.chat_messages
    ws["lc_history"] = st.session_state.lc_history
    ws["simulation_executed"] = st.session_state.simulation_executed
    ws["last_executed_signature"] = st.session_state.last_executed_signature
    ws["last_run_id"] = st.session_state.last_run_id
    ws["verification_prompt_pending"] = st.session_state.verification_prompt_pending
    ws["pending_user_message"] = st.session_state.get("pending_user_message")


def _load_workspace_state(workspace_id: str) -> None:
    ws = st.session_state.workspaces[workspace_id]
    st.session_state.sheet = ws["sheet"]
    st.session_state.chat_messages = ws["chat_messages"]
    st.session_state.lc_history = ws["lc_history"]
    st.session_state.simulation_executed = ws["simulation_executed"]
    st.session_state.last_executed_signature = ws["last_executed_signature"]
    st.session_state.last_run_id = ws["last_run_id"]
    st.session_state.verification_prompt_pending = ws["verification_prompt_pending"]
    st.session_state.pending_user_message = ws.get("pending_user_message")


def _workspace_runs(all_runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ws_run_ids = set(_active_workspace().get("run_ids", []))
    return [run for run in all_runs if run["run_id"] in ws_run_ids]


def _run_root(run_id: str) -> Path:
    run_paths = get_registered_run_paths(run_id)
    return run_paths.root if run_paths else (RUNS_ROOT / run_id)


def _artifact_kind(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in {".png", ".jpg", ".jpeg", ".bmp", ".gif"}:
        return "image"
    if ext == ".json":
        return "json"
    if ext in {".txt", ".log", ".comm", ".geo", ".export", ".mess", ".resu", ".out", ".csv", ".md", ".tcl"}:
        return "text"
    return "binary"


def _chart_display_name(filename: str) -> str:
    if filename in CHART_DISPLAY_NAMES:
        return CHART_DISPLAY_NAMES[filename]
    stem = Path(filename).stem.replace("_", " ").strip()
    return stem.title() if stem else filename


def _collect_artifacts(run_id: str) -> list[dict[str, str]]:
    root = _run_root(run_id)
    if not root.exists():
        return []
    items = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        try:
            stage = path.relative_to(root).parts[0]
        except Exception:
            stage = "other"
        items.append(
            {
                "label": f"{stage} / {path.name}",
                "path": str(path.resolve()),
                "stage": stage,
                "kind": _artifact_kind(path),
            }
        )
    return items


def _detect_phase() -> str:
    missing = get_missing_fields_for_execution(st.session_state.sheet)
    if st.session_state.simulation_executed and st.session_state.verification_prompt_pending:
        return "verification"
    if st.session_state.simulation_executed:
        return "interpretation"
    if not st.session_state.sheet.to_dict():
        return "elicitation"
    if not missing:
        return "confirmation"
    return "elicitation"


def _record_workspace_run(run_id: str) -> None:
    ws = _active_workspace()
    run_ids = ws.setdefault("run_ids", [])
    if run_id in run_ids:
        run_ids.remove(run_id)
    run_ids.insert(0, run_id)


def _record_run_label(run_id: str) -> None:
    resolved = st.session_state.sheet.resolved_state().get("resolved", {})
    material = resolved.get("resolved_material") or {}
    section = resolved.get("resolved_section") or {}
    scenario = _scenario_label(str(resolved.get("scenario") or get_sheet_scenario(st.session_state.sheet) or ""))
    force = _format_force_label(resolved)
    material_name = material.get("name") or st.session_state.sheet.to_dict().get("material_hint") or "material"
    section_name = (
        section.get("Profile")
        or section.get("Section")
        or section.get("shape")
        or st.session_state.sheet.to_dict().get("section_hint")
        or "section"
    )
    st.session_state.run_labels[run_id] = f"{scenario}/{force}/{material_name}/{section_name} ({run_id[-6:]})"


def _start_new_chat() -> None:
    _save_active_workspace_state()
    st.session_state.workspace_counter += 1
    ws_id = _create_workspace_id()
    st.session_state.workspaces[ws_id] = _default_workspace(st.session_state.workspace_counter)
    st.session_state.active_workspace_id = ws_id
    _load_workspace_state(ws_id)


def _process_message(user_text: str, user_already_appended: bool = False) -> None:
    text = (user_text or "").strip()
    if not text:
        return

    def _respond(assistant_text: str) -> None:
        if user_already_appended:
            _append_assistant_message(assistant_text)
        else:
            _append_exchange(text, assistant_text)

    ambiguous = get_ambiguous_dimension_message(text)
    if ambiguous:
        _respond(ambiguous)
        return

    agent_executor, extractor_llm = _get_models()
    sheet: ParameterSheet = st.session_state.sheet
    turn_update = sheet.update_from_text(text, llm=extractor_llm)
    state_input = build_state_aware_input(text, sheet, turn_update=turn_update)
    parsed_update = turn_update.get("applied_updates", {})
    current_signature = get_sheet_signature(sheet)
    if st.session_state.last_executed_signature and current_signature != st.session_state.last_executed_signature:
        st.session_state.simulation_executed = False
        st.session_state.verification_prompt_pending = False

    if st.session_state.simulation_executed and st.session_state.verification_prompt_pending and is_confirmation_input(text):
        if not st.session_state.last_run_id:
            out = "I could not find a completed run to verify yet."
        else:
            out = run_verification_for_run(st.session_state.last_run_id)
        st.session_state.verification_prompt_pending = False
        _respond(out)
        return

    if st.session_state.simulation_executed and st.session_state.verification_prompt_pending and is_negative_response(text):
        st.session_state.verification_prompt_pending = False
        _respond("Okay. You can request verification later.")
        return

    if st.session_state.simulation_executed and is_verification_input(text):
        out = (
            run_verification_for_run(st.session_state.last_run_id)
            if st.session_state.last_run_id
            else "I could not find a completed run to verify yet."
        )
        _respond(out)
        return

    if st.session_state.simulation_executed and not parsed_update:
        _respond(build_post_run_idle_message())
        return

    if st.session_state.simulation_executed and is_confirmation_input(text):
        _respond(build_post_run_message())
        return

    missing = get_missing_fields_for_execution(sheet)
    if not missing:
        if is_confirmation_input(text):
            run_output = str(run_full_pipeline.invoke(build_pipeline_kwargs_from_sheet(sheet))).strip()
            if is_successful_run_output(run_output):
                st.session_state.last_run_id, _ = extract_run_id_and_folder(run_output)
                st.session_state.simulation_executed = True
                st.session_state.last_executed_signature = get_sheet_signature(sheet)
                st.session_state.verification_prompt_pending = True
                if st.session_state.last_run_id:
                    _record_workspace_run(st.session_state.last_run_id)
                    _record_run_label(st.session_state.last_run_id)
                out = build_post_success_message(run_output)
            else:
                out = run_output
            _respond(out)
            return

        _respond(build_confirmation_summary(sheet))
        return

    if get_sheet_scenario(sheet) is None:
        _respond("Please choose scenario: cantilever_tip_load or simply_supported_midpoint_load")
        return

    family_guidance = build_catalog_family_guidance(sheet)
    if family_guidance:
        _respond(family_guidance)
        return

    try:
        result = agent_executor.invoke({"input": state_input, "chat_history": st.session_state.lc_history})
        out = str(result.get("output", "")).strip() or "No response returned."
    except Exception as exc:
        out = f"Agent error: {exc}"
    _respond(out)


def _inject_css() -> None:
    sidebar_intro_css = ""
    if not st.session_state.get("sidebar_intro_done", False):
        sidebar_intro_css = """
        section[data-testid="stSidebar"] {
            opacity: 0;
            transform: translateX(-14px);
            animation: sidebarReveal 380ms ease-out 180ms forwards;
        }
        @keyframes sidebarReveal {
            from { opacity: 0; transform: translateX(-14px); }
            to { opacity: 1; transform: translateX(0); }
        }
        """

    css = """
        <style>
        .stApp { font-size: 13px; }
        .block-container { max-width: 1700px; padding-top: 1rem; padding-bottom: 1rem; }
        h1, h2, h3, h4, h5 { margin-top: 0.2rem; margin-bottom: 0.4rem; }
        .stButton>button { padding: 0.35rem 0.6rem; font-size: 13px; }
        .stTextInput input, .stSelectbox, .stTextArea textarea { font-size: 13px !important; }
        div[data-testid="stForm"] div[data-testid="column"] .stButton>button {
            height: 2.45rem;
            border-radius: 999px;
            font-weight: 600;
            padding: 0 1rem;
        }
        div[data-testid="stPopover"] > button {
            border-radius: 999px;
            padding: 0.2rem 0.7rem;
            font-weight: 600;
            min-height: 2rem;
        }
        .app-footer {
            position: fixed;
            left: 21rem;
            right: 0;
            bottom: 0;
            z-index: 999;
            background: rgba(255, 255, 255, 0.96);
            border-top: 1px solid #e6e6e6;
            padding: 0.45rem 1rem 0.55rem 1rem;
            font-size: 12px;
            line-height: 1.35;
        }
        .app-footer .line1 { color: #4a4a4a; }
        .app-footer .line2 { color: #6a6a6a; margin-top: 0.2rem; }
        @media (max-width: 900px) {
            .app-footer { left: 0; }
        }
        </style>
        """
    css = css.replace("</style>", f"{sidebar_intro_css}\n</style>")
    st.markdown(
        css,
        unsafe_allow_html=True,
    )


def _safe_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return None


def _render_parameter_summary() -> None:
    sheet = st.session_state.sheet
    resolved_state = sheet.resolved_state()
    values = sheet.to_dict()
    resolved = resolved_state.get("resolved", {})
    missing = resolved_state.get("missing", [])

    section = resolved.get("resolved_section") or {}
    material = resolved.get("resolved_material") or {}

    st.write("Current Inputs")
    length_value = values.get("length_m")
    length_display = f"{length_value} m" if length_value not in (None, "") else "-"
    force_value = values.get("FY")
    force_unit = values.get("force_unit") or "N"
    force_display = f"{force_value} {force_unit}" if force_value not in (None, "") else "-"
    raw_scenario = values.get("scenario")
    raw_length = values.get("length_m")
    raw_n_elems = values.get("n_elems")
    raw_fy = values.get("FY")
    raw_material = material.get("name") or values.get("material_hint")
    raw_section = section.get("Profile") or section.get("Section") or section.get("shape") or values.get("section_hint")

    rows = [
        {"Field": "scenario", "Value": raw_scenario or "-", "has_value": raw_scenario not in (None, "")},
        {"Field": "length_m", "Value": length_display, "has_value": raw_length not in (None, "")},
        {"Field": "n_elems", "Value": raw_n_elems or "-", "has_value": raw_n_elems not in (None, "")},
        {"Field": "FY", "Value": force_display, "has_value": raw_fy not in (None, "")},
        {
            "Field": "material",
            "Value": raw_material or "-",
            "has_value": raw_material not in (None, ""),
        },
        {
            "Field": "section",
            "Value": raw_section or "-",
            "has_value": raw_section not in (None, ""),
        },
    ]
    df = pd.DataFrame(rows)

    display_df = df[["Field", "Value"]].copy()
    has_value_by_index = df["has_value"].to_dict()

    def _color_row(row: pd.Series) -> list[str]:
        bg = "#d4edda" if bool(has_value_by_index.get(row.name, False)) else "#fff3cd"
        return [f"background-color: {bg}" for _ in row]

    st.dataframe(
        display_df.style.apply(
            _color_row,
            axis=1,
        ),
        use_container_width=True,
        hide_index=True,
    )
    if missing:
        st.warning(f"Missing required: {', '.join(missing)}")
    else:
        st.success("All required inputs are collected. Click Run or type `yes` in chat.")


def _tab_help_modal(title: str, body: str) -> None:
    left, right = st.columns([12, 1])
    with left:
        st.caption("")
    with right:
        if hasattr(st, "popover"):
            with st.popover("❔"):
                st.markdown(f"**{title}**")
                st.caption(body)
        else:
            st.caption(body)


def _run_selector(label: str, runs: list[dict[str, Any]], key: str, default_run: str | None = None) -> str | None:
    if not runs:
        st.info("No runs in this chat workspace yet.")
        return None

    run_ids = [r["run_id"] for r in runs]
    run_label_map = {rid: _run_display_name(next(r for r in runs if r["run_id"] == rid)) for rid in run_ids}

    index = 0
    if default_run in run_ids:
        index = run_ids.index(default_run)

    return st.selectbox(label, run_ids, index=index, key=key, format_func=lambda rid: run_label_map[rid])


def _render_verification_reports(run_id: str) -> None:
    root = _run_root(run_id)
    manual_path = root / "verification" / "manual" / "report.txt"
    opensees_path = root / "verification" / "opensees" / "report.txt"
    comparison_path = root / "verification" / "comparison" / "comparison_report.txt"

    st.write("Comparison Report")
    if comparison_path.exists():
        st.text(comparison_path.read_text(encoding="utf-8", errors="replace")[:40000])
    else:
        st.info("No comparison report yet.")

    report_choice = st.selectbox(
        "Verification Detail",
        ["OpenSees Report", "Manual Report"],
        key=f"verification_detail_selector_{run_id}",
    )
    if report_choice == "OpenSees Report":
        if opensees_path.exists():
            st.text(opensees_path.read_text(encoding="utf-8", errors="replace")[:40000])
        else:
            st.info("No OpenSees report yet.")
    else:
        if manual_path.exists():
            st.text(manual_path.read_text(encoding="utf-8", errors="replace")[:40000])
        else:
            st.info("No manual report yet.")


def main() -> None:
    st.set_page_config(page_title="AI Assistant for Beam Analysis", layout="wide")
    _inject_css()
    _ensure_state()

    all_runs = _query_runs()
    runs = _workspace_runs(all_runs)
    workspace = _active_workspace()

    st.title("Beam Analysis Copilot")
    st.caption("AI-assisted modeling, verification, and results interpretation")
    st.session_state.sidebar_intro_done = True

    with st.sidebar:
        if st.button(
            "New Chat",
            use_container_width=True,
            help="با زدن این دکمه یک چت جدید ساخته می‌شود و می‌توانید تحلیل‌های جدید را در یک فضای مجزا بررسی کنید.",
        ):
            _start_new_chat()
            st.rerun()

        ws_ids = list(st.session_state.workspaces.keys())
        ws_names = {wid: st.session_state.workspaces[wid]["name"] for wid in ws_ids}
        current_ws = st.selectbox(
            "Chat Workspace",
            ws_ids,
            index=ws_ids.index(st.session_state.active_workspace_id),
            format_func=lambda wid: ws_names[wid],
            key="workspace_selector",
        )
        if current_ws != st.session_state.active_workspace_id:
            _save_active_workspace_state()
            st.session_state.active_workspace_id = current_ws
            _load_workspace_state(current_ws)
            st.rerun()

        st.subheader("Run Explorer")
        selected_explorer_run = _run_selector(
            "Run",
            runs,
            key="run_explorer_selector",
            default_run=st.session_state.last_run_id,
        )
        if runs:
            rows = []
            for run in runs:
                rows.append(
                    {
                        "name": _run_display_name(run),
                        "status": run["status"],
                        "verification": run["verification"],
                        "created_at": run["created_at"],
                    }
                )
            st.dataframe(rows, use_container_width=True, hide_index=True)

        st.subheader("Parameter Sheet")
        _render_parameter_summary()

    st.subheader("Chat")
    phase = _detect_phase()
    chat_placeholder = (
        "Example: simply_supported_midpoint_load, length 6 m, FY -12 kN, n_elems 40, material S355, section IPE200"
        if len(st.session_state.chat_messages) == 0
        else ""
    )
    if phase == "confirmation" and not st.session_state.chat_draft:
        st.session_state.chat_draft = "yes"

    pending_typewriter_idx = st.session_state.pending_assistant_typewriter_idx
    for idx, m in enumerate(st.session_state.chat_messages):
        with st.chat_message(m["role"]):
            if m["role"] == "assistant" and pending_typewriter_idx == idx:
                st.write_stream(_stream_typewriter_text(m["content"]))
                st.session_state.pending_assistant_typewriter_idx = None
            else:
                st.write(m["content"])

    with st.form("chat_input_form", clear_on_submit=True):
        input_col, send_col = st.columns([10, 1])
        with input_col:
            st.text_input(
                "Message",
                placeholder=chat_placeholder,
                label_visibility="collapsed",
                key="chat_draft",
            )
        with send_col:
            send_clicked = st.form_submit_button("Send", type="primary", use_container_width=True)
    if send_clicked and st.session_state.chat_draft.strip():
        submitted_text = st.session_state.chat_draft.strip()
        _append_user_message(submitted_text)
        st.session_state.pending_user_message = submitted_text
        _save_active_workspace_state()
        st.rerun()

    if st.session_state.pending_user_message:
        pending_text = str(st.session_state.pending_user_message)
        st.session_state.pending_user_message = None
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                _process_message(pending_text, user_already_appended=True)
        _save_active_workspace_state()
        st.rerun()

    missing_inputs = get_missing_fields_for_execution(st.session_state.sheet)
    if not missing_inputs and not st.session_state.simulation_executed:
        run_cols = st.columns([1, 1, 1])
        if run_cols[1].button("Run (Yes)", key="quick_run_yes", use_container_width=True):
            with st.spinner("Running simulation, please wait..."):
                _process_message("yes")
            _save_active_workspace_state()
            st.rerun()
    if st.session_state.verification_prompt_pending:
        verify_cols = st.columns([2, 1, 1, 2])
        if verify_cols[1].button("Run Verification (Yes)", key="quick_verify_yes", use_container_width=True):
            with st.spinner("Running verification, please wait..."):
                _process_message("yes")
            _save_active_workspace_state()
            st.rerun()
        if verify_cols[2].button("Skip Verification (No)", key="quick_verify_no", use_container_width=True):
            _process_message("no")
            _save_active_workspace_state()
            st.rerun()

    if runs:
        tabs = st.tabs(["Results", "Artifacts", "Verification", "Compare"])

        with tabs[0]:
            _tab_help_modal(
                "Results",
                "Charts and result figures generated for the selected run are shown here.",
            )
            run_id = _run_selector("Run for Results", runs, key="run_results_selector", default_run=selected_explorer_run)
            if run_id:
                imgs = [a for a in _collect_artifacts(run_id) if a["stage"] == "viz" and a["kind"] == "image"]
                if not imgs:
                    st.info("No generated figures yet.")
                else:
                    cols = st.columns(3)
                    for idx, item in enumerate(imgs):
                        cols[idx % 3].image(item["path"], caption=item["label"], use_container_width=True)

        with tabs[1]:
            _tab_help_modal(
                "Artifacts",
                "Browse run files from inputs and solver outputs for quick inspection.",
            )
            run_id = _run_selector("Run for Artifacts", runs, key="run_artifacts_selector", default_run=selected_explorer_run)
            if run_id:
                root = _run_root(run_id)
                st.caption(str(root))

                st.write("Code_Aster Output Files")
                for file_name in ["beam_job.mess", "beam_job.resu", "tab_disp.txt", "tab_sie.txt", "tab_reac.txt"]:
                    path = root / "outputs" / file_name
                    if path.exists():
                        with st.expander(file_name):
                            st.text(path.read_text(encoding="utf-8", errors="replace")[:40000])

                input_artifacts = []
                for path in sorted((root / "inputs").glob("*")):
                    if not path.is_file():
                        continue
                    if path.suffix.lower() not in {".comm", ".geo"}:
                        continue
                    input_artifacts.append(
                        {
                            "label": path.name,
                            "path": str(path.resolve()),
                            "kind": _artifact_kind(path),
                        }
                    )

                labels = [a["label"] for a in input_artifacts]
                if not labels:
                    st.info("No run input files found.")
                else:
                    sel = st.selectbox("run input files", labels, key="run_input_file_sel")
                    art = next(a for a in input_artifacts if a["label"] == sel)
                    path = Path(art["path"])
                    st.caption(str(path))
                    if art["kind"] == "image":
                        st.image(str(path), use_container_width=True)
                    elif art["kind"] == "json":
                        st.json(json.loads(path.read_text(encoding="utf-8", errors="replace")))
                    elif art["kind"] == "text":
                        st.text(path.read_text(encoding="utf-8", errors="replace")[:30000])
                    else:
                        st.warning("Binary artifact preview is not supported.")

        with tabs[2]:
            _tab_help_modal(
                "Verification",
                "Run and inspect OpenSees/manual checks and the comparison report for one run.",
            )
            run_id = _run_selector("Run for Verification", runs, key="run_verification_selector", default_run=selected_explorer_run)
            if run_id:
                st.caption("Verification is run-specific. Three reports are shown below for the selected run.")
                _render_verification_reports(run_id)

        with tabs[3]:
            _tab_help_modal(
                "Compare",
                "Select a chart type and place up to four runs side-by-side for visual comparison.",
            )
            run_to_images = {
                run["run_id"]: [a for a in _collect_artifacts(run["run_id"]) if a["stage"] == "viz" and a["kind"] == "image"]
                for run in runs
            }
            chart_types = sorted({Path(img["path"]).name for images in run_to_images.values() for img in images})
            if not chart_types:
                st.info("No image charts found in runs.")
            else:
                selected_chart = st.selectbox(
                    "Chart Type",
                    chart_types,
                    key="cmp_chart_type",
                    format_func=_chart_display_name,
                )
                run_ids = [r["run_id"] for r in runs]
                run_label_map = {rid: _run_display_name(next(r for r in runs if r["run_id"] == rid)) for rid in run_ids}
                run_labels = ["(empty)"] + run_ids
                c1, c2, c3, c4 = st.columns(4)
                selected_runs = [
                    c1.selectbox("A", run_labels, key="cmp_run_a", format_func=lambda rid: "(empty)" if rid == "(empty)" else run_label_map[rid]),
                    c2.selectbox("B", run_labels, key="cmp_run_b", format_func=lambda rid: "(empty)" if rid == "(empty)" else run_label_map[rid]),
                    c3.selectbox("C", run_labels, key="cmp_run_c", format_func=lambda rid: "(empty)" if rid == "(empty)" else run_label_map[rid]),
                    c4.selectbox("D", run_labels, key="cmp_run_d", format_func=lambda rid: "(empty)" if rid == "(empty)" else run_label_map[rid]),
                ]

                grid1, grid2 = st.columns(2)
                slots = [grid1, grid2, grid1, grid2]
                for idx, chosen_run in enumerate(selected_runs):
                    if chosen_run == "(empty)":
                        continue
                    matched = next(
                        (img for img in run_to_images.get(chosen_run, []) if Path(img["path"]).name == selected_chart),
                        None,
                    )
                    if matched:
                        slots[idx].image(
                            matched["path"],
                            caption=f"{_run_display_name(next(r for r in runs if r['run_id'] == chosen_run))} | {_chart_display_name(selected_chart)}",
                            use_container_width=True,
                        )
                    else:
                        slots[idx].warning(f"{_chart_display_name(selected_chart)} not found in selected run.")
    else:
        if _detect_phase() == "confirmation":
            st.info("After confirming analysis with `yes`, result tabs will appear here.")

    st.markdown("<div style='height: 4.5rem;'></div>", unsafe_allow_html=True)
    st.markdown(
        """
        <div class="app-footer">
          <div class="line1">About this app: Conversational beam-analysis assistant for defining inputs, running deterministic analysis, and reviewing artifacts.</div>
          <div class="line2">Workflow: 1) Describe scenario, geometry, material, section, mesh, and load. 2) Confirm with <code>yes</code> to run. 3) Inspect Results/Artifacts. 4) Optionally run Verification and Compare runs.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
