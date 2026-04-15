from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import streamlit as st
except ImportError as exc:  # pragma: no cover
    raise RuntimeError(
        "Streamlit is not installed. Run: python -m pip install streamlit"
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


def _ensure_state() -> None:
    if "sheet" not in st.session_state:
        st.session_state.sheet = ParameterSheet()
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []
    if "lc_history" not in st.session_state:
        st.session_state.lc_history = []
    if "simulation_executed" not in st.session_state:
        st.session_state.simulation_executed = False
    if "last_executed_signature" not in st.session_state:
        st.session_state.last_executed_signature = None
    if "last_run_id" not in st.session_state:
        st.session_state.last_run_id = None
    if "verification_prompt_pending" not in st.session_state:
        st.session_state.verification_prompt_pending = False
    if "selected_run_id" not in st.session_state:
        st.session_state.selected_run_id = None


@st.cache_resource
def _get_models():
    return build_agent(), build_state_extractor_llm()


def _append_exchange(user_text: str, assistant_text: str) -> None:
    st.session_state.chat_messages.append({"role": "user", "content": user_text})
    st.session_state.chat_messages.append({"role": "assistant", "content": assistant_text})
    st.session_state.lc_history.append(HumanMessage(content=user_text))
    st.session_state.lc_history.append(AIMessage(content=assistant_text))


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
                "verification": "yes" if (run_dir / "verification").exists() else "no",
            }
        )
    return out


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


def _process_message(user_text: str) -> None:
    text = (user_text or "").strip()
    if not text:
        return

    ambiguous = get_ambiguous_dimension_message(text)
    if ambiguous:
        _append_exchange(text, ambiguous)
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
        _append_exchange(text, out)
        return

    if st.session_state.simulation_executed and st.session_state.verification_prompt_pending and is_negative_response(text):
        st.session_state.verification_prompt_pending = False
        _append_exchange(text, "Okay. You can request verification later.")
        return

    if st.session_state.simulation_executed and is_verification_input(text):
        out = (
            run_verification_for_run(st.session_state.last_run_id)
            if st.session_state.last_run_id
            else "I could not find a completed run to verify yet."
        )
        _append_exchange(text, out)
        return

    if st.session_state.simulation_executed and not parsed_update:
        _append_exchange(text, build_post_run_idle_message())
        return

    if st.session_state.simulation_executed and is_confirmation_input(text):
        _append_exchange(text, build_post_run_message())
        return

    missing = get_missing_fields_for_execution(sheet)
    if not missing:
        if is_confirmation_input(text):
            run_output = str(run_full_pipeline.invoke(build_pipeline_kwargs_from_sheet(sheet))).strip()
            if is_successful_run_output(run_output):
                st.session_state.last_run_id, _ = extract_run_id_and_folder(run_output)
                st.session_state.selected_run_id = st.session_state.last_run_id
                st.session_state.simulation_executed = True
                st.session_state.last_executed_signature = get_sheet_signature(sheet)
                st.session_state.verification_prompt_pending = True
                out = build_post_success_message(run_output)
            else:
                out = run_output
            _append_exchange(text, out)
            return

        _append_exchange(text, build_confirmation_summary(sheet))
        return

    if get_sheet_scenario(sheet) is None:
        _append_exchange(text, "Please choose scenario: cantilever_tip_load or simply_supported_midpoint_load")
        return

    family_guidance = build_catalog_family_guidance(sheet)
    if family_guidance:
        _append_exchange(text, family_guidance)
        return

    try:
        result = agent_executor.invoke({"input": state_input, "chat_history": st.session_state.lc_history})
        out = str(result.get("output", "")).strip() or "No response returned."
    except Exception as exc:
        out = f"Agent error: {exc}"
    _append_exchange(text, out)


def _inject_css() -> None:
    st.markdown(
        """
        <style>
        .stApp { font-size: 13px; }
        .block-container { max-width: 1700px; padding-top: 1rem; padding-bottom: 1rem; }
        h1, h2, h3, h4, h5 { margin-top: 0.2rem; margin-bottom: 0.4rem; }
        .stButton>button { padding: 0.35rem 0.6rem; font-size: 13px; }
        .stTextInput input, .stSelectbox, .stTextArea textarea { font-size: 13px !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(page_title="Beam Analysis Assistant", layout="wide")
    _inject_css()
    _ensure_state()

    runs = _query_runs()

    st.title("Beam Analysis Assistant")
    top_l, top_r = st.columns([3, 2])
    with top_l:
        st.caption(f"Phase: {_detect_phase()}")
    with top_r:
        c1, c2, c3 = st.columns(3)
        if c1.button("Refresh Runs", use_container_width=True):
            st.rerun()
        if c2.button("Confirm And Run", use_container_width=True):
            _process_message("yes")
            st.rerun()
        if c3.button("New Session", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

    with st.sidebar:
        st.subheader("Run Explorer")
        run_ids = [r["run_id"] for r in runs]
        default_idx = 0
        if st.session_state.selected_run_id in run_ids:
            default_idx = run_ids.index(st.session_state.selected_run_id)
        if run_ids:
            st.session_state.selected_run_id = st.selectbox("Selected Run", run_ids, index=default_idx)
        st.dataframe(runs, use_container_width=True, hide_index=True)

    col_chat, col_info = st.columns([2.1, 1.2], gap="small")
    with col_chat:
        st.subheader("Chat")
        for m in st.session_state.chat_messages:
            with st.chat_message(m["role"]):
                st.write(m["content"])
        msg = st.chat_input("Type your request (or yes to run).")
        if msg:
            _process_message(msg)
            st.rerun()

    with col_info:
        st.subheader("Parameter Sheet")
        resolved = st.session_state.sheet.resolved_state()
        st.json(
            {
                "values": st.session_state.sheet.to_dict(),
                "missing": resolved.get("missing", []),
                "notes": resolved.get("notes", []),
            },
            expanded=False,
        )

    selected_run = st.session_state.selected_run_id or st.session_state.last_run_id
    tabs = st.tabs(["Results", "Artifacts", "Verification", "Compare"])

    with tabs[0]:
        if not selected_run:
            st.info("Select a run.")
        else:
            imgs = [a for a in _collect_artifacts(selected_run) if a["stage"] == "viz" and a["kind"] == "image"]
            if not imgs:
                st.info("No generated figures yet.")
            else:
                cols = st.columns(3)
                for idx, item in enumerate(imgs):
                    cols[idx % 3].image(item["path"], caption=item["label"], use_container_width=True)

    with tabs[1]:
        if not selected_run:
            st.info("Select a run.")
        else:
            artifacts = _collect_artifacts(selected_run)
            labels = [a["label"] for a in artifacts]
            if not labels:
                st.info("No artifacts found.")
            else:
                sel = st.selectbox("Artifact", labels, key="artifact_sel")
                art = next(a for a in artifacts if a["label"] == sel)
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
        if not selected_run:
            st.info("Select a run.")
        else:
            st.caption("Verification is run-specific and stored under run/verification/")
            if st.button("Run Verification"):
                out = run_verification_for_run(selected_run)
                st.session_state.chat_messages.append({"role": "assistant", "content": out})
                st.rerun()

    with tabs[3]:
        if not selected_run:
            st.info("Select a run.")
        else:
            images = [a for a in _collect_artifacts(selected_run) if a["kind"] == "image"]
            if not images:
                st.info("No image artifacts to compare.")
            else:
                labels = ["(empty)"] + [a["label"] for a in images]
                c1, c2, c3, c4 = st.columns(4)
                picks = [
                    c1.selectbox("A", labels, key="cmp_a"),
                    c2.selectbox("B", labels, key="cmp_b"),
                    c3.selectbox("C", labels, key="cmp_c"),
                    c4.selectbox("D", labels, key="cmp_d"),
                ]
                grid1, grid2 = st.columns(2)
                slots = [grid1, grid2, grid1, grid2]
                for idx, pick in enumerate(picks):
                    if pick == "(empty)":
                        continue
                    item = next(a for a in images if a["label"] == pick)
                    slots[idx].image(item["path"], caption=item["label"], use_container_width=True)


if __name__ == "__main__":
    main()
