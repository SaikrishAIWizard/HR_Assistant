"""LangGraph workflow that dispatches deterministic HR business logic."""

from __future__ import annotations

import logging
import time
from datetime import date
from pathlib import Path
from typing import Literal
from typing_extensions import NotRequired, TypedDict

from langgraph.graph import END, START, StateGraph

from veltrix_assistant.conversation import (
    _is_ambiguous_leave_check,
    _is_cancellation,
    _is_confirmation,
    _is_request_id_only,
    _needs_retrieval,
    _response,
    _tool_intent,
    respond_to_message,
)
from veltrix_assistant.llm import LLMProvider, RequestPlan
from veltrix_assistant.policy.applicability import parse_role
from veltrix_assistant.provenance import ProvenanceTrace
from veltrix_assistant.retrieval.bm25 import BM25Retriever
from veltrix_assistant.retrieval.hybrid import HybridRetriever
from veltrix_assistant.schemas.chat import ChatResponse
from veltrix_assistant.schemas.security import EmployeeIdentity
from veltrix_assistant.security.guards import assess_user_message
from veltrix_assistant.security.identity import (
    IdentityDirectoryError,
    SessionBindingStore,
    lookup_employee_identity,
)
from veltrix_assistant.schemas.security import RequestSafetyDecision
from veltrix_assistant.tools.leave import LeaveTools


_LOGGER = logging.getLogger("veltrix_assistant.workflow")
Retriever = BM25Retriever | HybridRetriever


class ChatWorkflowState(TypedDict):
    message: str
    session_id: str
    employee_id: str
    identity: EmployeeIdentity
    retriever: Retriever
    tools: LeaveTools
    session_store: SessionBindingStore
    today: date
    llm_provider: LLMProvider | None
    trace: ProvenanceTrace
    pii_redaction_enabled: bool
    pii_redaction_mode: Literal["redact", "mask"]
    pii_sensitive_terms: tuple[str, ...]
    safety: NotRequired[RequestSafetyDecision]
    identity_valid: NotRequired[bool]
    intent: NotRequired[str | None]
    needs_retrieval: NotRequired[bool]
    request_plan: NotRequired[RequestPlan]
    planned_action: NotRequired[str | None]
    planned_search_query: NotRequired[str | None]
    workflow_stage: NotRequired[str]
    response: NotRequired[ChatResponse]


class ChatWorkflow:
    """Run one chat request through a conditional graph or compatibility path."""

    def __init__(
        self,
        *,
        database_path: str | Path,
        enabled: bool = True,
    ) -> None:
        self._database_path = Path(database_path)
        self._enabled = enabled
        self._graph = self._build_graph() if enabled else None

    def invoke(self, state: ChatWorkflowState) -> ChatResponse:
        started = time.perf_counter()
        try:
            if self._graph is None:
                state["trace"].record(
                    "workflow",
                    "langgraph_disabled",
                    {"fallback": "direct"},
                )
                response = _run_conversation(state)
                state["trace"].set_request_id(response.request_id)
            else:
                result = self._graph.invoke(state)
                response = result.get("response")
                if not isinstance(response, ChatResponse):
                    raise RuntimeError("The workflow did not produce a chat response.")
            state["trace"].record(
                "workflow",
                "completed",
                {
                    "duration_ms": int((time.perf_counter() - started) * 1000),
                    "route": response.route,
                },
            )
            return response
        except Exception as exc:
            state["trace"].record(
                "workflow",
                "failed",
                {"error_code": type(exc).__name__},
            )
            _LOGGER.error("Chat workflow failed error_code=%s", type(exc).__name__)
            raise

    def _build_graph(self):
        graph = StateGraph(ChatWorkflowState)
        graph.add_node("request_guard_node", self._request_guard_node)
        graph.add_node("identity_validation_node", self._identity_validation_node)
        graph.add_node("intent_router_node", self._intent_router_node)
        graph.add_node(
            "policy_retrieval_node",
            self._dispatch_node("policy_retrieval_node"),
        )
        graph.add_node(
            "hybrid_retrieval_node",
            self._dispatch_node("hybrid_retrieval_node"),
        )
        graph.add_node(
            "leave_tool_node",
            self._dispatch_node("leave_tool_node"),
        )
        graph.add_node(
            "clarification_node",
            self._dispatch_node("clarification_node"),
        )
        graph.add_node(
            "confirmation_node",
            self._dispatch_node("confirmation_node"),
        )
        graph.add_node(
            "conversation_fallback_node",
            self._dispatch_node("conversation_fallback_node"),
        )
        graph.add_node("conversation_node", self._conversation_node)
        graph.add_node("citation_validation_node", self._citation_validation_node)
        graph.add_node("response_node", self._response_node)

        graph.add_edge(START, "request_guard_node")
        graph.add_conditional_edges(
            "request_guard_node",
            self._after_guard,
            {
                "identity": "identity_validation_node",
                "response": "response_node",
            },
        )
        graph.add_conditional_edges(
            "identity_validation_node",
            self._after_identity,
            {
                "intent": "intent_router_node",
                "response": "response_node",
            },
        )
        graph.add_conditional_edges(
            "intent_router_node",
            self._route_intent,
            {
                "policy": "policy_retrieval_node",
                "hybrid": "hybrid_retrieval_node",
                "tool": "leave_tool_node",
                "clarification": "clarification_node",
                "confirmation": "confirmation_node",
                "fallback": "conversation_fallback_node",
            },
        )
        for node in (
            "policy_retrieval_node",
            "hybrid_retrieval_node",
            "leave_tool_node",
            "clarification_node",
            "confirmation_node",
            "conversation_fallback_node",
        ):
            graph.add_edge(node, "conversation_node")
        graph.add_edge("conversation_node", "citation_validation_node")
        graph.add_edge("citation_validation_node", "response_node")
        graph.add_edge("response_node", END)
        return graph.compile()

    def _request_guard_node(self, state: ChatWorkflowState) -> dict[str, object]:
        safety = assess_user_message(state["message"])
        state["trace"].record(
            "request_guard_node",
            "evaluated",
            {"decision": safety.route, "reason": safety.reason or "none"},
        )
        if safety.route == "refuse":
            state["session_store"].consume_pending(
                state["session_id"],
                state["identity"].employee_id,
            )
            return {
                "safety": safety,
                "workflow_stage": "guarded_refusal",
                "response": _response(
                    safety.message or "I can't help with that request.",
                    "refuse",
                ),
            }
        return {"safety": safety}

    def _identity_validation_node(
        self,
        state: ChatWorkflowState,
    ) -> dict[str, object]:
        try:
            identity = lookup_employee_identity(
                self._database_path,
                state["employee_id"],
            )
        except IdentityDirectoryError:
            raise
        expected = state["identity"]
        valid = (
            identity is not None
            and identity.employee_id == expected.employee_id
            and identity.region == expected.region
            and identity.role == expected.role
            and parse_role(identity.role) is not None
        )
        state["trace"].record(
            "identity_validation_node",
            "validated",
            {"decision": "allowed" if valid else "denied"},
        )
        if not valid:
            return {
                "identity_valid": False,
                "workflow_stage": "identity_denied",
                "response": _response(
                    "I can't verify the authenticated employee for this request.",
                    "refuse",
                ),
            }
        return {"identity_valid": True}

    def _intent_router_node(self, state: ChatWorkflowState) -> dict[str, object]:
        message = state["message"]
        intent = _tool_intent(message)
        plan: RequestPlan | None = None
        provider = state["llm_provider"]
        if provider is not None and not _is_request_id_only(message):
            plan = provider.plan_request(
                message,
                reference_date=state["today"],
                corpus_tokens=0,
            )
        planned_action: str | None = None
        planned_search_query: str | None = None
        if plan is not None:
            if plan.action in {
                "get_leave_balance",
                "list_leave_requests",
                "get_team_leave",
                "apply_leave",
                "cancel_leave",
            }:
                intent = plan.action
                planned_action = plan.action
            elif plan.action == "search_documents":
                if intent is None:
                    planned_action = "search_documents"
                else:
                    planned_action = intent
                planned_search_query = plan.search_query
            elif plan.action == "clarify":
                planned_action = intent or "clarify"
            elif intent is not None:
                planned_action = intent
            elif _needs_retrieval(message, intent):
                intent = None
                planned_action = "search_documents"
            else:
                intent = None
                planned_action = "refuse"

        needs_retrieval = (
            planned_action != "refuse"
            and (
                _needs_retrieval(message, intent)
                or planned_action == "search_documents"
            )
        )
        state["trace"].record(
            "intent_router_node",
            "routed",
            {
                "intent": intent or "none",
                "needs_retrieval": needs_retrieval,
                "action": planned_action or "none",
            },
        )
        result: dict[str, object] = {
            "intent": intent,
            "needs_retrieval": needs_retrieval,
            "planned_action": planned_action,
            "planned_search_query": planned_search_query,
        }
        if plan is not None:
            result["request_plan"] = plan
        return result

    @staticmethod
    def _after_guard(state: ChatWorkflowState) -> str:
        return "response" if state.get("response") is not None else "identity"

    @staticmethod
    def _after_identity(state: ChatWorkflowState) -> str:
        return "intent" if state.get("identity_valid") else "response"

    @staticmethod
    def _route_intent(state: ChatWorkflowState) -> str:
        message = state["message"]
        intent = state.get("intent")
        planned_action = state.get("planned_action")
        if _is_confirmation(message) or _is_cancellation(message):
            return "confirmation"
        if _is_ambiguous_leave_check(message):
            return "clarification"
        if planned_action == "clarify":
            return "clarification"
        if planned_action == "refuse":
            return "fallback"
        if intent is not None:
            return "hybrid" if state.get("needs_retrieval") else "tool"
        if state.get("needs_retrieval"):
            return "policy"
        if any(
            term in message.casefold()
            for term in ("which", "what date", "what type", "employee id")
        ):
            return "clarification"
        return "fallback"

    @staticmethod
    def _dispatch_node(node_name: str):
        def dispatch(state: ChatWorkflowState) -> dict[str, object]:
            state["trace"].record(
                node_name,
                "dispatched",
                {"intent": state.get("intent") or "none"},
            )
            return {"workflow_stage": node_name}

        return dispatch

    @staticmethod
    def _conversation_node(state: ChatWorkflowState) -> dict[str, object]:
        return {
            "response": _run_conversation(
                state,
                safety=state.get("safety"),
            )
        }

    @staticmethod
    def _citation_validation_node(
        state: ChatWorkflowState,
    ) -> dict[str, object]:
        response = state.get("response")
        if not isinstance(response, ChatResponse):
            raise RuntimeError("Citation validation requires a chat response.")
        allowed = {
            (detail["doc_id"], detail["locator"])
            for event in state["trace"].events
            if event["event"] == "evidence_selected"
            for detail in (
                event["details"].get("citations", [])
                if isinstance(event["details"], dict)
                else []
            )
            if isinstance(detail, dict)
        }
        cited = {(item.doc_id, item.locator) for item in response.citations}
        valid = cited.issubset(allowed)
        state["trace"].record(
            "citation_validation_node",
            "validated",
            {
                "decision": "passed" if valid else "failed",
                "citation_count": len(cited),
            },
        )
        if valid:
            return {}
        return {
            "response": _response(
                "I could not validate the document citations for this answer. "
                "Please contact the HR team.",
                "no_answer",
            )
        }

    @staticmethod
    def _response_node(state: ChatWorkflowState) -> dict[str, object]:
        response = state.get("response")
        if not isinstance(response, ChatResponse):
            raise RuntimeError("Response node requires a chat response.")
        state["trace"].record(
            "response_node",
            "created",
            {
                "route": response.route,
                "tool_count": len(response.tool_calls),
                "citation_count": len(response.citations),
                "pending_confirmation": response.pending_confirmation,
            },
        )
        state["trace"].set_request_id(response.request_id)
        return {"response": response}


def _run_conversation(
    state: ChatWorkflowState,
    *,
    safety: RequestSafetyDecision | None = None,
) -> ChatResponse:
    return respond_to_message(
        message=state["message"],
        session_id=state["session_id"],
        identity=state["identity"],
        retriever=state["retriever"],
        tools=state["tools"],
        session_store=state["session_store"],
        today=state["today"],
        llm_provider=state["llm_provider"],
        safety_decision=safety,
        trace=state["trace"],
        pii_redaction_enabled=state["pii_redaction_enabled"],
        pii_redaction_mode=state["pii_redaction_mode"],
        pii_sensitive_terms=state["pii_sensitive_terms"],
        request_plan=state.get("request_plan"),
    )
