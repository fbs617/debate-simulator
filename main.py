# moderator_app.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Literal

from dotenv import load_dotenv
load_dotenv()  # loads OPENAI_API_KEY from .env

from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END

# -----------------------------
# 1) STATE (shared memory)
# -----------------------------
@dataclass
class DebateCitation:
    title: str
    url: str
    snippet: Optional[str] = None

@dataclass
class DebateMessage:
    content: str
    key_points: List[str] =  field(default_factory=list)
    citations: List[DebateCitation] = field(default_factory=list)

@dataclass
class JudgeMessage:
    decision: Literal["pro", "con", "tie"]
    rationale: str
    pro_feedback: str
    con_feedback: str
    key_points: List[str] =  field(default_factory=list)

@dataclass
class DebateHistory:
    role: Literal["pro", "con", "judge"]
    phase: Literal["opening", "rebuttal", "closing", "decision"]
    output: DebateMessage

@dataclass
class DebateState:
    topic: str
    status: str = "init"                       # "init" | "rejected" | "ready"
    safety_flags: List[Dict] = field(default_factory=list)  # audit trail
    message: Optional[str] = None              # explanation / next-step text
    history: List[DebateHistory] = field(default_factory=list)  # past messages


# -----------------------------
# 2) SCHEMA for structured output
# -----------------------------
Action = Literal["allow", "rephrase", "reject"]

class ModerationResult(BaseModel):
    action: Action
    categories: List[str] = Field(default_factory=list)     # e.g., ["illegal_activity", "hate"]
    explanation: str                                        # short human-safe rationale
    safe_topic: Optional[str] = None                        # required if action == "rephrase"
    notes: List[str] = Field(default_factory=list)          # optional extra guidance

# Use a Pydantic model for structured LLM output, then map to dataclass
class DebateCitationModel(BaseModel):
    title: str
    url: str
    snippet: Optional[str] = None

class DebateMessageModel(BaseModel):
    content: str
    key_points: List[str] = Field(default_factory=list)
    citations: List[DebateCitationModel] = Field(default_factory=list)

class JudgeMessageModel(BaseModel):
    decision: Literal["pro", "con", "tie"]
    rationale: str
    pro_feedback: str
    con_feedback: str
    key_points: List[str] =  field(default_factory=list)


# -----------------------------
# 3) LLM setup
# -----------------------------
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# -----------------------------
# 4) NODES
# -----------------------------
def moderator_node(state: DebateState) -> DebateState:
    """Classify a topic as allow / rephrase / reject; optionally rephrase it."""
    # 4a) quick keyword short-circuit (fast, local)
    quick_block = [
        "build a bomb", "make a bomb", "child sexual", "sexual content with minors",
        "self-harm instructions", "how to hack", "credit card skimming",
        "assassinate", "terrorist", "make a gun", "kill"
    ]
    topic_l = (state.topic or "").strip().lower()
    if any(k in topic_l for k in quick_block):
        res = ModerationResult(
            action="reject",
            categories=["illegal_or_extreme_harm"],
            explanation="Disallowed: violent/illegal instruction content.",
            safe_topic=None,
            notes=["Reframe as an ethics/policy discussion rather than instructions."]
        )
        state.safety_flags.append(res.model_dump())
        return state

    # 4b) LLM moderation + academic reframe
    system = (
        "You are a safety moderator for a debate platform. "
        "Decide one of: allow | rephrase | reject. "
        "Disallow topics with step-by-step illegal harm, sexual content with minors, "
        "self-harm instructions, targeted hate/violence, invasive privacy breaches, malware creation, etc. "
        "Also don't allow topics that have any relation to religion or politics, the topics should all be "
        "leaning more towards being lighthearted and less controversial. "
        "If rephrase: produce a neutral, academic framing (policy/ethics/history) that preserves learning value. "
        "Also ensure that the rephrase is detailed and provides two clear sides to argue about. "
        "Return ONLY structured JSON matching the provided schema."
    )
    user = f'Topic: "{state.topic}"'

    mod_llm = llm.with_structured_output(schema=ModerationResult)
    result: ModerationResult = mod_llm.invoke(f"{system}\n\n{user}")

    # audit trail
    state.safety_flags.append(result.model_dump())

    # apply rephrase if given
    if result.action == "rephrase" and result.safe_topic:
        state.topic = result.safe_topic

    return state

def moderator_router(state: DebateState) -> str:
    """Decide which node to run after moderator."""
    last = state.safety_flags[-1] if state.safety_flags else {}
    action = last.get("action")
    if action == "reject":
        return "reject_exit"
    return "proceed"  # allow OR rephrase both continue

def reject_exit(state: DebateState) -> DebateState:
    state.status = "rejected"
    last = state.safety_flags[-1] if state.safety_flags else {}
    state.message = f"❌ Rejected. Reason: {last.get('explanation','')}"
    return state

def proceed_node(state: DebateState) -> DebateState:
    state.status = "ready"
    last = state.safety_flags[-1] if state.safety_flags else {}
    if last.get("action") == "rephrase":
        state.message = f"✅ Rephrased to safe topic: “{state.topic}”. Ready to start the debate flow."
    else:
        state.message = f"✅ Allowed topic: “{state.topic}”. Ready to start the debate flow."
    return state


# ---------- PRO / CON NODES ----------
def pro_node(state: DebateState) -> DebateState:
    """Generate Pro opening or rebuttal, then store into history."""
    system = (
        "You are the 'Pro' debater in a formal debate. "
        "Support the topic with clear reasoning and evidence. "
        "Be concise and persuasive."
    )
    mod_llm = llm.with_structured_output(schema=DebateMessageModel)

    if not state.history:
        # Pro Opening
        state.message = f"Current phase is Pro Opening argument for topic: {state.topic}."
        result: DebateMessageModel = mod_llm.invoke(f"{system}\n\n{state.message}\n\n")
        msg = DebateMessage(
            content=result.content,
            key_points=result.key_points,
            citations=[DebateCitation(**c.model_dump()) for c in result.citations]
        )
        state.history.append(DebateHistory(role="pro", phase="opening", output=msg))
    else:
        # Pro Rebuttal (respond to last turn)
        prior = state.history[-1].output.content
        state.message = f"Current phase is Pro Rebuttal argument for topic: {state.topic}."
        result: DebateMessageModel = mod_llm.invoke(
            f"{system}\n\n{state.message}\n\n"
            f"Consider the previous opposing argument in your rebuttal:\n{prior}\n"
        )
        msg = DebateMessage(
            content=result.content,
            key_points=result.key_points,
            citations=[DebateCitation(**c.model_dump()) for c in result.citations]
        )
        state.history.append(DebateHistory(role="pro", phase="rebuttal", output=msg))

    return state

def con_node(state: DebateState) -> DebateState:
    """Generate Con rebuttal or closing, then store into history."""
    system = (
        "You are the 'Con' debater in a formal debate. "
        "Oppose the topic with clear reasoning and evidence. "
        "Be concise and persuasive."
    )
    mod_llm = llm.with_structured_output(schema=DebateMessageModel)

    # Safety: if somehow called first, do a rebuttal vs nothing
    prior_text = state.history[-1].output.content if state.history else ""

    # If the last phase was Pro opening -> Con rebuttal, else Con closing
    if state.history and state.history[-1].phase == "opening":
        state.message = f"Current phase is Con Rebuttal argument for topic: {state.topic}."
        result: DebateMessageModel = mod_llm.invoke(
            f"{system}\n\n{state.message}\n\n"
            f"Consider the previous Pro argument in your rebuttal:\n{prior_text}\n"
        )
        phase = "rebuttal"
    else:
        state.message = f"Current phase is Con Closing argument for topic: {state.topic}."
        result: DebateMessageModel = mod_llm.invoke(
            f"{system}\n\n{state.message}\n\n"
            f"Address the previous argument in your closing:\n{prior_text}\n"
        )
        phase = "closing"

    msg = DebateMessage(
        content=result.content,
        key_points=result.key_points,
        citations=[DebateCitation(**c.model_dump()) for c in result.citations]
    )
    state.history.append(DebateHistory(role="con", phase=phase, output=msg))
    return state

def judge_node(state: DebateState) -> DebateState:
    """Generate Con rebuttal or closing, then store into history."""
    system = """
        You are the Judge in a formal debate. 
        Your task is to carefully evaluate the arguments made by the Pro and Con debaters, 
        then deliver a fair, structured judgment.

        Follow these instructions:

        1. **Decision**
        - Choose exactly one: "pro", "con", or "tie".
        - "pro" if the Pro side was overall stronger.
        - "con" if the Con side was overall stronger.
        - "tie" only if both sides were balanced and neither clearly won.

        2. **Rationale**
        - Provide a concise but clear explanation (3-6 sentences) 
            summarizing why you reached your decision.
        - Reference the strength of reasoning, evidence, and persuasiveness.

        3. **Pro Feedback**
        - Give constructive feedback directly to the Pro debater.
        - Mention what they did well and how they could improve.

        4. **Con Feedback**
        - Give constructive feedback directly to the Con debater.
        - Mention what they did well and how they could improve.

        5. **Key Points**
        - Extract 3-5 bullet-point key reasons or themes that drove your decision.
        - Keep them short and impactful.

        Return your answer as structured JSON that matches this schema:

        {
        "decision": "pro" | "con" | "tie",
        "rationale": "...",
        "pro_feedback": "...",
        "con_feedback": "...",
        "key_points": ["...", "..."]
        }
    """
    mod_llm = llm.with_structured_output(schema=JudgeMessageModel)
    state.message = f"Current phase is Judging phase for topic: {state.topic}."
    prior_text = []
    for h in state.history:
        prior_text.append(f"{h.role.upper()} ({h.phase}): {h.output.content}")
    result: JudgeMessageModel = mod_llm.invoke(
            f"{system}\n\n{state.message}\n\n"
            f"Consider the previous Pro and Con arguments in your evaluation:\n{prior_text}\n"
        )
    
    phase = "evaluation"

    msg = JudgeMessage(
        decision=result.decision,
        rationale=result.rationale,
        pro_feedback=result.pro_feedback,
        con_feedback=result.con_feedback,
        key_points=result.key_points
    )
    state.history.append(DebateHistory(role="judge", phase=phase, output=msg))
    return state


# ---------- FLOW ROUTER ----------
def debate_flow_router(state: DebateState) -> str:
    """
    Turn order:
      - (no history) -> pro (opening)
      - after pro opening -> con (rebuttal)
      - after con rebuttal -> pro (rebuttal)
      - after pro rebuttal -> con (closing)
      - after con closing -> judge
      - after judge -> END
    """
    if not state.history:
        return "pro"

    last = state.history[-1]
    if last.role == "pro" and last.phase in ("opening", "rebuttal"):
        return "con"

    if last.role == "con":
        if last.phase == "rebuttal":
            return "pro"
        if last.phase == "closing":
            return "judge"
    
    if last.role == "judge":
        return "__end__"

    # Fallback
    return "__end__"


# -----------------------------
# 5) GRAPH WIRING
# -----------------------------
graph = StateGraph(DebateState)

# Existing nodes
graph.add_node("moderator", moderator_node)
graph.add_node("reject_exit", reject_exit)
graph.add_node("proceed", proceed_node)

# New debate nodes
graph.add_node("pro", pro_node)
graph.add_node("con", con_node)
graph.add_node("judge", judge_node)

graph.set_entry_point("moderator")

# Moderator decides whether to proceed or reject
graph.add_conditional_edges(
    "moderator",
    moderator_router,
    {"reject_exit": "reject_exit", "proceed": "proceed"}
)

graph.add_edge("reject_exit", END)

# Enter debate flow after proceed
graph.add_conditional_edges(
    "proceed",
    debate_flow_router,
    {"pro": "pro", "con": "con", "__end__": END}
)

# Keep routing between pro/con until closing -> END
graph.add_conditional_edges(
    "pro",
    debate_flow_router,
    {"pro": "pro", "con": "con", "__end__": END}
)
graph.add_conditional_edges(
    "con",
    debate_flow_router,
    {"pro": "pro", "con": "con", "judge": "judge", "__end__": END}
)
graph.add_conditional_edges(
    "judge",
    debate_flow_router,
    {"__end__": END}
)

app = graph.compile()

# -----------------------------
# 6) CLI entrypoint
# -----------------------------
if __name__ == "__main__":
    try:
        topic = input("Enter a debate topic: ")
    except KeyboardInterrupt:
        raise SystemExit

    initial = DebateState(topic=topic)
    final_state = app.invoke(initial)  # LangGraph returns a dict-like state

    # Defensive: ensure dict for easy access
    state = final_state if isinstance(final_state, dict) else final_state.__dict__

    # -------- Moderator result (do NOT use state['message'] here) --------
    print("\n--- Moderator Result ---")
    safety_flags = state.get("safety_flags", [])
    mod = safety_flags[0] if safety_flags else {}  # your flow adds one moderation result

    action = mod.get("action")
    explanation = mod.get("explanation", "")
    categories = mod.get("categories", [])
    notes = mod.get("notes", [])
    resolved_topic = state.get("topic")

    if action == "reject":
        print("Status: rejected")
        print("Reason:", explanation)
        if categories:
            print("Categories:", ", ".join(categories))
        # No debate will follow
    elif action == "rephrase":
        print("Status: ready (rephrased)")
        print(f"Safe Topic: {resolved_topic}")
        print("Why rephrased:", explanation)
        if categories:
            print("Categories:", ", ".join(categories))
        if notes:
            print("Notes:", "; ".join(notes))
    else:
        print("Status: ready (allowed)")
        print(f"Allowed Topic: {resolved_topic}")
        if explanation:
            print("Moderator note:", explanation)
        if categories:
            print("Categories:", ", ".join(categories))

    # -------- Full debate flow from history --------
    print("\n--- Debate Flow ---")
    for h in state.get("history", []):
        if h.phase == "evaluation":
            # Special formatting for judge decision
            output: JudgeMessage = h.output
            print(f"\n[JUDGE DECISION]")
            print(f"Decision: {output.decision.upper()}")
            print("Rationale:", output.rationale)
            print("Pro Feedback:", output.pro_feedback)
            print("Con Feedback:", output.con_feedback)
            if output.key_points:
                print("Key Points:", ", ".join(output.key_points))
            continue
        role = h.role
        phase = h.phase
        output = h.output
        print(f"\n[{role.upper()} - {phase.capitalize()}]")
        print(output.content)
        if output.key_points:
            print("Key Points:", ", ".join(output.key_points))
        if output.citations:
            print("Citations:")
            for c in output.citations:
                snippet = f" - {c.snippet}" if c.snippet else ""
                print(f" • {c.title} ({c.url}){snippet}")
    
