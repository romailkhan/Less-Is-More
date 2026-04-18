import json
import os
import re
from typing import Any, Dict, List, Optional

from langchain_core.prompts import ChatPromptTemplate

from agents.openrouter_llm import get_openrouter_llm
from agents.token_usage import record_message

TASK_COMMONGEN = "commongen"
TASK_MGSM = "mgsm"
TASK_LOGIC_GRID = "logic_grid"


_COMMONGEN_SYSTEM = """
You are a {topic} creative sentence generator. Your role is to generate a single, coherent, and grammatically correct sentence that incorporates all the given concepts.

Background: You excel at understanding concepts and weaving them together naturally into a sentence.

Your goal is to generate a single sentence using all the concepts provided in the input. Do not provide alternate sentences.

You will be given a list of concepts. You must generate a single sentence using all the concepts.

Respond with ONLY valid JSON with exactly this structure:
{{
    "final_answer": "The generated sentence using all concepts."
}}
"""

_MGSM_SYSTEM = """
You are a {topic} expert at solving mathematical word problems.

Solve the problem step by step internally, then output ONLY valid JSON:
{{
    "final_answer": 42
}}
where final_answer is the numeric result (integer or float) with no units or words in the JSON value.
If the answer must be an integer, use an integer.
"""

_LOGIC_GRID_SYSTEM = """
You are a {topic} expert at logic grid puzzles and deductive reasoning.

Read all clues carefully. Deduce the unique solution. The question asks for a specific choice (often a house number).

Respond with ONLY valid JSON:
{{
    "final_answer": "3"
}}
where final_answer is the answer string exactly as required (e.g. house number as digits only, matching one of the given choices).
"""


class Reasoning_Benchmark:
    """Single-agent reasoning with task-specific prompts (CommonGen, MGSM, Logic Grid)."""

    def __init__(self, llm_role: str = "single") -> None:
        temperature = float(os.getenv("TEMPERATURE", 0.6))
        max_tokens = int(os.getenv("MAX_TOKENS", 8192))

        self.llm = get_openrouter_llm(
            role=llm_role, temperature=temperature, max_tokens=max_tokens
        )

        self._chains = {
            TASK_COMMONGEN: self._make_chain(_COMMONGEN_SYSTEM),
            TASK_MGSM: self._make_chain(_MGSM_SYSTEM),
            TASK_LOGIC_GRID: self._make_chain(_LOGIC_GRID_SYSTEM),
        }

    def _make_chain(self, system_template: str):
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_template),
                ("user", "{input}"),
            ]
        )
        return prompt | self.llm

    def analyze(
        self,
        input_text: str,
        memories: List[Dict],
        topic: str = "Expert Reasoning Specialist",
        task_type: str = TASK_COMMONGEN,
        refinement_context: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Run single-agent reasoning for the given benchmark task.

        refinement_context: optional feedback from a prior attempt (for refinement loops).
        """
        chain = self._chains.get(task_type, self._chains[TASK_COMMONGEN])
        user_parts = [input_text]
        if refinement_context:
            user_parts.append(
                "\n\nPrior feedback to address (improve your answer):\n" + refinement_context
            )
        full_input = "\n".join(user_parts)

        response = chain.invoke(
            {
                "input": full_input,
                "memories": memories,
                "topic": topic,
            }
        )
        record_message(response)

        response_text = str(response.content)
        return self._parse_final_answer(response_text)

    @staticmethod
    def _parse_final_answer(response_text: str) -> Dict[str, Any]:
        """Extract final_answer from model output (JSON or regex fallbacks)."""
        text = response_text.strip()
        try:
            start_idx = text.find("{")
            end_idx = text.rfind("}") + 1
            if start_idx >= 0 and end_idx > start_idx:
                json_str = text[start_idx:end_idx]
                json_str = " ".join(line.strip() for line in json_str.splitlines())
                obj = json.loads(json_str)
                if isinstance(obj, dict) and "final_answer" in obj:
                    return {"final_answer": obj["final_answer"]}
        except (json.JSONDecodeError, TypeError):
            pass

        # Quoted string
        m = re.search(r'"final_answer"\s*:\s*"(.*?)"', response_text, re.DOTALL)
        if m:
            return {"final_answer": m.group(1).strip()}

        # Unquoted number
        m = re.search(r'"final_answer"\s*:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)', response_text)
        if m:
            val = m.group(1)
            try:
                if "." in val or "e" in val.lower():
                    return {"final_answer": float(val)}
                return {"final_answer": int(val)}
            except ValueError:
                return {"final_answer": val}

        return {
            "final_answer": f"Error: Parsing failed. Response: {response_text[:800]}",
        }
