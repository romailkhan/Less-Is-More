from typing import Dict, List

from langchain_core.prompts import ChatPromptTemplate
import json

from agents.openrouter_llm import get_openrouter_llm
from agents.token_usage import record_message

# Task types for multi-agent final output
TASK_COMMONGEN = "commongen"
TASK_MGSM = "mgsm"
TASK_LOGIC_GRID = "logic_grid"


class Language:
    def __init__(self):
        self.llm = get_openrouter_llm(role="language")

        self._system_templates = {
            TASK_MGSM: """
        You are a {topic} Language Specialist. Your role is to analyze language usage
        and communication effectiveness. Use the provided memories to help inform your analysis.

        Background: You are an expert at understanding language patterns, meaning, and communication
        effectiveness across different contexts.

        Your goal is to analyze language usage and meaning, then give the final numeric answer.

        Focus on:
        - Key meanings
        - Language patterns
        - Communication style
        - Overall effectiveness

        IMPORTANT: You must respond with ONLY valid JSON. Your entire response must be a single JSON object with exactly this structure:
        {{
            "role": "Language Specialist",
            "analysis": {{
                "semantic_interpretations": ["interpretation1", "interpretation2"],
                "stylistic_patterns": ["pattern1", "pattern2"],
                "final_response": "42"
            }}
        }}
        Do not include any other text, thoughts, or explanations. The response must be pure JSON only.

        For math word problems: put ONLY the final integer or decimal number in final_response (no units, no words).
        """,
            TASK_COMMONGEN: """
        You are a {topic} Language Specialist. Your role is to synthesize prior cognitive analyses into
        one coherent sentence that uses ALL given concepts from the user query.

        IMPORTANT: You must respond with ONLY valid JSON. Your entire response must be a single JSON object with exactly this structure:
        {{
            "role": "Language Specialist",
            "analysis": {{
                "semantic_interpretations": ["interpretation1", "interpretation2"],
                "stylistic_patterns": ["pattern1", "pattern2"],
                "final_response": "Your single coherent sentence here."
            }}
        }}
        The final_response must be one grammatically correct sentence that incorporates every concept from the task.
        Do not include any other text outside the JSON object.
        """,
            TASK_LOGIC_GRID: """
        You are a {topic} Language Specialist. Your role is to produce the final answer to a logic grid puzzle
        based on prior analyses.

        IMPORTANT: You must respond with ONLY valid JSON. Your entire response must be a single JSON object with exactly this structure:
        {{
            "role": "Language Specialist",
            "analysis": {{
                "semantic_interpretations": ["interpretation1", "interpretation2"],
                "stylistic_patterns": ["pattern1", "pattern2"],
                "final_response": "3"
            }}
        }}
        Put ONLY the correct choice in final_response: the number or label that answers the question (e.g. house number).
        No explanation in final_response.
        Do not include any other text outside the JSON object.
        """,
        }

        self._chains = {}
        for task, tmpl in self._system_templates.items():
            prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", tmpl),
                    ("user", "{input}"),
                ]
            )
            self._chains[task] = prompt | self.llm

        # Default chain for backward compatibility
        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self._system_templates[TASK_MGSM]),
                ("user", "{input}"),
            ]
        )
        self.chain = self.prompt | self.llm

    def analyze(
        self,
        input_text: str,
        memories: List[Dict],
        topic: str = "General",
        task_type: str = TASK_MGSM,
    ) -> Dict:
        """
        Analyze the language usage in the input text, informed by memories.

        task_type: one of commongen | mgsm | logic_grid
        """
        chain = self._chains.get(task_type, self._chains[TASK_MGSM])
        response = chain.invoke(
            {
                "input": input_text,
                "memories": memories,
                "topic": topic,
            }
        )
        record_message(response)

        response_text = str(response.content)

        try:
            start_idx = response_text.find("{")
            end_idx = response_text.rfind("}") + 1
            json_str = response_text[start_idx:end_idx]

            json_str = json_str.strip()
            json_str = " ".join(line.strip() for line in json_str.splitlines())

            result = json.loads(json_str)
            if "role" not in result or "analysis" not in result:
                raise ValueError("Missing required top-level fields")

            required_analysis_fields = ["semantic_interpretations", "stylistic_patterns", "final_response"]
            if not all(field in result["analysis"] for field in required_analysis_fields):
                raise ValueError("Missing required analysis fields")

            return result

        except (json.JSONDecodeError, ValueError) as e:
            print(f"Error parsing response: {e}")
            print(f"Attempted to parse: {json_str}")
            return {
                "role": "Language Specialist",
                "analysis": {
                    "semantic_interpretations": ["Error in analysis"],
                    "stylistic_patterns": ["Unable to parse response"],
                    "final_response": "",
                },
            }
