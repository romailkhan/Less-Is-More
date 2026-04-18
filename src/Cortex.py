from typing import Any, Dict, List
import json
from datetime import datetime
from agents.Perception import Perception
from agents.Emotion import Emotion
from agents.Reasoning import Reasoning
from agents.Language import Language
from agents.Feedback import Feedback
from agents.Language import TASK_MGSM
from uuid import uuid4
import os
import chromadb

from agents.chroma_embedding import get_chroma_embedding_function
from agents.token_usage import usage_run_begin, usage_run_end


class Cortex:
    def __init__(self):
        self.perception_agent = Perception()
        self.emotion_agent = Emotion()
        self.reasoning_agent = Reasoning()
        self.language_agent = Language()
        self.feedback_agent = Feedback()

        chroma_path = "./long_term_memory_store"
        self.chroma_client = chromadb.PersistentClient(path=chroma_path)

        self.embedding_function = get_chroma_embedding_function()

        self.collections = {}
        agent_names = ["perception", "emotion", "reasoning", "language", "feedback"]
        for agent_name in agent_names:
            try:
                self.collections[agent_name] = self.chroma_client.get_or_create_collection(
                    name=f"{agent_name}_memories",
                    embedding_function=self.embedding_function,
                )
            except Exception as e:
                print(f"Error creating/getting collection for {agent_name}: {e}")

        self.reset_state()

    def reset_state(self) -> None:
        self.current_state: Dict[str, Any] = {
            "initial_query": None,
            "agents": {},
            "task_type": TASK_MGSM,
            "refinement_history": [],
        }

    def _context_for_llm(self) -> str:
        """JSON context for reasoning, language, and feedback (includes refinement history)."""
        return json.dumps(
            {
                "initial_query": self.current_state.get("initial_query"),
                "agents": self.current_state.get("agents", {}),
                "refinement_history": self.current_state.get("refinement_history", []),
            },
            default=str,
        )

    def _query_chroma(self, collection_name: str, query_text: str, n_results: int = 3) -> List[Dict]:
        """Helper function to query a ChromaDB collection."""
        try:
            results = self.collections[collection_name].query(
                query_texts=[query_text],
                n_results=n_results,
            )
            if results and results.get("documents") and results["documents"][0]:
                memories = [json.loads(doc) for doc in results["documents"][0]]
                return memories
            else:
                return []
        except Exception as e:
            print(f"Error querying ChromaDB collection {collection_name}: {e}")
            return []

    def _add_to_chroma(self, collection_name: str, document: Dict, doc_id: str):
        """Helper function to add a document to a ChromaDB collection."""
        try:
            doc_string = json.dumps(document)
            self.collections[collection_name].add(
                documents=[doc_string],
                ids=[doc_id],
            )
        except Exception as e:
            # Non-fatal: pipeline still works without long-term memory (e.g. bad embedding API key)
            print(f"Warning: could not add to ChromaDB collection {collection_name}: {e}")

    def perception(self, query: str) -> Dict:
        self.current_state["initial_query"] = query
        agent_name = "perception"

        relevant_memories = self._query_chroma(agent_name, query)

        perception_output = self.perception_agent.analyze(query, relevant_memories)
        self.current_state["agents"]["perception"] = perception_output

        doc_id = f"{agent_name}_{datetime.now().isoformat()}_{uuid4()}"
        self._add_to_chroma(agent_name, perception_output, doc_id)

        return self.current_state

    def emotion(self, state: Dict) -> Dict:
        agent_name = "emotion"
        context_query = json.dumps(state["agents"].get("perception", state["initial_query"]))

        relevant_memories = self._query_chroma(agent_name, context_query)

        emotion_output = self.emotion_agent.analyze(context_query, relevant_memories)
        state["agents"]["emotion"] = emotion_output

        doc_id = f"{agent_name}_{datetime.now().isoformat()}_{uuid4()}"
        self._add_to_chroma(agent_name, emotion_output, doc_id)

        return state

    def reasoning(self, state: Dict) -> Dict:
        agent_name = "reasoning"
        context_query = self._context_for_llm()

        relevant_memories = self._query_chroma(agent_name, context_query)

        reasoning_output = self.reasoning_agent.analyze(context_query, relevant_memories)
        state["agents"]["reasoning"] = reasoning_output

        doc_id = f"{agent_name}_{datetime.now().isoformat()}_{uuid4()}"
        self._add_to_chroma(agent_name, reasoning_output, doc_id)

        return state

    def language(self, state: Dict) -> Dict:
        agent_name = "language"
        context_query = self._context_for_llm()

        relevant_memories = self._query_chroma(agent_name, context_query)

        task_type = self.current_state.get("task_type", TASK_MGSM)
        language_output = self.language_agent.analyze(
            context_query, relevant_memories, topic="General", task_type=task_type
        )
        state["agents"]["language"] = language_output

        doc_id = f"{agent_name}_{datetime.now().isoformat()}_{uuid4()}"
        self._add_to_chroma(agent_name, language_output, doc_id)

        return state

    def feedback(self, state: Dict) -> Dict:
        agent_name = "feedback"
        context_query = self._context_for_llm()

        relevant_memories = self._query_chroma(agent_name, context_query)

        feedback_output = self.feedback_agent.analyze(context_query, relevant_memories)
        state["agents"]["feedback"] = feedback_output

        return state

    def process_query(
        self,
        query: str,
        topic: str = "General",
        task_type: str = TASK_MGSM,
        enable_refinement: bool = False,
        max_refinement_iterations: int = 3,
    ) -> Dict:
        """
        Run the cognitive pipeline. Optionally loop reasoning→language→feedback when quality is low.

        task_type: commongen | mgsm | logic_grid (must match agents.Language / benchmarks).
        """
        self.reset_state()
        self.current_state["task_type"] = task_type
        self.current_state["topic"] = topic

        usage_run_begin()
        try:
            state = self.perception(query)
            state = self.emotion(state)
            state = self.reasoning(state)
            state = self.language(state)
            state = self.feedback(state)

            refinement_round = 0
            while enable_refinement and refinement_round < max_refinement_iterations:
                fb = state["agents"].get("feedback", {})
                if fb.get("decision") != "Refinement Needed":
                    break

                refinement_round += 1
                self.current_state.setdefault("refinement_history", []).append(
                    {
                        "round": refinement_round,
                        "feedback": fb,
                    }
                )

                state = self.reasoning(state)
                state = self.language(state)
                state = self.feedback(state)

            self.current_state["refinement_rounds_executed"] = refinement_round
        finally:
            self.current_state["token_usage"] = usage_run_end()

        self.save_state("cortex_output.json")
        return state

    def save_state(self, filename: str):
        with open(filename, "w") as f:
            json.dump(self.current_state, f, indent=2)
