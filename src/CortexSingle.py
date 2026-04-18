from typing import Dict, List
import json
from datetime import datetime
from agents.Reasoning_benchmark import (
    Reasoning_Benchmark,
    TASK_COMMONGEN,
    TASK_LOGIC_GRID,
    TASK_MGSM,
)
from uuid import uuid4
import os
import chromadb

from agents.chroma_embedding import get_chroma_embedding_function
from agents.token_usage import usage_run_begin, usage_run_end


class CortexSingle:
    def __init__(self, reasoning_role: str = "single"):
        self.reasoning_agent = Reasoning_Benchmark(llm_role=reasoning_role)
        
        chroma_path = "./long_term_memory_store_single_agent"
        self.chroma_client = chromadb.PersistentClient(path=chroma_path)
        
        self.embedding_function = get_chroma_embedding_function()
        
        self.agent_name = "reasoning"
        try:
            self.collection = self.chroma_client.get_or_create_collection(
                name=f"{self.agent_name}_memories",
                embedding_function=self.embedding_function
            )
        except Exception as e:
            print(f"Error creating/getting collection for {self.agent_name}: {e}")
            self.collection = None 
        
        self.reset_state()

    def reset_state(self):
        self.current_state = {
            "initial_query": None,
            "agents": {} 
        }

    def _query_chroma(self, query_text: str, n_results: int = 3) -> List[Dict]:
        """Helper function to query the ChromaDB collection."""
        if not self.collection:
            print("ChromaDB collection not initialized.")
            return []
        try:
            results = self.collection.query(
                query_texts=[query_text],
                n_results=n_results,
            )
            if results and results.get('documents') and results['documents'][0]:
                # Filter out potential None values before parsing
                valid_docs = [doc for doc in results['documents'][0] if doc is not None]
                memories = [json.loads(doc) for doc in valid_docs]
                return memories
            else:
                return []
        except Exception as e:
            print(f"Error querying ChromaDB collection {self.agent_name}_memories: {e}")
            return []

    def _add_to_chroma(self, document: Dict, doc_id: str):
        """Helper function to add a document to the ChromaDB collection."""
        if not self.collection:
            print("ChromaDB collection not initialized. Cannot add document.")
            return
        try:
            doc_string = json.dumps(document)
            self.collection.add(
                documents=[doc_string],
                ids=[doc_id]
            )
        except Exception as e:
            print(f"Warning: could not add to ChromaDB collection {self.agent_name}_memories: {e}")

    def process_query(
        self,
        query: str,
        topic: str = "General",
        task_type: str = TASK_COMMONGEN,
        refinement_context: str | None = None,
    ) -> Dict:
        """Processes a query using only the Reasoning agent."""
        self.reset_state()
        self.current_state["initial_query"] = query
        self.current_state["task_type"] = task_type

        # Prepare context for the reasoning agent - just the initial query for now
        context_query = query

        # Query relevant memories
        relevant_memories = self._query_chroma(context_query)

        usage_run_begin()
        try:
            reasoning_output = self.reasoning_agent.analyze(
                input_text=context_query,
                memories=relevant_memories,
                topic=topic,
                task_type=task_type,
                refinement_context=refinement_context,
            )
            self.current_state["agents"][self.agent_name] = reasoning_output
        finally:
            self.current_state["token_usage"] = usage_run_end()

        doc_id = f"{self.agent_name}_{datetime.now().isoformat()}_{uuid4()}"
        self._add_to_chroma(reasoning_output, doc_id)

        self.save_state("cortex_single_output.json") # Save to a different file
        return self.current_state

    def save_state(self, filename: str):
        """Saves the current state to a JSON file."""
        try:
            with open(filename, 'w') as f:
                json.dump(self.current_state, f, indent=2)
        except IOError as e:
            print(f"Error saving state to {filename}: {e}")
