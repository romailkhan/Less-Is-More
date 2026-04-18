from typing import Dict, List
from langchain_core.prompts import ChatPromptTemplate
import json

from agents.openrouter_llm import get_openrouter_llm
from agents.token_usage import record_message


class Feedback:
    def __init__(self):
        self.llm = get_openrouter_llm(role="feedback")
        
        self.system_template = """
        You are a {topic} Feedback Specialist. Your role is to evaluate all analyses 
        and determine if refinement is needed. Use the provided memories to help inform your analysis.

        Background: You are skilled at evaluating quality, identifying gaps, and determining when
        work needs improvement. Your focus is on ensuring comprehensive and accurate results.

        Your goal is to review analyses and determine if improvements are needed.

        Focus on:
        - Quality of analyses (provide a score between 0.0 and 1.0)
        - Missing elements
        - Areas for improvement
        - Detailed feedback

        The quality score should reflect:
        - Comprehensiveness (0.0-0.25)
        - Accuracy (0.0-0.25)
        - Clarity (0.0-0.25)
        - Coherence (0.0-0.25)

        IMPORTANT: You must respond with ONLY valid JSON. Your entire response must be a single JSON object with exactly this structure:
        {{
            "role": "Feedback Specialist",
            "analysis": {{
                "quality_score": 0.85,  # Must be between 0.0 and 1.0. Shows the quality and confidence of the analysis.
                "gaps_identified": ["gap1", "gap2"],
                "improvement_areas": ["area1", "area2"]
            }},
            "feedback": ["feedback1", "feedback2"]
        }}
        Do not include any other text, thoughts, or explanations. The response must be pure JSON only.
        """

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_template),
            ("user", "{input}")
        ])

        self.chain = self.prompt | self.llm

    def analyze(self, input_text: str, memories: List[Dict], topic: str = "General") -> Dict:
        """
        Evaluate the analyses and provide feedback, informed by memories.
        
        Args:
            input_text (str): The text to analyze
            memories (List[Dict]): Relevant past memories.
            topic (str): The specific topic area for analysis (defaults to "General")
            
        Returns:
            Dict: Analysis results in the specified JSON format
        """
        response = self.chain.invoke({
            "input": input_text,
            "memories": memories,
            "topic": topic
        })
        record_message(response)

        response_text = str(response.content)
        
        try:
            start_idx = response_text.find("{")
            end_idx = response_text.rfind("}") + 1
            json_str = response_text[start_idx:end_idx]
            
            json_str = json_str.strip()
            json_str = ' '.join(line.strip() for line in json_str.splitlines())
            
            result = json.loads(json_str)
            if "role" not in result or "analysis" not in result or "feedback" not in result:
                raise ValueError("Missing required top-level fields")
                
            required_analysis_fields = ["quality_score", "gaps_identified", "improvement_areas"]
            if not all(field in result["analysis"] for field in required_analysis_fields):
                raise ValueError("Missing required analysis fields")
            
            # Automatically set decision based on quality score
            quality_score = float(result["analysis"]["quality_score"])
            result["decision"] = "No Refinement Needed" if quality_score >= 0.70 else "Refinement Needed"
                
            return result
            
        except (json.JSONDecodeError, ValueError) as e:
            print(f"Error parsing response: {e}")
            print(f"Attempted to parse: {json_str}")
            return {
                "role": "Feedback Specialist",
                "analysis": {
                    "quality_score": 0.0,
                    "gaps_identified": ["Error in analysis"],
                    "improvement_areas": ["Unable to parse response"]
                },
                "decision": "Refinement Needed",
                "feedback": ["Please try again"]
            } 