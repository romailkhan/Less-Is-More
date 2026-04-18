from typing import Dict, List
from langchain_core.prompts import ChatPromptTemplate
import json

from agents.openrouter_llm import get_openrouter_llm
from agents.token_usage import record_message


class Emotion:
    def __init__(self):
        self.llm = get_openrouter_llm(role="emotion")
        
        self.system_template = """
        You are a {topic} Emotional Intelligence Specialist. Your role is to analyze emotional content 
        and patterns in the input. Use the provided memories to help inform your analysis.

        Background: You are skilled at detecting emotions, understanding emotional context, and identifying
        emotional patterns in communication.

        Your goal is to analyze emotional content in the input.

        Focus on:
        - Primary emotions
        - Emotional undertones
        - Mood patterns
        - Emotional context

        IMPORTANT: You must respond with ONLY valid JSON. Your entire response must be a single JSON object with exactly this structure:
        {{
            "role": "Emotional Intelligence Specialist",
            "analysis": {{
                "primary_emotions": ["emotion1", "emotion2"],
                "emotional_patterns": ["pattern1", "pattern2"],
                "recommendations": ["recommendation1", "recommendation2"]
            }}
        }}
        Do not include any other text, thoughts, or explanations. The response must be pure JSON only.
        Only focus on the query to analyze emotional content.
        """

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_template),
            ("user", "{input}")
        ])

        self.chain = self.prompt | self.llm

    def analyze(self, input_text: str, memories: List[Dict], topic: str = "General") -> Dict:
        """
        Analyze the emotional content in the input text, informed by memories.
        
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
            if "role" not in result or "analysis" not in result:
                raise ValueError("Missing required top-level fields")
                
            required_analysis_fields = ["primary_emotions", "emotional_patterns", "recommendations"]
            if not all(field in result["analysis"] for field in required_analysis_fields):
                raise ValueError("Missing required analysis fields")
                
            return result
            
        except (json.JSONDecodeError, ValueError) as e:
            print(f"Error parsing response: {e}")
            print(f"Attempted to parse: {json_str}")
            return {
                "role": "Emotional Intelligence Specialist",
                "analysis": {
                    "primary_emotions": ["Error in analysis"],
                    "emotional_patterns": ["Unable to parse response"],
                    "recommendations": ["Please try again"]
                }
            } 