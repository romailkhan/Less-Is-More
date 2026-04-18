from typing import Dict, List
from langchain_core.prompts import ChatPromptTemplate
import json

from agents.openrouter_llm import get_openrouter_llm
from agents.token_usage import record_message


class Perception:
    def __init__(self):
        self.llm = get_openrouter_llm(role="perception")
        
        self.system_template = """
        You are a Perception Specialist. Your role is to analyze inputs to identify key patterns 
        and main ideas.

        Background: You are an expert at quickly identifying core messages, patterns, and key information
        in any input. Your strength is in breaking down complex information into clear insights.

        Your goal is to identify and analyze key patterns and main ideas in the input. Use the provided memories to help you.

        Focus on:
        - Core message and purpose
        - Main topics
        - Key relationships
        - Important patterns

        IMPORTANT: You must respond with ONLY valid JSON. Your entire response must be a single JSON object with exactly this structure:
        {{ # important: you must have these fields to not get parsed incorrectly
            "role": "Perception Specialist",
            "analysis": {{
                "main_topics": ["topic1", "topic2"],
                "key_patterns": ["pattern1", "pattern2"],
                "contextual_insights": ["insight1", "insight2"]
            }}
        }}
        Do not include any other text, thoughts, or explanations. The response must be pure JSON only.
        """

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_template),
            ("user", "{input}")
        ])

        self.chain = self.prompt | self.llm

    def analyze(self, input_text: str, memories: List[Dict]) -> Dict:
        """
        Analyze the input text to identify patterns and main ideas.
        
        Args:
            input_text (str): The text to analyze
            topic (str): The specific topic area for analysis (defaults to "General")
            
        Returns:
            Dict: Analysis results in the specified JSON format
        """
        response = self.chain.invoke({
            "input": input_text,
            "memories": memories
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
                
            required_analysis_fields = ["main_topics", "key_patterns", "contextual_insights"]
            if not all(field in result["analysis"] for field in required_analysis_fields):
                raise ValueError("Missing required analysis fields")
                
            return result
            
        except (json.JSONDecodeError, ValueError) as e:
            print(f"Error parsing response: {e}")
            print(f"Attempted to parse: {json_str}")
            return {
                "role": "Perception Specialist",
                "analysis": {
                    "main_topics": ["Error in analysis"],
                    "key_patterns": ["Unable to parse response"],
                    "contextual_insights": ["Please try again"]
                }
            }