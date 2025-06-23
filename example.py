from EmbeddingModel import EmbeddingModel
from PDFLoader import PDFLoader
from QwenModel import QwenModel
from TopTracker import TopKTracker
import re
import heapq
from Tools import PDFSearchTool

pdf_loader = PDFLoader('relativity.pdf')
embedding_model = EmbeddingModel()
qwen_model = QwenModel()
tools = {
    "PDFSearchTool": PDFSearchTool(pdf_loader)
}

question = "In the text, what means Experimental tests of general relativity?"
page = 6
top_relevant_text = TopKTracker(3)
max_iterations = 10

system_prompt = """You are an AI that can reason step-by-step and use tools to answer questions.

Available tools:
1. PDFSearchTool[page]: search the page for the answer

IMPORTANT: Only provide ONE step at a time. I will provide you the Observation after you answer with the Action.

Use this format for each step:
Thought: your reasoning for this single step
Action: ToolName[parameters]

OR if you have enough information:
Final Answer: your final answer

Wait for the Observation before proceeding to the next step.
"""
conversation = system_prompt + f"\nQuestion: {question}\n"

for i in range(max_iterations):
    print(f"\n--- Iteration {i + 1} ---")
    answer = qwen_model.generate_text(conversation)
    print(f"Model Response:\n{answer}")
    
    # Check for final answer first
    if "Final Answer:" in answer:
        print("\n=== FINAL ANSWER FOUND ===")
        final_answer = answer.split("Final Answer:")[1].strip()
        print(f"Final Answer: {final_answer}")
        break
    match = re.search(r"Action:\s*(\w+)\[(.*?)\]", answer)
    if match:
        tool_name, params = match.groups()
        print(f"\nExecuting: {tool_name}[{params}]")
        
        tool = tools.get(tool_name)
        if not tool:
            observation = f"Error: Tool {tool_name} not available"
        else:
            try:
                observation = tool(params)
                print(f"Tool Result: {observation[:200]}...")  # Show first 200 chars
            except Exception as e:
                observation = f"Error executing tool: {str(e)}"
        
        # Add the model's response and observation to conversation
        conversation += answer + f"\nObservation: {observation}\n"
    else:
        print("No action found in response. Ending loop.")
        break

print(f"\nTotal iterations: {i + 1}")
