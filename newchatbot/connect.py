import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.llms import LLM
from create_memory_for_llm import Memory  # Import the Memory class

# Initialize memory
memory = Memory()

# Example function to utilize memory in LLM interactions
def interact_with_llm(user_input):
    try:
        # Retrieve relevant memory based on user input
        relevant_memory = memory.retrieve_memory(user_input)
        
        # Integrate memory into LLM processing
        llm_response = LLM.process(user_input, context=relevant_memory)
        
        return llm_response
    except Exception as e:
        print(f"An error occurred: {e}")
        return "Sorry, I couldn't process your request."

# Existing code for LLM interactions
# ...
