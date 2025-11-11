import os
import re
import sys
from typing import Dict, List, Any, Optional
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from dotenv import load_dotenv

import rag_db_retrieval

load_dotenv()

END_TOKEN = ""
MAX_HISTORY_TURNS = 10

def add_end_token(s: str) -> str:
    if not s.endswith(END_TOKEN):
        return s + END_TOKEN
    return s

def strip_end_token(s: str) -> str:
    if s.endswith(END_TOKEN):
        return s[:-len(END_TOKEN)]
    return s

# ============================================================================
# RAG RETRIEVAL INTEGRATION
# ============================================================================

def retrieve_rag_for_query(query: str, top_k: int = 3, verbose: bool = False) -> List[str]:
    """Retrieve relevant documents from the vector database."""
    try:
        if not hasattr(retrieve_rag_for_query, '_initialized'):
            if verbose:
                print("[RAG] Initializing...")
            
            retrieve_rag_for_query._embedding_manager = rag_db_retrieval.EmbeddingManager()
            retrieve_rag_for_query._vectorstore = rag_db_retrieval.VectorStore()
            retrieve_rag_for_query._rag_retriever = rag_db_retrieval.RAGRetriever(
                retrieve_rag_for_query._vectorstore,
                retrieve_rag_for_query._embedding_manager
            )
            retrieve_rag_for_query._initialized = True
        
        retrieved_results = retrieve_rag_for_query._rag_retriever.retrieve(query, top_k=top_k)
        docs_content = [result['content'] for result in retrieved_results]
        
        if verbose and docs_content:
            print(f"[RAG] Retrieved {len(docs_content)} documents")
        
        return docs_content
        
    except Exception as e:
        if verbose:
            print(f"[RAG] Error: {str(e)}")
        return []

# ============================================================================
# LCEL CHAINS
# ============================================================================

class StageAnalyzerChain:
    """Determines conversation stage"""
    
    def __init__(self, llm: ChatOpenAI, verbose: bool = False):
        self.llm = llm
        self.verbose = verbose
        
        prompt_text = """You are analyzing a conversation to determine its stage.

Conversation:
{history}

Select the stage (1-7):
1. Information Gathering
2. Service Inquiry
3. Needs Assessment
4. Solution Recommendation
5. Interest Confirmation
6. Lead Qualification
7. Conclusion

Respond with ONLY the number (1-7), nothing else."""
        
        self.prompt = PromptTemplate(template=prompt_text, input_variables=["history"])
        self.chain = self.prompt | self.llm | StrOutputParser()
    
    def run(self, history: str) -> str:
        try:
            result = self.chain.invoke({"history": history}).strip()
            match = re.search(r'[1-7]', result)
            return match.group(0) if match else '1'
        except Exception as e:
            if self.verbose:
                print(f"[ERROR] Stage analyzer: {e}")
            return '1'

class ConversationChain:
    """Generates bot responses"""
    
    def __init__(self, llm: ChatOpenAI, verbose: bool = False):
        self.llm = llm
        self.verbose = verbose
        
        prompt_text = """You are {name}, a sales assistant for {company}.

About {company}:
{business}

Purpose: {purpose}

Current Stage: {stage}

Conversation:
{history}

{name}: """
        
        self.prompt = PromptTemplate(
            template=prompt_text,
            input_variables=["name", "company", "business", "purpose", "stage", "history"]
        )
        self.chain = self.prompt | self.llm | StrOutputParser()
    
    def run(self, name: str, company: str, business: str, purpose: str, stage: str, history: str) -> str:
        try:
            result = self.chain.invoke({
                "name": name,
                "company": company,
                "business": business,
                "purpose": purpose,
                "stage": stage,
                "history": history
            })
            
            if not result or not result.strip():
                return "How can I help you further?"
            
            return result.strip()
        except Exception as e:
            if self.verbose:
                print(f"[ERROR] Conversation chain: {e}")
                import traceback
                traceback.print_exc()
            return "How can I help you further?"

# ============================================================================
# SALES BOT
# ============================================================================

class SalesBot:
    """Main sales bot controller"""
    
    def __init__(self, config: Dict, llm: ChatOpenAI, verbose: bool = False):
        self.config = config
        self.verbose = verbose
        self.conversation_history = []
        self.current_stage = '1'
        
        self.stage_analyzer = StageAnalyzerChain(llm, verbose=verbose)
        self.conversation_chain = ConversationChain(llm, verbose=verbose)
        
        self.stages = {
            '1': "Information Gathering",
            '2': "Service Inquiry",
            '3': "Needs Assessment",
            '4': "Solution Recommendation",
            '5': "Interest Confirmation",
            '6': "Lead Qualification",
            '7': "Conclusion"
        }
    
    def add_human_message(self, text: str):
        """Add user message to history"""
        self.conversation_history.append(f"User: {text}")
        if len(self.conversation_history) > MAX_HISTORY_TURNS * 2:
            self.conversation_history = self.conversation_history[-MAX_HISTORY_TURNS * 2:]
    
    def get_response(self):
        """Generate bot response"""
        try:
            # Get user query for RAG retrieval
            user_queries = [h.replace("User: ", "") for h in self.conversation_history if h.startswith("User:")]
            last_query = user_queries[-1] if user_queries else ""
            
            # Retrieve RAG context
            rag_docs = retrieve_rag_for_query(last_query, top_k=3, verbose=self.verbose)
            
            # Build history with RAG context
            history_text = "\n".join(self.conversation_history)
            if rag_docs:
                history_text = "Context:\n" + "\n---\n".join(rag_docs[:2]) + "\n\n" + history_text
            
            # Update stage
            try:
                self.current_stage = self.stage_analyzer.run(history_text)
            except:
                pass
            
            # Generate response
            response = self.conversation_chain.run(
                name=self.config['salesperson_name'],
                company=self.config['company_name'],
                business=self.config['company_business'],
                purpose=self.config['conversation_purpose'],
                stage=self.stages.get(self.current_stage, "Information Gathering"),
                history=history_text
            )
            
            # Add to history
            self.conversation_history.append(f"Assistant: {response}")
            
            return response
        
        except Exception as e:
            print(f"[ERROR] Failed to generate response: {e}")
            import traceback
            traceback.print_exc()
            return "I apologize, I encountered an error. Please try again."

# ============================================================================
# MAIN
# ============================================================================

def main():
    # Configuration
    config = {
        'salesperson_name': 'DMI Assistance',
        'company_name': 'DataM Intelligence',
        'company_business': 'Market research firm providing business intelligence across 70+ domains in 45+ countries. We offer syndicated reports, custom research, and dashboard subscriptions.',
        'conversation_purpose': 'Help leads with questions about DataM Intelligence and collect contact info if interested.'
    }
    
    # Initialize LLM
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("[ERROR] OPENROUTER_API_KEY not set!")
        return
    
    llm = ChatOpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
        model="openai/gpt-4o-mini",
        temperature=0.7
    )
    
    # Initialize bot
    bot = SalesBot(config, llm, verbose=True)
    
    # Print header
    print("\n" + "="*60)
    print("DATAM INTELLIGENCE SALES BOT")
    print("Type 'quit' to exit")
    print("="*60 + "\n")
    
    # Print initial greeting
    greeting = f"{config['salesperson_name']}: Hi there! I'm {config['salesperson_name']}, your virtual sales assistant from {config['company_name']}. How can I help you today?"
    print(greeting)
    bot.conversation_history.append(f"Assistant: {greeting}")
    
    # Main loop
    while True:
        try:
            user_input = input("\nYou: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ['quit', 'exit', 'bye', 'goodbye']:
                print(f"\n{config['salesperson_name']}: Thanks for chatting! Have a great day!")
                break
            
            # Add user message
            bot.add_human_message(user_input)
            
            # Get and print response
            print()
            response = bot.get_response()
            print(f"{config['salesperson_name']}: {response}")
        
        except KeyboardInterrupt:
            print(f"\n\n{config['salesperson_name']}: Goodbye!")
            break
        except Exception as e:
            print(f"\n[ERROR] {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()