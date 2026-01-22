"""
Childcare AI Assistant - Main Chainlit Application
Advanced RAG system with real-time processing visualization
"""

import sys
import os
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

# CPU-only mode (for deployment)
os.environ['USE_GPU'] = 'false'

from src.utils.warnings_suppressor import suppress_warnings, setup_clean_logging

import chainlit as cl
from rag_chainlit_integration import rag_integration

import time
from dotenv import load_dotenv
load_dotenv(os.path.join(parent_dir, '.env'))

@cl.password_auth_callback
def auth_callback(username: str, password: str):
    """
    Simple password authentication using credentials from .env
    Supports predefined users without session management
    """
    auth_users_env = [
        os.getenv('AUTH_USER_1'),
        os.getenv('AUTH_USER_2'), 
        os.getenv('AUTH_USER_3'),
        os.getenv('AUTH_USER_4')
    ]
    
    print(f"DEBUG: Auth attempt for username: {username}")
    print(f"DEBUG: Environment variables loaded: {[f'{i+1}: {bool(v)}' for i, v in enumerate(auth_users_env)]}")
    
    # Filter out None values
    auth_users_env = [user for user in auth_users_env if user]
    
    if not auth_users_env:
        print("ERROR: No authentication credentials configured")
        return None
    
    auth_users = {}
    try:
        for auth_user in auth_users_env:
            if ':' in auth_user:
                user, pwd = auth_user.split(':', 1)
                auth_users[user] = pwd
        print(f"DEBUG: Loaded users: {list(auth_users.keys())}")
    except Exception as e:
        print(f"ERROR: Error parsing authentication credentials: {e}")
        return None
    
    if username in auth_users and auth_users[username] == password:
        print(f"SUCCESS: Login successful for {username}")
        role = 'admin' if 'admin' in username.lower() else \
               'expert' if 'expert' in username.lower() else \
               'demo' if 'demo' in username.lower() else 'user'
            
        return cl.User(
            identifier=username,
            metadata={
                "role": role,
                "provider": "simple_auth",
                "login_time": str(time.time())
            }
        )
    else:
        print(f"FAILED: Invalid credentials for {username}")
        return None


@cl.on_chat_start
async def on_chat_start():
    """Initialize chat session"""
    
    user = cl.user_session.get("user")
    username = user.identifier if user else "Guest"
    user_role = user.metadata.get("role", "user") if user else "guest"
    
    cl.user_session.set("session_started", True)
    cl.user_session.set("query_count", 0)
    cl.user_session.set("username", username)
    cl.user_session.set("user_role", user_role)
    
    welcome_message = f"""
# Welcome to Childcare AI Assistant

**Hello {username}!** I'm your advanced AI assistant powered by a sophisticated **Retrieval-Augmented Generation (RAG)** system, specifically designed to help with childcare questions and parenting guidance.

## **What I Can Help With:**
- **Child Development**: Physical, cognitive, emotional, and social milestones
- **Parenting Strategies**: Discipline, communication, and positive parenting techniques  
- **Sleep & Routines**: Bedtime routines, sleep training, and healthy habits
- **Nutrition & Health**: Age-appropriate foods, feeding schedules, and health concerns
- **Behavioral Guidance**: Managing tantrums, encouraging good behavior
- **Educational Activities**: Learning games, developmental activities
- **Safety & Childproofing**: Home safety, age-appropriate precautions

## **Advanced Processing Features:**
- **Real-time Step Visualization**: Watch each processing step unfold
- **Query Expansion**: Multiple formulations for comprehensive search
- **HyDE Generation**: Hypothetical document enhancement
- **Multi-Retrieval**: Parallel search across knowledge base
- **Intelligent Re-ranking**: Cohere-powered relevance optimization
- **Confidence Assessment**: CRAG-based answer validation

## **Example Questions:**
- "How can I establish a bedtime routine for my 2-year-old?"
- "What are effective discipline strategies for toddlers?"
- "When should my child reach certain developmental milestones?"
- "How do I handle separation anxiety in preschoolers?"

**Ask me anything about childcare - I'll process your question through our advanced 7-step pipeline!**
"""
    
    await cl.Message(
        content=welcome_message,
        author="Childcare AI Assistant"
    ).send()


@cl.on_message
async def on_message(message: cl.Message):
    """Handle incoming messages"""
    try:
        suppress_warnings()
        setup_clean_logging()
        
        query_count = cl.user_session.get("query_count", 0) + 1
        cl.user_session.set("query_count", query_count)
        username = cl.user_session.get("username", "Guest")
        
        user_query = message.content.strip()
        
        if not user_query:
            await cl.Message(
                content="Please ask me a question about childcare!",
                author="Childcare AI Assistant"
            ).send()
            return
        
        processing_msg = cl.Message(
            content="**Processing your query through Advanced RAG pipeline...**\n\nWatch the steps unfold below:",
            author="System"
        )
        await processing_msg.send()
        
        response = await rag_integration.process_complete_pipeline(user_query)
        
        answer = response.get('answer', 'No answer generated')
        confidence = response.get('confidence', 0.0)
        action = response.get('action', 'unknown')
        strategy = response.get('strategy', 'unknown')
        confidence_tier = response.get('confidence_tier', 'unknown')
        assessment_details = response.get('assessment_details', {})
        evaluation_data = response.get('evaluation', {})
        
        quality_display = ""
        if evaluation_data:
            quality_grade = evaluation_data.get('quality_grade', 'N/A')
            composite_score = evaluation_data.get('llm_quality_score', 0.0)
            
            quality_label = "Excellent Quality" if quality_grade in ['A+', 'A'] else \
                           "Good Quality" if quality_grade in ['A-', 'B+', 'B'] else \
                           "Fair Quality"
            
            quality_display = f"**{quality_label} ({quality_grade}): {composite_score:.3f}/1.0**"
        else:
            quality_label = "High Confidence" if confidence > 0.85 else \
                           "Medium Confidence" if confidence > 0.65 else \
                           "Low Confidence"
            
            quality_display = f"**{quality_label}: {confidence:.3f}/1.0**"
        
        strategy_display = action.replace('_', ' ').title()
        if 'local' in action:
            strategy_display += " (Knowledge Base Only)"
        elif 'hybrid' in action:
            strategy_display += " (Knowledge Base + Web Search)"
        elif 'web' in action:
            strategy_display += " (Web Search Primary)"
        
        final_response = f"""**Answer:**
{answer}

{quality_display}
**Action Taken:** {strategy_display}
**Query #{query_count}** | **Processing Complete**"""
        
        await cl.Message(
            content=final_response,
            author="Childcare AI Assistant"
        ).send()
        
        processing_msg.content = "**RAG Pipeline Processing Complete!**\n\nAll 7 steps executed successfully."
        await processing_msg.update()
        
    except Exception as e:
        error_message = f"""
**Error Processing Query**

I encountered an issue while processing your question:
```
{str(e)}
```

Please try asking your question again, or rephrase it differently.
"""
        await cl.Message(
            content=error_message,
            author="Childcare AI Assistant"
        ).send()


@cl.on_stop
async def on_stop():
    """Handle stop button during processing"""
    await cl.Message(
        content="**Processing Stopped**\n\nYou can ask a new question anytime!",
        author="System"
    ).send()


@cl.on_chat_end
async def on_chat_end():
    """Handle chat session end"""
    username = cl.user_session.get("username", "Guest")
    query_count = cl.user_session.get("query_count", 0)
    
    goodbye_message = f"""
**Goodbye {username}!**

Thank you for using the Childcare AI Assistant!

**Session Summary:**
- Queries processed: {query_count}
- Session duration: Active

Take care and feel free to return anytime with your childcare questions!
"""
    
    await cl.Message(
        content=goodbye_message,
        author="Childcare AI Assistant"
    ).send()


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    import subprocess
    import sys
    
    subprocess.run([
        sys.executable, "-m", "chainlit", "run", "app.py", 
        "--host", "0.0.0.0", 
        "--port", str(port),
        "--headless"
    ])
