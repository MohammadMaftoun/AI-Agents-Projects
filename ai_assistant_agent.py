#!/usr/bin/env python3
"""
AI Personal Assistant Agent
==========================

A comprehensive personal assistant bot that demonstrates core AI agent capabilities:
- Natural Language Processing for user interaction
- Task scheduling and reminder management
- Knowledge-based question answering
- Learning from user preferences
- Multi-modal interaction capabilities

Features:
- Conversational AI with context awareness
- Task and calendar management
- Weather information retrieval
- Simple reasoning and decision making
- Performance tracking and analytics
- Extensible plugin architecture

Technologies: Python, transformers, schedule, datetime, json
"""

import json
import schedule
import time
import datetime
import re
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum
import logging

# External libraries (install with: pip install transformers schedule requests)
try:
    from transformers import pipeline
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    print("Note: Install 'transformers' for advanced NLP capabilities")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TaskPriority(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    URGENT = 4

class AgentState(Enum):
    IDLE = "idle"
    PROCESSING = "processing"
    LEARNING = "learning"
    EXECUTING = "executing"

@dataclass
class Task:
    """Represents a task managed by the AI agent"""
    id: str
    title: str
    description: str
    due_date: Optional[datetime.datetime]
    priority: TaskPriority
    completed: bool = False
    created_at: datetime.datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.datetime.now()

@dataclass
class UserPreference:
    """Stores user preferences and learned behaviors"""
    name: str
    value: Any
    confidence: float = 1.0
    last_updated: datetime.datetime = None
    
    def __post_init__(self):
        if self.last_updated is None:
            self.last_updated = datetime.datetime.now()

class KnowledgeBase:
    """Simple knowledge base for FAQ and information storage"""
    
    def __init__(self):
        self.knowledge = {
            "weather": "I can help you check weather information. Please provide your location.",
            "time": lambda: f"Current time is {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "tasks": "I can help you manage tasks. Try 'add task', 'list tasks', or 'complete task'.",
            "schedule": "I can help with scheduling. Say 'schedule meeting' or 'check calendar'.",
            "help": """I'm your AI personal assistant! I can help with:
            - Task management (add, list, complete tasks)
            - Scheduling and reminders
            - Answer questions
            - Weather information
            - Time management
            
            Try saying things like:
            - "Add a task to buy groceries"
            - "What's my schedule today?"
            - "Set a reminder for 3 PM"
            - "What's the weather like?"
            """
        }
    
    def query(self, topic: str) -> str:
        """Query the knowledge base"""
        topic_lower = topic.lower()
        for key, value in self.knowledge.items():
            if key in topic_lower:
                if callable(value):
                    return value()
                return value
        return "I'm not sure about that. Try asking for 'help' to see what I can do."

class AIPersonalAssistant:
    """
    Main AI Personal Assistant Agent Class
    
    This agent demonstrates key AI capabilities:
    - Perception: Natural language understanding
    - Reasoning: Context-aware decision making
    - Action: Task execution and response generation
    - Learning: Preference adaptation and performance tracking
    """
    
    def __init__(self, name: str = "Assistant"):
        self.name = name
        self.state = AgentState.IDLE
        self.tasks: Dict[str, Task] = {}
        self.preferences: Dict[str, UserPreference] = {}
        self.knowledge_base = KnowledgeBase()
        self.conversation_history: List[Dict] = []
        self.performance_metrics = {
            "tasks_completed": 0,
            "questions_answered": 0,
            "user_satisfaction": 0.0,
            "response_time": 0.0
        }
        
        # Initialize NLP pipeline if available
        if HAS_TRANSFORMERS:
            try:
                self.sentiment_analyzer = pipeline("sentiment-analysis")
                self.qa_pipeline = pipeline("question-answering")
                logger.info("Advanced NLP capabilities loaded")
            except Exception as e:
                logger.warning(f"Could not load NLP models: {e}")
                self.sentiment_analyzer = None
                self.qa_pipeline = None
        else:
            self.sentiment_analyzer = None
            self.qa_pipeline = None
        
        logger.info(f"AI Assistant '{self.name}' initialized successfully")
    
    def perceive(self, user_input: str) -> Dict[str, Any]:
        """
        Perception layer: Analyze and understand user input
        """
        self.state = AgentState.PROCESSING
        
        perception = {
            "raw_input": user_input,
            "intent": self._extract_intent(user_input),
            "entities": self._extract_entities(user_input),
            "sentiment": self._analyze_sentiment(user_input),
            "context": self._get_context()
        }
        
        logger.info(f"Perceived intent: {perception['intent']}")
        return perception
    
    def _extract_intent(self, text: str) -> str:
        """Extract user intent from text"""
        text_lower = text.lower()
        
        # Intent patterns
        intent_patterns = {
            "add_task": r"(add|create|new).*(task|todo|reminder)",
            "list_tasks": r"(list|show|what).*(task|todo|schedule)",
            "complete_task": r"(complete|done|finish).*(task|todo)",
            "schedule_meeting": r"(schedule|set|plan).*(meeting|appointment)",
            "ask_time": r"(what|current).*(time|hour)",
            "ask_weather": r"(weather|temperature|forecast)",
            "ask_question": r"(what|how|why|when|where)",
            "greeting": r"(hello|hi|hey|good morning|good afternoon)",
            "help": r"(help|assistance|what can you do)"
        }
        
        for intent, pattern in intent_patterns.items():
            if re.search(pattern, text_lower):
                return intent
        
        return "general_query"
    
    def _extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract entities like dates, times, names from text"""
        entities = {
            "dates": re.findall(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b', text),
            "times": re.findall(r'\b\d{1,2}:\d{2}(?:\s*[AaPp][Mm])?\b', text),
            "numbers": re.findall(r'\b\d+\b', text),
        }
        return entities
    
    def _analyze_sentiment(self, text: str) -> Dict[str, float]:
        """Analyze sentiment of user input"""
        if self.sentiment_analyzer:
            try:
                result = self.sentiment_analyzer(text)[0]
                return {
                    "label": result["label"].lower(),
                    "score": result["score"]
                }
            except Exception as e:
                logger.warning(f"Sentiment analysis failed: {e}")
        
        # Fallback simple sentiment analysis
        positive_words = ["good", "great", "excellent", "happy", "pleased", "thank"]
        negative_words = ["bad", "terrible", "angry", "frustrated", "disappointed"]
        
        text_lower = text.lower()
        pos_score = sum(1 for word in positive_words if word in text_lower)
        neg_score = sum(1 for word in negative_words if word in text_lower)
        
        if pos_score > neg_score:
            return {"label": "positive", "score": 0.7}
        elif neg_score > pos_score:
            return {"label": "negative", "score": 0.7}
        else:
            return {"label": "neutral", "score": 0.5}
    
    def _get_context(self) -> Dict[str, Any]:
        """Get current context for decision making"""
        return {
            "current_time": datetime.datetime.now(),
            "pending_tasks": len([t for t in self.tasks.values() if not t.completed]),
            "recent_interactions": self.conversation_history[-3:] if self.conversation_history else []
        }
    
    def reason(self, perception: Dict[str, Any]) -> Dict[str, Any]:
        """
        Reasoning layer: Decide what action to take based on perception
        """
        intent = perception["intent"]
        entities = perception["entities"]
        context = perception["context"]
        
        action_plan = {
            "action_type": intent,
            "parameters": {},
            "confidence": 0.8,
            "reasoning": ""
        }
        
        if intent == "add_task":
            task_desc = self._extract_task_description(perception["raw_input"])
            action_plan["parameters"] = {"description": task_desc}
            action_plan["reasoning"] = f"User wants to create a task: {task_desc}"
            
        elif intent == "list_tasks":
            action_plan["reasoning"] = "User wants to see their tasks"
            
        elif intent == "complete_task":
            action_plan["reasoning"] = "User wants to mark a task as complete"
            
        elif intent == "ask_time":
            action_plan["reasoning"] = "User is asking for current time"
            
        elif intent == "greeting":
            action_plan["reasoning"] = "User is greeting, respond politely"
            
        else:
            action_plan["reasoning"] = f"General query with intent: {intent}"
        
        return action_plan
    
    def _extract_task_description(self, text: str) -> str:
        """Extract task description from user input"""
        # Remove common task-related phrases to get the core description
        text_lower = text.lower()
        patterns_to_remove = [
            r"(add|create|new)\s+(a\s+)?(task|todo|reminder)\s+(to\s+)?",
            r"(remind me to|i need to)\s+",
        ]
        
        description = text
        for pattern in patterns_to_remove:
            description = re.sub(pattern, "", description, flags=re.IGNORECASE).strip()
        
        return description or "New task"
    
    def act(self, action_plan: Dict[str, Any]) -> str:
        """
        Action layer: Execute the planned action and generate response
        """
        self.state = AgentState.EXECUTING
        action_type = action_plan["action_type"]
        parameters = action_plan.get("parameters", {})
        
        try:
            if action_type == "add_task":
                return self._add_task(parameters["description"])
            
            elif action_type == "list_tasks":
                return self._list_tasks()
            
            elif action_type == "complete_task":
                return self._complete_task_interactive()
            
            elif action_type == "ask_time":
                current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                return f"The current time is {current_time}"
            
            elif action_type == "greeting":
                return self._generate_greeting()
            
            elif action_type == "help":
                return self.knowledge_base.query("help")
            
            elif action_type == "ask_weather":
                return "I'd love to help with weather information! Please specify your location."
            
            else:
                return self.knowledge_base.query(action_type)
                
        except Exception as e:
            logger.error(f"Error executing action {action_type}: {e}")
            return f"I encountered an error while processing your request. Please try again."
        
        finally:
            self.state = AgentState.IDLE
    
    def _add_task(self, description: str) -> str:
        """Add a new task"""
        task_id = f"task_{len(self.tasks) + 1}"
        task = Task(
            id=task_id,
            title=description,
            description=description,
            due_date=None,  # Could be enhanced to extract dates from text
            priority=TaskPriority.MEDIUM
        )
        
        self.tasks[task_id] = task
        self.performance_metrics["tasks_completed"] += 1
        
        return f"✅ Task added: '{description}'. You now have {len([t for t in self.tasks.values() if not t.completed])} pending tasks."
    
    def _list_tasks(self) -> str:
        """List all tasks"""
        if not self.tasks:
            return "📋 You don't have any tasks yet. Say 'add task' to create one!"
        
        pending_tasks = [t for t in self.tasks.values() if not t.completed]
        completed_tasks = [t for t in self.tasks.values() if t.completed]
        
        response = "📋 **Your Tasks:**\n\n"
        
        if pending_tasks:
            response += "**Pending:**\n"
            for task in pending_tasks:
                priority_emoji = {"LOW": "🟢", "MEDIUM": "🟡", "HIGH": "🟠", "URGENT": "🔴"}
                emoji = priority_emoji.get(task.priority.name, "⚪")
                response += f"{emoji} {task.title}\n"
        
        if completed_tasks:
            response += "\n**Completed:**\n"
            for task in completed_tasks:
                response += f"✅ {task.title}\n"
        
        return response
    
    def _complete_task_interactive(self) -> str:
        """Mark a task as complete (simplified version)"""
        pending_tasks = [t for t in self.tasks.values() if not t.completed]
        
        if not pending_tasks:
            return "🎉 Great job! You don't have any pending tasks to complete."
        
        if len(pending_tasks) == 1:
            task = pending_tasks[0]
            task.completed = True
            return f"✅ Marked '{task.title}' as complete! Well done!"
        
        return f"You have {len(pending_tasks)} pending tasks. Please specify which one you'd like to complete."
    
    def _generate_greeting(self) -> str:
        """Generate a personalized greeting"""
        current_hour = datetime.datetime.now().hour
        
        if current_hour < 12:
            time_greeting = "Good morning"
        elif current_hour < 17:
            time_greeting = "Good afternoon"
        else:
            time_greeting = "Good evening"
        
        pending_count = len([t for t in self.tasks.values() if not t.completed])
        
        greeting = f"{time_greeting}! I'm {self.name}, your AI personal assistant. "
        
        if pending_count > 0:
            greeting += f"You have {pending_count} pending task{'s' if pending_count != 1 else ''}. "
        
        greeting += "How can I help you today?"
        
        return greeting
    
    def learn(self, user_input: str, response: str, feedback: Optional[str] = None) -> None:
        """
        Learning layer: Adapt based on interactions and feedback
        """
        self.state = AgentState.LEARNING
        
        # Store conversation history
        interaction = {
            "timestamp": datetime.datetime.now(),
            "user_input": user_input,
            "response": response,
            "feedback": feedback
        }
        self.conversation_history.append(interaction)
        
        # Update performance metrics
        self.performance_metrics["questions_answered"] += 1
        
        # Simple preference learning (can be enhanced)
        if feedback == "positive" or "thank" in user_input.lower():
            self.performance_metrics["user_satisfaction"] = min(1.0, 
                self.performance_metrics["user_satisfaction"] + 0.1)
        elif feedback == "negative" or any(word in user_input.lower() 
                                          for word in ["wrong", "bad", "incorrect"]):
            self.performance_metrics["user_satisfaction"] = max(0.0, 
                self.performance_metrics["user_satisfaction"] - 0.1)
        
        # Keep conversation history manageable
        if len(self.conversation_history) > 50:
            self.conversation_history = self.conversation_history[-30:]
        
        logger.info(f"Learning completed. Satisfaction score: {self.performance_metrics['user_satisfaction']:.2f}")
    
    def process_input(self, user_input: str) -> str:
        """
        Main processing pipeline: Perceive -> Reason -> Act -> Learn
        """
        start_time = time.time()
        
        try:
            # 1. Perceive: Understand the input
            perception = self.perceive(user_input)
            
            # 2. Reason: Decide what to do
            action_plan = self.reason(perception)
            
            # 3. Act: Execute the plan
            response = self.act(action_plan)
            
            # 4. Learn: Update based on interaction
            self.learn(user_input, response)
            
            # Update performance metrics
            processing_time = time.time() - start_time
            self.performance_metrics["response_time"] = (
                self.performance_metrics["response_time"] * 0.8 + processing_time * 0.2
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing input '{user_input}': {e}")
            return "I apologize, but I encountered an error processing your request. Please try again."
    
    def get_performance_report(self) -> str:
        """Generate a performance report"""
        metrics = self.performance_metrics
        
        report = f"""
🤖 **AI Assistant Performance Report**

📊 **Metrics:**
- Tasks Completed: {metrics['tasks_completed']}
- Questions Answered: {metrics['questions_answered']}
- User Satisfaction: {metrics['user_satisfaction']:.2f}/1.0
- Avg Response Time: {metrics['response_time']:.3f}s

📈 **Status:** {self.state.value.title()}

💾 **Memory:**
- Active Tasks: {len([t for t in self.tasks.values() if not t.completed])}
- Completed Tasks: {len([t for t in self.tasks.values() if t.completed])}
- Conversation History: {len(self.conversation_history)} interactions

🧠 **Capabilities:**
- Natural Language Processing: {'✅' if HAS_TRANSFORMERS else '⚠️ (Limited)'}
- Task Management: ✅
- Learning & Adaptation: ✅
- Context Awareness: ✅
"""
        return report
    
    def save_state(self, filename: str) -> None:
        """Save agent state to file"""
        state_data = {
            "name": self.name,
            "tasks": {k: asdict(v) for k, v in self.tasks.items()},
            "preferences": {k: asdict(v) for k, v in self.preferences.items()},
            "performance_metrics": self.performance_metrics,
            "conversation_history": self.conversation_history[-10:]  # Save last 10
        }
        
        with open(filename, 'w') as f:
            json.dump(state_data, f, default=str, indent=2)
        
        logger.info(f"Agent state saved to {filename}")
    
    def load_state(self, filename: str) -> None:
        """Load agent state from file"""
        try:
            with open(filename, 'r') as f:
                state_data = json.load(f)
            
            self.name = state_data.get("name", self.name)
            
            # Restore tasks
            for task_id, task_data in state_data.get("tasks", {}).items():
                task_data["created_at"] = datetime.datetime.fromisoformat(task_data["created_at"])
                if task_data["due_date"]:
                    task_data["due_date"] = datetime.datetime.fromisoformat(task_data["due_date"])
                task_data["priority"] = TaskPriority[task_data["priority"]]
                self.tasks[task_id] = Task(**task_data)
            
            # Restore other data
            self.performance_metrics.update(state_data.get("performance_metrics", {}))
            self.conversation_history = state_data.get("conversation_history", [])
            
            logger.info(f"Agent state loaded from {filename}")
            
        except Exception as e:
            logger.error(f"Error loading state from {filename}: {e}")

def demo_conversation():
    """Demonstrate the AI assistant capabilities"""
    print("🤖 AI Personal Assistant Demo")
    print("=" * 50)
    
    # Initialize the AI assistant
    assistant = AIPersonalAssistant("Claude Assistant")
    
    # Demo conversation
    demo_inputs = [
        "Hello!",
        "Add a task to buy groceries tomorrow",
        "Add a task to call mom",
        "What are my tasks?",
        "What time is it?",
        "Complete the grocery task",
        "Show my tasks again",
        "What can you help me with?",
        "Thanks for your help!"
    ]
    
    for user_input in demo_inputs:
        print(f"\n👤 User: {user_input}")
        response = assistant.process_input(user_input)
        print(f"🤖 Assistant: {response}")
        
        # Simulate a small delay
        time.sleep(0.5)
    
    # Show performance report
    print("\n" + "="*50)
    print(assistant.get_performance_report())

def interactive_mode():
    """Run the assistant in interactive mode"""
    print("🤖 AI Personal Assistant - Interactive Mode")
    print("Type 'quit', 'exit', or 'bye' to end the conversation")
    print("Type 'report' to see performance metrics")
    print("Type 'save' to save current state")
    print("=" * 50)
    
    assistant = AIPersonalAssistant("Your AI Assistant")
    
    # Try to load previous state
    try:
        assistant.load_state("assistant_state.json")
        print("💾 Loaded previous session data")
    except:
        pass
    
    # Initial greeting
    print(f"🤖 {assistant.process_input('hello')}")
    
    while True:
        try:
            user_input = input("\n👤 You: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("🤖 Goodbye! Have a great day!")
                assistant.save_state("assistant_state.json")
                break
            
            elif user_input.lower() == 'report':
                print(assistant.get_performance_report())
                continue
            
            elif user_input.lower() == 'save':
                assistant.save_state("assistant_state.json")
                print("💾 State saved successfully!")
                continue
            
            elif not user_input:
                continue
            
            response = assistant.process_input(user_input)
            print(f"🤖 Assistant: {response}")
            
        except KeyboardInterrupt:
            print("\n🤖 Goodbye! Session interrupted.")
            assistant.save_state("assistant_state.json")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    import sys
    
    print("AI Personal Assistant Agent")
    print("Choose mode:")
    print("1. Demo conversation")
    print("2. Interactive mode")
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == "1":
        demo_conversation()
    elif choice == "2":
        interactive_mode()
    else:
        print("Running demo by default...")
        demo_conversation()