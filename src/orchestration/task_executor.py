"""
Task executor with sequential execution.
Executes tasks one by one, respecting dependencies.
"""
from typing import List, Dict, Any
from src.orchestration.decomposer import Task, TaskPlan
from src.agents.sec_rag_agent import SECRAGAgent
from src.agents.openbb_agent import OpenBBAgent
from src.agents.fred_agent import FREDAgent
from src.agents.synthesis_agent import SynthesisAgent
from src.guardrails.schemas import AgentInput
from src.models.mlx_model import get_mlx_model
from src.utils.logging import get_logger
from src.utils.telemetry import get_tracer
from opentelemetry import trace

logger = get_logger(__name__)
tracer = get_tracer("orchestration.task_executor")


class TaskExecutor:
    """Executes task plans with parallel execution support."""
    
    def __init__(self):
        """Initialize task executor."""
        self.model = get_mlx_model()
        self.logger = logger
        
        # Agent instances (reused across tasks)
        self._agents = {}
    
    def _get_agent(self, agent_type: str):
        """Get or create agent instance."""
        if agent_type not in self._agents:
            if agent_type == "openbb":
                self._agents[agent_type] = OpenBBAgent(model=self.model)
            elif agent_type == "sec":
                self._agents[agent_type] = SECRAGAgent(model=self.model)
            elif agent_type == "fred":
                self._agents[agent_type] = FREDAgent(model=self.model)
            elif agent_type == "synthesis":
                self._agents[agent_type] = SynthesisAgent(model=self.model)
            else:
                raise ValueError(f"Unknown agent type: {agent_type}")
        
        return self._agents[agent_type]
    
    async def execute_plan(self, plan: TaskPlan, original_query: str) -> Dict[str, Any]:
        """
        Execute task plan sequentially (one task at a time).
        
        Args:
            plan: Task plan to execute
            original_query: Original user query
        
        Returns:
            Dictionary mapping task_id -> result
        """
        with tracer.start_as_current_span("task_executor.execute_plan") as span:
            span.set_attribute("task_count", len(plan.tasks))
            span.set_attribute("execution_mode", "sequential")
            
            self.logger.info(
                f"Executing plan with {len(plan.tasks)} tasks sequentially"
            )
            
            task_results = {}
            completed = set()
            
            # Execute tasks one by one, respecting dependencies
            while len(completed) < len(plan.tasks):
                # Find next ready task (dependencies met)
                ready_task = None
                for task in plan.tasks:
                    if task.id not in completed and all(dep in completed for dep in task.depends_on):
                        ready_task = task
                        break
                
                if not ready_task:
                    # Deadlock - shouldn't happen with optimized plan
                    self.logger.error(
                        f"No ready tasks. Completed: {completed}, "
                        f"Remaining: {[t.id for t in plan.tasks if t.id not in completed]}"
                    )
                    break
                
                self.logger.info(f"Executing task {ready_task.id} ({ready_task.type})")
                
                # Execute single task
                try:
                    result = await self._execute_task(ready_task, task_results, original_query)
                    task_results[ready_task.id] = result
                except Exception as e:
                    self.logger.error(f"Task {ready_task.id} failed: {e}")
                    task_results[ready_task.id] = {
                        "type": ready_task.type,
                        "status": "failed",
                        "error": str(e),
                        "result": {
                            "response_text": f"Error: {e}",
                            "citations": [],
                            "confidence_score": 0.0
                        }
                    }
                
                completed.add(ready_task.id)
            
            span.set_attribute("completed_tasks", len(completed))
            
            self.logger.info(
                f"Plan execution complete. {len(completed)}/{len(plan.tasks)} tasks completed sequentially"
            )
            
            return task_results
    
    async def _execute_task(
        self, 
        task: Task, 
        task_results: Dict[str, Any],
        original_query: str
    ) -> Dict[str, Any]:
        """
        Execute a single task.
        
        Args:
            task: Task to execute
            task_results: Results from previously completed tasks
            original_query: Original user query
        
        Returns:
            Task result dictionary
        """
        with tracer.start_as_current_span(f"execute_task.{task.id}") as span:
            span.set_attribute("task.id", task.id)
            span.set_attribute("task.type", task.type)
            span.set_attribute("task.query", task.query)
            span.set_attribute("task.depends_on", ",".join(task.depends_on))
            
            self.logger.info(f"Executing task {task.id} ({task.type}): {task.query}")
            
            try:
                # Get appropriate agent
                agent = self._get_agent(task.type)
                
                # Prepare input
                if task.type == "synthesis":
                    # Synthesis needs results from dependent tasks
                    dependent_results = {
                        dep_id: task_results[dep_id]
                        for dep_id in task.depends_on
                        if dep_id in task_results
                    }
                    
                    agent_input = AgentInput(
                        query=task.query or original_query,
                        filters={"ticker": task.ticker} if task.ticker else {},
                        metadata={"task_results": dependent_results}
                    )
                else:
                    # Regular agent
                    agent_input = AgentInput(
                        query=task.query,
                        filters={"ticker": task.ticker} if task.ticker else {}
                    )
                
                # Execute agent
                result = await agent.execute(agent_input)
                
                span.set_attribute("task.status", "success")
                span.set_attribute("task.confidence", result.confidence_score)
                span.set_attribute("task.citation_count", len(result.citations))
                
                return {
                    "type": task.type,
                    "status": "success",
                    "result": result.model_dump()
                }
                
            except Exception as e:
                self.logger.error(f"Task {task.id} execution failed: {e}")
                span.set_attribute("task.status", "failed")
                span.set_attribute("task.error", str(e))
                raise
