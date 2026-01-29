"""
Query decomposition module.
Breaks complex queries into executable subtasks using LLM.
"""
import json
import re
from typing import List, Dict, Any
from pydantic import BaseModel, Field
from src.models.mlx_model import MLXModel
from src.utils.logging import get_logger

logger = get_logger(__name__)


class Task(BaseModel):
    """Represents a single executable task."""
    id: str = Field(description="Unique task identifier (e.g., 't1', 't2')")
    type: str = Field(description="Agent type: 'openbb', 'sec', 'fred', 'synthesis'")
    query: str = Field(description="Specific query for this task")
    ticker: str | None = Field(default=None, description="Stock ticker if applicable")
    depends_on: List[str] = Field(default_factory=list, description="Task IDs this task depends on")


class TaskPlan(BaseModel):
    """Complete task execution plan."""
    tasks: List[Task] = Field(description="List of tasks to execute")
    reasoning: str = Field(description="Explanation of task decomposition")
    can_parallelize: bool = Field(default=True, description="Whether tasks can run in parallel")


class QueryDecomposer:
    """Decomposes complex queries into executable task plans."""
    
    DECOMPOSITION_PROMPT = """You are a financial query analyzer. Break this complex query into executable subtasks.

Available agents:
- openbb: Stock prices, P/E ratios, financial metrics, market data (requires ticker symbol)
- sec: SEC filings, risk factors, 10-K/10-Q data, regulatory documents (requires ticker symbol)
- fred: GDP, inflation, unemployment, interest rates, macroeconomic indicators (no ticker needed)
- synthesis: Combine and analyze results from multiple agents (depends on other tasks)

Rules for task decomposition:
1. Create tasks that can run INDEPENDENTLY in parallel when possible
2. Only add dependencies (depends_on) when task truly needs results from another task
3. Always add a synthesis task if query needs multiple data sources combined
4. Keep task queries specific and actionable
5. Extract ticker symbols from original query
6. Avoid unnecessary sequential dependencies

Query: {query}

Output valid JSON in this exact format:
{{
  "tasks": [
    {{"id": "t1", "type": "openbb", "query": "Get AAPL P/E ratio", "ticker": "AAPL", "depends_on": []}},
    {{"id": "t2", "type": "fred", "query": "Get current GDP growth rate", "depends_on": []}},
    {{"id": "t3", "type": "synthesis", "query": "Compare AAPL P/E ratio to GDP growth", "depends_on": ["t1", "t2"]}}
  ],
  "reasoning": "Need both financial metric (P/E) and macro data (GDP), then compare",
  "can_parallelize": true
}}

Generate task plan:"""

    def __init__(self, model: MLXModel):
        """
        Initialize decomposer.
        
        Args:
            model: MLX model for task generation
        """
        self.model = model
        self.logger = logger
    
    async def decompose(self, query: str) -> TaskPlan:
        """
        Decompose query into task plan.
        
        Args:
            query: Complex user query
        
        Returns:
            TaskPlan with executable tasks
        
        Raises:
            ValueError: If decomposition fails
        """
        self.logger.info(f"Decomposing query: {query}")
        
        prompt = self.DECOMPOSITION_PROMPT.format(query=query)
        
        try:
            response = await self.model.generate(prompt, temperature=0.3)
            task_plan = self._parse_response(response.content)
            
            # Validate and optimize task plan
            task_plan = self._optimize_task_plan(task_plan)
            
            self.logger.info(
                f"Decomposed into {len(task_plan.tasks)} tasks, "
                f"parallel={task_plan.can_parallelize}"
            )
            
            return task_plan
            
        except Exception as e:
            self.logger.error(f"Decomposition failed: {e}")
            raise ValueError(f"Failed to decompose query: {e}")
    
    def _parse_response(self, content: str) -> TaskPlan:
        """
        Parse LLM response into TaskPlan.
        
        Args:
            content: LLM response content
        
        Returns:
            Parsed TaskPlan
        
        Raises:
            ValueError: If parsing fails
        """
        try:
            # Extract JSON from response
            if "```json" in content:
                json_start = content.find("```json") + 7
                json_end = content.find("```", json_start)
                json_str = content[json_start:json_end].strip()
            elif "{" in content:
                json_start = content.find("{")
                json_end = content.rfind("}") + 1
                json_str = content[json_start:json_end]
            else:
                raise ValueError("No JSON found in response")
            
            # Parse JSON
            data = json.loads(json_str)
            
            # Validate and create TaskPlan
            return TaskPlan.model_validate(data)
            
        except json.JSONDecodeError as e:
            self.logger.error(f"JSON parse error: {e}\nContent: {content}")
            raise ValueError(f"Failed to parse JSON: {e}")
        except Exception as e:
            self.logger.error(f"TaskPlan validation error: {e}")
            raise ValueError(f"Failed to validate task plan: {e}")
    
    def _optimize_task_plan(self, plan: TaskPlan) -> TaskPlan:
        """
        Optimize task plan by removing unnecessary dependencies.
        
        Args:
            plan: Original task plan
        
        Returns:
            Optimized task plan
        """
        # Remove circular dependencies
        for task in plan.tasks:
            # Remove self-dependencies
            task.depends_on = [dep for dep in task.depends_on if dep != task.id]
            
            # Remove dependencies on non-existent tasks
            valid_task_ids = {t.id for t in plan.tasks}
            task.depends_on = [dep for dep in task.depends_on if dep in valid_task_ids]
        
        # Verify no circular dependencies
        if self._has_circular_dependency(plan.tasks):
            self.logger.warning("Circular dependency detected, removing all dependencies")
            # Fallback: remove all dependencies, add synthesis at end
            for task in plan.tasks:
                if task.type != "synthesis":
                    task.depends_on = []
                else:
                    # Synthesis depends on all non-synthesis tasks
                    task.depends_on = [t.id for t in plan.tasks if t.type != "synthesis"]
        
        return plan
    
    def _has_circular_dependency(self, tasks: List[Task]) -> bool:
        """
        Check if task list has circular dependencies.
        
        Args:
            tasks: List of tasks to check
        
        Returns:
            True if circular dependency exists
        """
        visited = set()
        rec_stack = set()
        
        def has_cycle(task_id: str) -> bool:
            visited.add(task_id)
            rec_stack.add(task_id)
            
            # Find task
            task = next((t for t in tasks if t.id == task_id), None)
            if not task:
                return False
            
            # Check dependencies
            for dep_id in task.depends_on:
                if dep_id not in visited:
                    if has_cycle(dep_id):
                        return True
                elif dep_id in rec_stack:
                    return True
            
            rec_stack.remove(task_id)
            return False
        
        # Check all tasks
        for task in tasks:
            if task.id not in visited:
                if has_cycle(task.id):
                    return True
        
        return False
    
    def get_execution_order(self, plan: TaskPlan) -> List[List[str]]:
        """
        Get task execution order respecting dependencies.
        Returns list of task ID groups that can be executed in parallel.
        
        Args:
            plan: Task plan
        
        Returns:
            List of lists, where each inner list contains task IDs 
            that can run in parallel
        """
        execution_order = []
        completed = set()
        
        while len(completed) < len(plan.tasks):
            # Find ready tasks (dependencies met)
            ready = [
                task.id for task in plan.tasks
                if task.id not in completed
                and all(dep in completed for dep in task.depends_on)
            ]
            
            if not ready:
                # Deadlock, shouldn't happen with optimization
                remaining = [t.id for t in plan.tasks if t.id not in completed]
                self.logger.warning(f"Deadlock detected, forcing execution: {remaining}")
                execution_order.append(remaining)
                break
            
            execution_order.append(ready)
            completed.update(ready)
        
        return execution_order
