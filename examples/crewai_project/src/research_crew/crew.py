"""The standard `crewai create crew` template - two agents, two sequential tasks.

    pip install crewai crewai-tools        # and set OPENAI_API_KEY / SERPER_API_KEY
    visualagents run examples/crewai_project --cwd examples/crewai_project/src -- python -m research_crew.main
"""

from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import SerperDevTool


@CrewBase
class ResearchCrew:
    """Research crew"""

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    @agent
    def researcher(self) -> Agent:
        return Agent(config=self.agents_config["researcher"], verbose=True, tools=[SerperDevTool()])

    @agent
    def reporting_analyst(self) -> Agent:
        return Agent(config=self.agents_config["reporting_analyst"], verbose=True)

    @task
    def research_task(self) -> Task:
        return Task(config=self.tasks_config["research_task"])

    @task
    def reporting_task(self) -> Task:
        return Task(config=self.tasks_config["reporting_task"], output_file="report.md")

    @crew
    def crew(self) -> Crew:
        return Crew(agents=self.agents, tasks=self.tasks, process=Process.sequential, verbose=True)
