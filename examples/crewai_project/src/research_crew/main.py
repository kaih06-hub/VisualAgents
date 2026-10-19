from research_crew.crew import ResearchCrew


def run():
    ResearchCrew().crew().kickoff(inputs={"topic": "AI agents"})


if __name__ == "__main__":
    run()
