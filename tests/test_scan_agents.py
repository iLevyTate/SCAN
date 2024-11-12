from scan.scan_agents import PFCAgents


def test_initialize_agents():
    pfc_agents = PFCAgents(topic="Some topic")
    pfc_agents.initialize_agents()
    assert list(pfc_agents.agents.keys()) == ["DLPFC", "VMPFC", "OFC", "ACC", "MPFC"]


def test_create_agent():
    pfc_agents = PFCAgents(topic="Some topic")
    result = pfc_agents.create_agent("DLPFC")

    assert result.role == "DLPFC"
    assert (
        result.backstory
        == "You are the DLPFC, focusing on executive functions like planning and decision-making for the topic 'Some topic'.\nPlease ensure you follow the task instructions precisely and provide concise responses."
    )
    assert (
        result.goal
        == "Make decisions based on integrated logical, emotional, and social perspectives.\nEnsure you synthesize information effectively and provide strategic recommendations."
    )
