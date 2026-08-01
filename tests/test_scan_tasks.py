import pytest

from scan.roles import ROLES, execution_order
from scan.scan_agents import PFCAgents
from scan.scan_tasks import PFCTasks

TOPIC = "Test topic"


@pytest.fixture
def pfc_tasks():
    return PFCTasks(PFCAgents(topic="Some topic"))


@pytest.mark.parametrize("role", ROLES, ids=lambda role: role.name)
def test_build_produces_the_roles_task(pfc_tasks, role):
    task = pfc_tasks.build(role.name, TOPIC)

    assert task.name == role.task_name
    assert TOPIC in task.description
    assert task.expected_output == role.expected_output
    # Regression: nothing used to assert which agent a task was given, so a task could be
    # reassigned and the report heading would keep naming the old region.
    assert task.agent.role == role.name


def test_build_rejects_an_unknown_role(pfc_tasks):
    with pytest.raises(KeyError, match="Unknown PFC role"):
        pfc_tasks.build("DLPFCC", TOPIC)


def test_task_context_defaults_empty(pfc_tasks):
    assert pfc_tasks.build("DLPFC", TOPIC).context == []


def test_build_all_wires_context_from_the_role_table(pfc_tasks):
    # Regression: dependencies were once passed via an invalid `dependencies=` kwarg that
    # crewai silently dropped, so inter-task context was never actually applied.
    tasks = pfc_tasks.build_all(TOPIC)
    by_name = {task.name: task for task in tasks}

    for role in ROLES:
        expected = [by_name[dependency] for dependency in role.depends_on]
        assert by_name[role.task_name].context == expected


def test_build_all_orders_dependencies_first(pfc_tasks):
    tasks = pfc_tasks.build_all(TOPIC)
    order = [task.name for task in tasks]

    assert order == [role.task_name for role in execution_order()]
    for task in tasks:
        for dependency in task.context:
            assert order.index(dependency.name) < order.index(task.name)
