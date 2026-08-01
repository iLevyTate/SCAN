from __future__ import annotations

from typing import TYPE_CHECKING

from crewai import Task

from scan.roles import execution_order, get_role

if TYPE_CHECKING:
    from collections.abc import Sequence

    from scan.roles import PFCRole, RoleName
    from scan.scan_agents import PFCAgents

__all__ = ["PFCTasks"]


class PFCTasks:
    """Builds the crewai task for each PFC role.

    This replaced five near-identical 16-line methods that differed only in their description
    text, expected output and agent -- all of which now live in :mod:`scan.roles`.
    """

    def __init__(self, agents: PFCAgents) -> None:
        self.agents = agents

    def build(self, role_name: RoleName | str, topic: str, context: Sequence[Task] = ()) -> Task:
        """Build the task belonging to one role."""
        role: PFCRole = get_role(role_name)
        return Task(
            name=role.task_name,
            description=role.description(topic),
            expected_output=role.expected_output,
            agent=self.agents.agents[role.name],
            context=list(context),
        )

    def build_all(self, topic: str) -> list[Task]:
        """Build every task, wiring dependencies through crewai's ``context``.

        Returned in execution order, so each task is preceded by the tasks it consumes.
        """
        by_task_name: dict[str, Task] = {}
        ordered: list[Task] = []
        for role in execution_order():
            context = [by_task_name[dependency] for dependency in role.depends_on]
            task = self.build(role.name, topic, context=context)
            by_task_name[role.task_name] = task
            ordered.append(task)
        return ordered
