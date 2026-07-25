"""The single source of truth for SCAN's prefrontal-cortex roles.

Each region owns exactly one agent and exactly one task, so one table describes both. Everything
else in the package derives from ``ROLES``: the agent personas (:mod:`scan.scan_agents`), the crew
tasks and their dependency wiring (:mod:`scan.scan_tasks`), the execution order and the report
layout (:mod:`scan.main`).

Adding a region means adding one :class:`PFCRole` here plus its ``*_MODEL`` field in
:class:`scan.config.Settings`; a test asserts those two stay in step.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Literal, TypeAlias

if TYPE_CHECKING:
    from collections.abc import Mapping

RoleName: TypeAlias = Literal["DLPFC", "VMPFC", "OFC", "ACC", "MPFC"]


@dataclass(frozen=True, slots=True)
class PFCRole:
    """One prefrontal-cortex region: its agent persona and the task it performs."""

    name: RoleName
    #: Attribute name on ``Settings`` -- never a resolved model string. The value must be read
    #: lazily via ``getattr(settings, role.model_setting)``, because callers (and tests) override
    #: settings after import.
    model_setting: str
    #: Completes "You are the {name}, focusing on {focus} for the topic '{topic}'."
    focus: str
    goal: str
    #: Report heading, rendered as "{section_title} ({name})".
    section_title: str
    #: Position in the report, which is not the order tasks execute in.
    report_order: int
    task_name: str
    task_intro: str
    task_actions: tuple[str, ...]
    task_expectation: str
    expected_output: str
    #: ``task_name``s whose output is fed to this task as crewai ``context``.
    depends_on: tuple[str, ...] = ()

    def backstory(self, topic: str) -> str:
        return (
            f"You are the {self.name}, focusing on {self.focus} for the topic '{topic}'.\n"
            "Please ensure you follow the task instructions precisely and provide concise responses."
        )

    def context_legend(self) -> str:
        """Name the upstream analyses crewai will splice in, in the order it splices them.

        crewai concatenates dependency outputs with rows of dashes and *no labels at all*
        (``crewai/utilities/formatter.py``), so without this the model receives anonymous
        blobs and cannot attribute anything. Generated from ``depends_on`` -- the same field
        that builds ``Task.context`` -- so the legend cannot drift from the actual wiring.
        """
        if not self.depends_on:
            return ""
        listed = "\n".join(
            f"{index}. {BY_TASK_NAME[task_name].section_title}, "
            f"produced by the {BY_TASK_NAME[task_name].name}."
            for index, task_name in enumerate(self.depends_on, start=1)
        )
        return (
            "The analyses you depend on are supplied below, separated by rows of dashes, "
            "in exactly this order:\n"
            f"{listed}\n"
            "Attribute every claim you take from them to the region that produced it, by name."
        )

    def description(self, topic: str) -> str:
        """Render the crewai task description for this role."""
        actions = "\n".join(f"- {action.format(topic=topic)}" for action in self.task_actions)
        parts = [
            self.task_intro.format(topic=topic),
            "Actions Required:",
            actions,
            "Expected Output:",
            self.task_expectation.format(topic=topic),
        ]
        legend = self.context_legend()
        if legend:
            parts.insert(1, legend)
        return "\n".join(parts)


ROLES: tuple[PFCRole, ...] = (
    # Declared in execution order. Every role except the first MUST declare depends_on:
    # crewai's Crew._get_context uses task.context only when it is truthy, and otherwise
    # falls back to the *immediately preceding* task's raw output. An empty context is
    # therefore impossible to express -- a task that declares nothing silently inherits its
    # predecessor's answer, unlabelled. Declaring dependencies explicitly is the only way to
    # control what each agent sees. (test_roles.py enforces this.)
    PFCRole(
        name="VMPFC",
        model_setting="VMPFC_MODEL",
        focus="assessing emotional outcomes and risks",
        goal=(
            "Provide emotional insights to aid in decision-making.\n"
            "Assess emotional factors thoroughly and concisely."
        ),
        section_title="Emotional analysis",
        report_order=1,
        task_name="emotional_risk_assessment_task",
        task_intro="Evaluate decisions involving high emotional impact related to '{topic}'.",
        task_actions=(
            "Identify the emotional and psychological factors at play in decisions about '{topic}'.",
            "Assess the potential risks associated with these emotional factors.",
            "Recommend strategies to mitigate risks while addressing emotional concerns.",
        ),
        task_expectation=(
            "A balanced evaluation report detailing emotional factors, associated risks, "
            "and mitigation strategies for '{topic}'."
        ),
        expected_output=(
            "An evaluation report with balanced insights into emotional and rational aspects."
        ),
    ),
    PFCRole(
        name="OFC",
        model_setting="OFC_MODEL",
        focus="balancing rewards against emotional risks",
        goal=(
            "Assess actions based on rewards and manage impulses effectively.\n"
            "Provide a concise evaluation of potential rewards and risks."
        ),
        section_title="Reward evaluation",
        report_order=2,
        task_name="reward_evaluation_task",
        task_intro=(
            "Assess different actions or options based on potential rewards related to '{topic}'."
        ),
        task_actions=(
            "Evaluate the potential rewards associated with each option concerning '{topic}'.",
            "Consider long-term impacts and sustainability of the rewards.",
            "Provide a ranked list of options based on the overall benefit analysis.",
        ),
        task_expectation=(
            "A detailed assessment of options with a focus on long-term rewards and strategic "
            "benefits related to '{topic}'."
        ),
        expected_output=(
            "An assessment document ranking options by potential rewards and strategic value."
        ),
        # Already received VMPFC's output implicitly by running second; now it is declared
        # and labelled instead of arriving as an anonymous blob.
        depends_on=("emotional_risk_assessment_task",),
    ),
    PFCRole(
        name="MPFC",
        model_setting="MPFC_MODEL",
        focus="understanding social dynamics and self-reflection",
        goal=(
            "Analyze social interactions and provide insights for personal growth.\n"
            "Focus on social cognition aspects relevant to the topic."
        ),
        section_title="Social insights",
        report_order=4,
        task_name="social_cognition_task",
        task_intro="Analyze and enhance social dynamics related to '{topic}'.",
        task_actions=(
            "Assess current social interactions and their impact on '{topic}'.",
            "Identify areas for improvement in social interactions.",
            "Propose interventions to enhance social cognition and personal growth.",
        ),
        task_expectation="A strategic plan to improve social interactions related to '{topic}'.",
        expected_output="A strategic plan with interventions for enhancing social cognition.",
        # Was silently receiving OFC's output alone; now both upstream analyses, labelled.
        depends_on=("emotional_risk_assessment_task", "reward_evaluation_task"),
    ),
    PFCRole(
        name="ACC",
        model_setting="ACC_MODEL",
        focus="resolving conflicts between emotional, reward-based, and logical inputs",
        goal=(
            "Resolve conflicts in the decision-making process.\n"
            "Analyze conflicts carefully and provide clear resolution strategies."
        ),
        section_title="Conflict resolution",
        report_order=3,
        task_name="conflict_resolution_task",
        task_intro=(
            "Resolve conflicts between emotional, reward-based, and logical inputs for '{topic}'."
        ),
        task_actions=(
            "Identify sources of conflict within the context of '{topic}'.",
            "Analyze the underlying causes of these conflicts.",
            "Develop and implement conflict resolution strategies.",
        ),
        task_expectation=(
            "A conflict resolution report with actionable steps and outcomes for '{topic}'."
        ),
        expected_output="A report detailing conflict resolution strategies and outcomes.",
        depends_on=("emotional_risk_assessment_task", "reward_evaluation_task"),
    ),
    PFCRole(
        name="DLPFC",
        model_setting="DLPFC_MODEL",
        focus="executive functions like planning and decision-making",
        goal=(
            "Make decisions based on integrated logical, emotional, and social perspectives.\n"
            "Ensure you synthesize information effectively and provide strategic recommendations."
        ),
        section_title="Decision-making analysis",
        report_order=0,
        task_name="complex_decision_making_task",
        task_intro=(
            "Analyze a complex situation involving '{topic}' by integrating insights from "
            "other agents."
        ),
        task_actions=(
            "Conduct a thorough analysis of all available data on '{topic}'.",
            "Synthesize information to identify key trends and insights.",
            "Develop a set of recommendations based on analytical findings.",
        ),
        task_expectation=(
            "A comprehensive report that outlines the situation analysis, key findings, "
            "and strategic recommendations on '{topic}'."
        ),
        expected_output="A comprehensive analytical report with strategic recommendations.",
        # DLPFC is the integrator, so it sees all four upstream analyses. It previously never
        # received the conflict resolution it is supposed to be integrating.
        depends_on=(
            "emotional_risk_assessment_task",
            "reward_evaluation_task",
            "social_cognition_task",
            "conflict_resolution_task",
        ),
    ),
)

BY_NAME: Mapping[RoleName, PFCRole] = MappingProxyType({role.name: role for role in ROLES})
BY_TASK_NAME: Mapping[str, PFCRole] = MappingProxyType({role.task_name: role for role in ROLES})

#: Roles in the order their sections appear in the report, which differs from execution order.
REPORT_ORDER: tuple[PFCRole, ...] = tuple(sorted(ROLES, key=lambda role: role.report_order))


def get_role(name: str) -> PFCRole:
    """Look up a role, failing loudly on an unknown name.

    The if/elif chains this replaced ended in a bare ``else`` returning MPFC's persona, so a
    typo silently produced the wrong agent.
    """
    try:
        return BY_NAME[name]  # type: ignore[index]
    except KeyError:
        known = ", ".join(role.name for role in ROLES)
        raise KeyError(f"Unknown PFC role {name!r}; expected one of: {known}") from None


def execution_order() -> tuple[PFCRole, ...]:
    """Order roles so every task's dependencies run before it (stable topological sort)."""
    ordered: list[PFCRole] = []
    done: set[str] = set()
    remaining = list(ROLES)
    while remaining:
        ready = [role for role in remaining if done.issuperset(role.depends_on)]
        if not ready:
            unresolved = ", ".join(role.task_name for role in remaining)
            raise ValueError(f"Cyclic or unsatisfiable task dependencies among: {unresolved}")
        for role in ready:
            ordered.append(role)
            done.add(role.task_name)
            remaining.remove(role)
    return tuple(ordered)
