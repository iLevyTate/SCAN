"""The single source of truth for SCAN's prefrontal-cortex roles.

Each region owns exactly one agent and exactly one task, so one table describes both. Everything
else in the package derives from ``ROLES``: the agent personas (:mod:`scan.scan_agents`), the crew
tasks and their dependency wiring (:mod:`scan.scan_tasks`), the execution order and the report
layout (:mod:`scan.report`).

Adding a region means adding one :class:`PFCRole` here plus its ``*_MODEL`` field in
:class:`scan.config.Settings`; a test asserts those two stay in step.

The briefs below are deliberately region-specific. Five prompts that all said "conduct a thorough
analysis" produced five interchangeable essays, which defeats the point of modelling distinct
regions at all: each one now asks for the kind of reasoning its region actually does.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Literal, TypeAlias

if TYPE_CHECKING:
    from collections.abc import Mapping

RoleName: TypeAlias = Literal["DLPFC", "VMPFC", "OFC", "ACC", "MPFC"]


@dataclass(frozen=True, slots=True)
class OutputContract:
    """What a task must return.

    Rendered into ``Task.expected_output`` and nowhere else. crewai injects that field into the
    prompt itself, so the old "Expected Output:" block inside each description handed the model
    two competing specifications of the same deliverable, worded differently.

    The word budget is the only length instruction in the system. Descriptions used to ask for
    "comprehensive" and "thorough" output while the agent goals asked for "concise" -- a direct
    contradiction that made length a lottery, with MAX_TOKENS as the only real bound.
    """

    summary: str
    sections: tuple[str, ...]
    min_words: int
    max_words: int

    def render(self) -> str:
        labels = ", ".join(f"**{section}**" for section in self.sections)
        return (
            f"{self.summary} Use exactly these bold subsection labels, in this order: {labels}. "
            f"Write {self.min_words}-{self.max_words} words in total. Do not open with a title "
            "or heading -- the caller adds one. Do not restate the task or quote the supplied "
            "context back at length."
        )


@dataclass(frozen=True, slots=True)
class PFCRole:
    """One prefrontal-cortex region: its agent persona and the task it performs."""

    name: RoleName
    #: Attribute name on ``Settings`` -- never a resolved model string. The value must be read
    #: lazily via ``getattr(settings, role.model_setting)``, because callers (and tests) override
    #: settings after import.
    model_setting: str
    #: Completes "You contribute {focus}".
    focus: str
    goal: str
    #: Report heading, rendered as "{section_title} ({name})".
    section_title: str
    #: Position in the report, which is not the order tasks execute in.
    report_order: int
    task_name: str
    #: Opening sentence of the task description.
    mandate: str
    #: The analysis moves this region performs. These are what make the five prompts different.
    steps: tuple[str, ...]
    contract: OutputContract
    #: ``task_name``s whose output is fed to this task as crewai ``context``.
    depends_on: tuple[str, ...] = ()
    #: The integrating region, promoted to the top of the report.
    is_summary: bool = False

    @property
    def expected_output(self) -> str:
        return self.contract.render()

    def backstory(self, topic: str) -> str:
        return (
            f"You are the {self.name}, a region of the prefrontal cortex. You contribute "
            f"{self.focus} to the decision about '{topic}'. Stay in your role: the other "
            "regions cover the other angles, and their analyses are supplied to you where "
            "they are relevant. Follow the output contract exactly."
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
        steps = "\n".join(
            f"{index}. {step.format(topic=topic)}" for index, step in enumerate(self.steps, 1)
        )
        parts = [self.mandate.format(topic=topic)]
        legend = self.context_legend()
        if legend:
            parts.append(legend)
        parts += ["Work through this:", steps]
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
        focus="affective forecasting -- how each option will actually feel to live with",
        goal=(
            "Surface the emotional stakes a purely analytical account would miss.\n"
            "Separate discomfort that tracks a real risk from discomfort that tracks "
            "unfamiliarity."
        ),
        section_title="Emotional analysis",
        report_order=1,
        task_name="emotional_risk_assessment_task",
        mandate=(
            "Assess how the options around '{topic}' would feel to live with, not just how "
            "they score."
        ),
        steps=(
            "Say what is actually at stake emotionally here, in plain language.",
            "Describe how each option would feel the morning after committing to it.",
            "Separate the apprehension that is tracking a genuine risk from the apprehension "
            "that is only tracking unfamiliarity, and say how you can tell which is which in "
            "this case.",
            "Name the one outcome that would be hardest to live with five years from now.",
        ),
        contract=OutputContract(
            summary="An account of the emotional stakes and where they are and are not reliable.",
            sections=(
                "Stakes",
                "How each option would feel",
                "Signal or noise",
                "Hardest to live with",
            ),
            min_words=200,
            max_words=350,
        ),
    ),
    PFCRole(
        name="OFC",
        model_setting="OFC_MODEL",
        focus="valuing the options against each other, and against time",
        goal=(
            "Rank the available options by expected value.\n"
            "Make the time horizon explicit, and state what evidence would overturn the ranking."
        ),
        section_title="Reward evaluation",
        report_order=2,
        task_name="reward_evaluation_task",
        mandate="Value the options for '{topic}' against one another, across time.",
        steps=(
            "Enumerate two to four concrete options, including the option of doing nothing.",
            "Score each option's expected value at one month, at one year, and at five years.",
            "Identify where the ranking changes between those horizons, and say what drives "
            "the change.",
            "For your top-ranked option, state the specific observable signal that should make "
            "you abandon it.",
        ),
        contract=OutputContract(
            summary="A ranking of concrete options with the time horizon made explicit.",
            sections=("Options", "Value by horizon", "Where the ranking flips", "Abandon signal"),
            min_words=200,
            max_words=350,
        ),
        # Already received VMPFC's output implicitly by running second; now it is declared
        # and labelled instead of arriving as an anonymous blob.
        depends_on=("emotional_risk_assessment_task",),
    ),
    PFCRole(
        name="MPFC",
        model_setting="MPFC_MODEL",
        focus=(
            "modelling how other people will read this decision, and whether it fits who the "
            "decider takes themselves to be"
        ),
        goal=(
            "Represent the people affected as they would represent themselves.\n"
            "Surface where their view of this decision differs from the decider's."
        ),
        section_title="Social insights",
        report_order=4,
        task_name="social_cognition_task",
        mandate="Work out how the people around '{topic}' will understand this decision.",
        steps=(
            "List everyone materially affected, and say how each one is affected.",
            "Write one sentence in each person's own voice about this decision.",
            "Name one belief each of them holds that the decider does not, and say what follows "
            "if they turn out to be right.",
            "Say whether this decision is consistent with the kind of person the decider takes "
            "themselves to be.",
        ),
        contract=OutputContract(
            summary="An account of how this decision reads to the people it touches.",
            sections=(
                "Who is affected",
                "In their words",
                "Where they disagree",
                "Self-consistency",
            ),
            min_words=200,
            max_words=350,
        ),
        # Was silently receiving OFC's output alone; now both upstream analyses, labelled.
        depends_on=("emotional_risk_assessment_task", "reward_evaluation_task"),
    ),
    PFCRole(
        name="ACC",
        model_setting="ACC_MODEL",
        focus="detecting where the other analyses genuinely disagree, and adjudicating between them",
        goal=(
            "Find the real contradictions between the analyses supplied to you.\n"
            "Decide which side should win, and state what would change that."
        ),
        section_title="Conflict resolution",
        report_order=3,
        task_name="conflict_resolution_task",
        mandate="Adjudicate the disagreements between the analyses of '{topic}' supplied to you.",
        steps=(
            "Quote the specific sentences from the analyses above that contradict each other. "
            "Quote them; do not paraphrase.",
            "Classify each conflict: a disagreement about values, about evidence, or about the "
            "time horizon being used.",
            "For each one, name which side should win. Do not split the difference.",
            "State the fact that would have to be true for the other side to win instead.",
        ),
        contract=OutputContract(
            summary="A resolution of the actual conflicts between the upstream analyses.",
            sections=(
                "Conflicts",
                "What kind of disagreement",
                "Which side wins",
                "What would change it",
            ),
            min_words=200,
            max_words=400,
        ),
        depends_on=("emotional_risk_assessment_task", "reward_evaluation_task"),
    ),
    PFCRole(
        name="DLPFC",
        model_setting="DLPFC_MODEL",
        focus=(
            "integrating the other regions' analyses into a decision, and keeping track of what "
            "that decision rests on"
        ),
        goal=(
            "Reach a decision the other regions' analyses actually support.\n"
            "Attribute your reasoning, and name what you are trading away to get it."
        ),
        section_title="Recommendation",
        report_order=0,
        task_name="complex_decision_making_task",
        mandate="Decide what to do about '{topic}', using the analyses supplied to you.",
        steps=(
            "State the recommendation in one sentence.",
            "Give the three reasons it rests on, attributing each to the region that raised it.",
            "State the strongest argument against the recommendation, as fairly as you can, and "
            "say why you are accepting that cost anyway.",
            "Name the single largest thing you are still uncertain about.",
            "Give one concrete action to take this week, and the observable check that would "
            "tell you it is working.",
        ),
        contract=OutputContract(
            summary="A decision, what it rests on, and what it costs.",
            sections=(
                "Recommendation",
                "Why",
                "Strongest case against",
                "Biggest uncertainty",
                "First step",
            ),
            min_words=250,
            max_words=400,
        ),
        # DLPFC is the integrator, so it sees all four upstream analyses. It previously never
        # received the conflict resolution it is supposed to be integrating.
        depends_on=(
            "emotional_risk_assessment_task",
            "reward_evaluation_task",
            "social_cognition_task",
            "conflict_resolution_task",
        ),
        is_summary=True,
    ),
)

BY_NAME: Mapping[RoleName, PFCRole] = MappingProxyType({role.name: role for role in ROLES})
BY_TASK_NAME: Mapping[str, PFCRole] = MappingProxyType({role.task_name: role for role in ROLES})

#: Roles in the order their sections appear in the report, which differs from execution order.
REPORT_ORDER: tuple[PFCRole, ...] = tuple(sorted(ROLES, key=lambda role: role.report_order))

#: The integrating region, rendered as the report's opening summary.
SUMMARY_ROLE: PFCRole = next(role for role in ROLES if role.is_summary)


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
