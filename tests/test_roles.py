"""Properties of the role table that line coverage cannot see.

Before scan/roles.py existed, the personas lived in two if/elif chains that both ended in a bare
`else` returning MPFC's text. Deleting the ACC branch kept the entire suite green, because only
DLPFC's strings were ever asserted. These tests close that class of hole.
"""

import typing

import pytest

from scan.config import Settings
from scan.roles import (
    BY_NAME,
    BY_TASK_NAME,
    REPORT_ORDER,
    ROLES,
    RoleName,
    execution_order,
    get_role,
)

ROLE_NAMES = [role.name for role in ROLES]


def test_role_literal_matches_the_table():
    assert set(typing.get_args(RoleName)) == set(ROLE_NAMES)


def test_every_role_has_a_model_setting_on_settings():
    # A new role cannot be added without also adding its *_MODEL env-var contract.
    for role in ROLES:
        assert role.model_setting in Settings.model_fields


@pytest.mark.parametrize(
    "field",
    ["name", "model_setting", "goal", "section_title", "task_name", "expected_output"],
)
def test_field_is_distinct_across_roles(field):
    values = [getattr(role, field) for role in ROLES]

    assert len(set(values)) == len(ROLES), f"duplicate {field} across roles"


@pytest.mark.parametrize("role", ROLES, ids=lambda role: role.name)
def test_role_text_is_populated_and_stripped(role):
    for field in ("focus", "goal", "section_title", "expected_output"):
        value = getattr(role, field)
        assert value, f"{role.name}.{field} is empty"
        assert value == value.strip(), f"{role.name}.{field} has stray whitespace"


@pytest.mark.parametrize("role", ROLES, ids=lambda role: role.name)
def test_backstory_and_description_mention_the_topic(role):
    assert "a rescue dog" in role.backstory("a rescue dog")
    assert role.name in role.backstory("a rescue dog")
    assert "a rescue dog" in role.description("a rescue dog")


@pytest.mark.parametrize("role", ROLES, ids=lambda role: role.name)
def test_description_renders_every_action(role):
    description = role.description("a topic")

    for action in role.task_actions:
        assert action.format(topic="a topic") in description


def test_lookups_cover_every_role():
    assert set(BY_NAME) == set(ROLE_NAMES)
    assert set(BY_TASK_NAME) == {role.task_name for role in ROLES}


def test_get_role_returns_the_matching_role():
    for role in ROLES:
        assert get_role(role.name) is role


def test_get_role_raises_on_an_unknown_name():
    # Regression: the if/elif chains fell through to MPFC's persona for any unknown role.
    with pytest.raises(KeyError, match="Unknown PFC role"):
        get_role("HIPPOCAMPUS")


def test_dependencies_reference_real_tasks():
    task_names = {role.task_name for role in ROLES}

    for role in ROLES:
        assert set(role.depends_on) <= task_names


def test_every_role_after_the_first_declares_its_inputs():
    # crewai's Crew._get_context uses task.context only when truthy and otherwise falls back
    # to the *previous* task's raw output, so an empty context cannot be expressed. A role
    # that declares nothing silently inherits its predecessor's answer with no label. Only
    # the first task in the execution order can legitimately run with no dependencies.
    ordered = execution_order()

    assert ordered[0].depends_on == ()
    for role in ordered[1:]:
        assert role.depends_on, (
            f"{role.name} declares no dependencies, so crewai will silently feed it the "
            f"preceding task's raw output instead of nothing"
        )


def test_the_synthesis_role_depends_on_every_other_role():
    synthesis = REPORT_ORDER[0]

    assert set(synthesis.depends_on) == {role.task_name for role in ROLES if role is not synthesis}


@pytest.mark.parametrize("role", [r for r in ROLES if r.depends_on], ids=lambda role: role.name)
def test_context_legend_names_each_dependency_in_order(role):
    # The legend is generated from depends_on -- the same field that builds Task.context --
    # so it cannot disagree with what crewai actually splices in.
    legend = role.context_legend()
    positions = [legend.index(BY_TASK_NAME[dep].section_title) for dep in role.depends_on]

    assert positions == sorted(positions)
    for dependency in role.depends_on:
        assert BY_TASK_NAME[dependency].name in legend
    assert legend in role.description("a topic")


def test_roles_without_dependencies_have_no_legend():
    assert execution_order()[0].context_legend() == ""


def test_execution_order_is_a_valid_topological_sort():
    ordered = execution_order()
    positions = {role.task_name: index for index, role in enumerate(ordered)}

    assert len(ordered) == len(ROLES)
    for role in ordered:
        for dependency in role.depends_on:
            assert positions[dependency] < positions[role.task_name]


def test_report_order_is_a_permutation_of_the_roles():
    assert sorted(role.name for role in REPORT_ORDER) == sorted(ROLE_NAMES)
    assert [role.report_order for role in REPORT_ORDER] == list(range(len(ROLES)))


def test_the_synthesis_role_is_reported_first():
    # DLPFC integrates the others, so it leads the report even though it runs last.
    assert REPORT_ORDER[0].name == "DLPFC"
    assert execution_order()[-1].name == "DLPFC"


def test_roles_are_immutable():
    with pytest.raises(AttributeError):
        ROLES[0].goal = "mutated"  # type: ignore[misc]
