"""The evaluation scripts themselves: quick checks that they run and measure what they claim."""
from dimgrad.frontend.parser import parse
from dimgrad.glue.fd import TOLERANCE
from dimgrad.units.checker import check
from evaluation import fd_check, methods, mutate
from evaluation.common import benchmark_names, load_grids, load_source


def test_methods_table_is_as_derived_by_hand():
    for method, description, step, expected, expected_unit in methods.CASES:
        outcome, detail = methods.run_case(step, expected_unit)
        assert outcome == expected, (method, description, detail)


def test_finite_difference_check_with_interpreter():
    rows = fd_check.gradient_errors("kinetic_energy", load_grids(), use_c=False, workdir=None)
    assert [r["gradient"] for r in rows] == ["energy_grad_v", "energy_grad_mass"]
    assert all(r["points"] == 9 and r["max_rel_error"] < TOLERANCE for r in rows)


def test_mutants_of_the_sample_are_all_rejected():
    rows = mutate.run_benchmark("kinetic_energy")
    assert rows and all(r["detected"] for r in rows)
    assert {r["category"] for r in rows} == {2, 3}


def test_every_mutation_category_is_exercised():
    categories = set()
    for bench in benchmark_names():
        program = parse(load_source(bench))
        categories |= {m.category for m in mutate.generate(program, check(program))}
    assert categories == set(mutate.CATEGORIES)


def test_mutation_changes_the_program():
    rows = mutate.run_benchmark("newton_root")
    wrong_variable = [r for r in rows if "g_grad_a" in r["mutation"]]
    assert wrong_variable and all(r["detected"] for r in wrong_variable)
