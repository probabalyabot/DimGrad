from dimgrad.glue.cli import main
from evaluation.common import BENCH_DIR

SAMPLE = str(BENCH_DIR / "kinetic_energy.dg")


def test_check_prints_the_report(capsys):
    assert main(["check", SAMPLE]) == 0
    out = capsys.readouterr().out
    assert "fn energy(mass: kg, v: m/s) -> J" in out
    assert "grad energy wrt v : kg·m/s" in out


def test_check_reports_errors(tmp_path, capsys):
    path = tmp_path / "bad.dg"
    path.write_text("unit m, s;\nfn f(x: m, t: s) -> m { return x + t; }\n", encoding="utf-8")
    assert main(["check", str(path)]) == 1
    assert "bad.dg:2:34: error: cannot add m and s" in capsys.readouterr().err


def test_missing_file(capsys):
    assert main(["check", "no_such_file.dg"]) == 2
    assert "cannot read" in capsys.readouterr().err


def test_compile_writes_c(tmp_path, capsys):
    out_path = tmp_path / "out" / "ke.c"
    assert main(["compile", SAMPLE, "-o", str(out_path), "--emit-tac", "--stats", "--harness"]) == 0
    c_text = out_path.read_text(encoding="utf-8")
    assert "double energy_grad_v(double mass, double v, double *value)" in c_text
    assert "int main(void)" in c_text
    listing = capsys.readouterr().out
    assert "d_t3 = t2 + t2" in listing
    assert "energy_grad_v" in listing and "12" in listing


def test_compile_to_stdout(capsys):
    assert main(["compile", SAMPLE, "--no-opt"]) == 0
    assert "double energy(double mass, double v)" in capsys.readouterr().out


def test_run(capsys):
    assert main(["run", SAMPLE, "energy_grad_v", "2", "3"]) == 0
    out = capsys.readouterr().out
    assert "d energy / d v = 6.0 kg·m/s" in out
    assert "energy = 9.0 kg·m^2/s^2" in out
    assert main(["run", SAMPLE, "energy", "2"]) == 2
