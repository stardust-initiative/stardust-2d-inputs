"""Independent angular identities and safe radius extension."""

import numpy as np
import pytest

pytest.importorskip("miepython")
pytest.importorskip("scipy")
from scipy.special import roots_legendre  # noqa: E402
import miepython  # noqa: E402
from stardust_2d_inputs.generators.optical._angular import beta_calc  # noqa: E402
from stardust_2d_inputs.generators.optical.refined import extended_grid  # noqa: E402


@pytest.mark.parametrize("x", [0.1, 2.0, 30.0, 117.0])
def test_angular_convergence_and_backward_integral(x):
    m = 1.5 + 0.01j
    mu = np.array([0.0, 0.2, 0.7, 1.0])
    a = beta_calc(m, x, 512, mu)
    b = beta_calc(m, x, 1024, mu)
    np.testing.assert_allclose(a[:2], b[:2], atol=1e-9, rtol=1e-9)
    np.testing.assert_allclose(a[2], b[2], atol=1e-9, rtol=1e-9)
    z, w = roots_legendre(1024)
    direct = 0.25 * np.sum(
        w * miepython.i_unpolarized(np.conjugate(m), x, (z - 1) / 2, "4pi")
    )
    np.testing.assert_allclose(b[0], direct, atol=1e-9)
    np.testing.assert_allclose(b[2][0], 0.5, atol=1e-9)
    np.testing.assert_allclose(b[2][-1], b[0], atol=1e-12)
    assert np.all((b[2] >= 0) & (b[2] <= 1))


def test_diffuse_is_uniform_incident_cosine_average():
    z, w = roots_legendre(200)
    _, diffuse, per_mu = beta_calc(1.5 + 0.01j, 12.0, 1024, (z + 1) / 2)
    np.testing.assert_allclose(diffuse, np.sum(w * per_mu) / 2, atol=1e-12)


def test_grid_preserves_old_samples_and_spacing():
    old = np.linspace(np.log(50e-9), np.log(1250e-9), 51)
    new = extended_grid(old)
    assert new[:51].tobytes() == old.tobytes()
    assert len(new) == 68
    np.testing.assert_allclose(np.diff(new), np.diff(old)[0], atol=1e-14)
    np.testing.assert_allclose(np.exp(new[-1]), 3.734303536687189e-6)


@pytest.mark.parametrize("grid", [[1], [1, 1], [1, 2, 4], [1, np.nan]])
def test_invalid_grid(grid):
    with pytest.raises(ValueError):
        extended_grid(grid)


def test_angular_limits():
    with pytest.raises(ValueError):
        beta_calc(1.5, 140)
    with pytest.raises(ValueError):
        beta_calc(1.5, 1.0, 384)
    with pytest.raises(ValueError):
        beta_calc(1.5, 1.0, mu_vect=[1.1])


@pytest.fixture
def parent_table(tmp_path):
    import netCDF4 as nc
    from stardust_2d_inputs.generators.optical.refined import FIELDS, ANGULAR

    path = tmp_path / "parent.nc"
    with nc.Dataset(path, "w") as ds:
        ds.createDimension("mu_samples", 2)
        ds.createDimension("band", 1)
        ds.createDimension("coszen", 1)
        ds.createVariable("mu_samples", "f8", ("mu_samples",))[:] = [-17.0, -16.0]
        ds.createVariable("coszen", "f8", ("coszen",))[:] = [0.5]
        for key in FIELDS:
            dims = (
                ("mu_samples", "band", "coszen")
                if key == ANGULAR[1]
                else ("mu_samples", "band")
            )
            ds.createVariable(key, "f8", dims)[:] = 0.2 if key in ANGULAR else 1.0
    return path


def test_assembly_preserves_parent_and_refines_angular(parent_table, tmp_path):
    import netCDF4 as nc
    from stardust_2d_inputs.generators.optical.refined import FIELDS, ANGULAR, assemble

    rows = {
        key: np.full(
            (3, 1, 1) if key == ANGULAR[1] else (3, 1), 0.3 if key in ANGULAR else 1.0
        )
        for key in FIELDS
    }
    # Within the permitted recomputation tolerance, but parent must be copied exactly.
    rows["babs_lw"][:2] += 1e-12
    out = tmp_path / "refined.nc"
    assemble(parent_table, out, np.array([-17.0, -16.0, -15.0]), rows, {"test": True})
    with nc.Dataset(parent_table) as old, nc.Dataset(out) as new:
        for key in FIELDS:
            if key not in ANGULAR:
                assert old[key][:].tobytes() == new[key][:2].tobytes()
            else:
                np.testing.assert_array_equal(new[key][:], rows[key])
        np.testing.assert_array_equal(old["coszen"][:], new["coszen"][:])
    with pytest.raises(FileExistsError):
        assemble(parent_table, out, np.array([-17.0, -16.0, -15.0]), rows, {})


def test_assembly_rejects_unphysical_rows_before_writing(parent_table, tmp_path):
    from stardust_2d_inputs.generators.optical.refined import FIELDS, ANGULAR, assemble

    rows = {key: np.ones((3, 1, 1) if key == ANGULAR[1] else (3, 1)) for key in FIELDS}
    rows[ANGULAR[0]][2] = 2.0
    out = tmp_path / "invalid.nc"
    with pytest.raises(ValueError, match="bounds"):
        assemble(parent_table, out, np.array([-17.0, -16.0, -15.0]), rows, {})
    assert not out.exists()
