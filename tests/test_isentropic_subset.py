import numpy as np
import pytest
import xarray as xr

from stardust_2d_inputs.generators.transport.isentropic_subset import (
    DIMS,
    FIELDS,
    create_subset,
)


def test_lossless_subset_preserves_grid_missing_values_and_metadata(tmp_path):
    coords = dict(
        year=[2008, 2009],
        month=[1, 2],
        model_pressure=[100, 3000],
        latitude=[-90, 0, 90],
    )
    a = np.arange(24, dtype=np.float64).reshape(2, 2, 2, 3)
    a[0, 0, 0, 0] = np.nan
    a[0, 0, 0, 1] = -0.0
    source = xr.Dataset({name: (DIMS, a.copy()) for name in FIELDS}, coords=coords)
    source["unused_diagnostic"] = (DIMS, a * 2)
    source["configuration_json"] = (
        ("year", "month"),
        np.full((2, 2), '{"step_minutes":30}'),
    )
    src, dst = tmp_path / "full.nc", tmp_path / "lean.nc"
    source.to_netcdf(src)
    record = create_subset(src, dst)
    with xr.open_dataset(dst) as lean:
        assert "unused_diagnostic" not in lean
        for name in list(FIELDS) + ["configuration_json"]:
            xr.testing.assert_identical(source[name], lean[name])
        assert lean.attrs["parent_content_hash"] == record["parent_content_hash"]
    with pytest.raises(FileExistsError):
        create_subset(src, dst)
