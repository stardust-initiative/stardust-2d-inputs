"""Losslessly select the four fields used by mean-first reduced transport.

Run with ``python -m stardust_2d_inputs.generators.transport.isentropic_subset
SOURCE.nc OUTPUT.nc``. The output retains every year, month, model pressure
and latitude, original floating-point precision, missing values and per-month
generation metadata. No tensor preparation or climatological averaging occurs.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import xarray as xr

FIELDS = tuple(
    "iso_mean_first_on_model_" + name
    for name in ("D_yy", "D_yp_tilt", "D_pp_tilt", "D_pp_theta")
)
METADATA = (
    "configuration_json",
    "theta3000_patch_configuration_json",
    "script_sha256",
    "software_version",
)
DIMS = ("year", "month", "model_pressure", "latitude")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def create_subset(source, output):
    """Write a new lossless subset and verify all selected values and metadata."""
    source, output = Path(source), Path(output)
    if output.exists():
        raise FileExistsError(output)
    parent_hash = sha256(source)
    with xr.open_dataset(source) as full:
        for name in FIELDS:
            if full[name].dims != DIMS:
                raise ValueError(f"Unexpected dimensions for {name}: {full[name].dims}")
        metadata = [name for name in METADATA if name in full]
        result = full[list(FIELDS) + metadata].copy()
        result.attrs = dict(full.attrs)
        result.attrs.update(
            title="ERA5 mean-first reduced diffusion components, monthly",
            parent_content_hash="sha256:" + parent_hash,
            subset_generator="stardust_2d_inputs.generators.transport.isentropic_subset",
            subset_generator_sha256=sha256(__file__),
            retained_fields=json.dumps(FIELDS),
            selection_recipe="Dyy=D_yy; Dyp=Dpy=D_yp_tilt; Dpp=D_pp_tilt+D_pp_theta",
            processing="Field selection and lossless compression only; no thinning, averaging, interpolation or precision reduction.",
            usage="Select year/month; retain the saved products. Polar continuation, tensor bounds and model-grid climatology are consumer operations, not applied to this file.",
        )
        encoding = {name: dict(zlib=True, complevel=4, shuffle=True) for name in FIELDS}
        result.to_netcdf(output, mode="w", engine="netcdf4", encoding=encoding)
        with xr.open_dataset(output) as restored:
            for name in list(FIELDS) + metadata + list(result.coords):
                xr.testing.assert_identical(full[name], restored[name])
                if np.issubdtype(full[name].dtype, np.floating):
                    original, actual = full[name].values, restored[name].values
                    # Also retain signed zero; NaN payload bits are not scientific data.
                    np.testing.assert_array_equal(
                        np.signbit(original[np.isfinite(original)]),
                        np.signbit(actual[np.isfinite(actual)]),
                    )
    return dict(
        parent_content_hash="sha256:" + parent_hash,
        content_hash="sha256:" + sha256(output),
        filename=output.name,
        size_bytes=output.stat().st_size,
        exact_values_coordinates_and_metadata=True,
        fields=list(FIELDS),
        metadata=metadata,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(json.dumps(create_subset(args.source, args.output), indent=2))


if __name__ == "__main__":
    main()
