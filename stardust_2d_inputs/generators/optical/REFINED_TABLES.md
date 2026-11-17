# Refined angular quadrature and extended radius coverage

The `optical.refined` generator refines the hemispheric scattering fractions
and extends an existing silica or calcite RRTMG table. It uses the same
refractive indices, Mie efficiencies, spectral grids and band weighting as the
material generators. The legacy material generators remain available for
existing table versions.

```bash
python -m stardust_2d_inputs.generators.optical.refined \
  --material silica --parent stardust_particles_silica_climlab.nc \
  --output silica_refined_extended.nc --work-dir silica-refinement \
  --workers 4
```

For calcite, use `--material calcite` and the calcite parent/output paths.
Install the package's `[generators]` extra. Each worker processes an independent
radius; choose workers according to available memory and cores. Set
`OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1` to avoid nested thread pools.
The output and working directory must not already exist, preventing accidental
overwrite or reuse of cached rows from different inputs.

The default appends 17 rows to the original 51-point log-radius grid, reaching
3.734303536687189 micrometres. It does not redistribute the original grid.
Every angular row is recomputed; original non-angular rows are copied exactly.
Changing only `nr` in the legacy generator would change the radius spacing and
is not this procedure.

For a phase function P normalized to integrate to 2 over scattering cosine,
write P(u) = sum a_l P_l(u), and J_l = integral_0^1 P_l(u) du. The backward
hemisphere fraction for incident cosine mu is

    beta(mu) = (1/2) sum a_l P_l(-mu) J_l.

The existing table's diffuse fraction is the uniform incident-cosine average,

    beta_bar = (1/2) sum a_l (-1)^l J_l^2.

This preserves the table's angular definitions. It does not introduce a new
Lambertian flux-weighting convention. A degree-384 Legendre expansion and
512/1024-point Gauss-Legendre projection replace the uniform-grid interpolation
that could yield negative hemisphere fractions for larger size parameters.
Both orders must agree at rtol=1e-9, atol=1e-25 for the band coefficients; the
1024-point result is retained. No clipping or phase-normalization repair occurs.
Size parameters outside 0 < x < 140 require separate qualification and are
rejected. Phase normalization errors >=1e-5 are also rejected.

The assembly checks finiteness, nonnegative absorption/scattering, extinction
>= scattering, |asymmetry-weighted scattering| <= scattering, and hemispheric
coefficients <= scattering. It verifies recomputed non-angular parent rows
before retaining the parent values exactly.

## Provenance and versioning

The generated table records parent, generator and angular-module hashes and
settings. `provenance.json` additionally records the output hash. Work directories
contain both quadrature orders' per-radius arrays. Retain these alongside the
input versions used to generate a released table.

Registry versions `optical_silica/v2` and `optical_calcite/v2` identify the already
qualified immutable artifacts, not an in-place edit of v1. Their existing
attributes preserve the original qualification scripts' hashes. The maintained
implementation uses a subclass instead of a process-global monkeypatch.
Integration validation compared every assembled variable byte-for-byte with
both qualified tables, and independently recomputed each material's largest
radius row with exact agreement. Angular tests independently integrate the
backward hemisphere and check the diffuse identity and quadrature convergence.
A complete fresh 68-row calculation was not repeated during integration.

NetCDF creation timestamps and provenance metadata differ on regeneration, so
scientific array agreement is distinct from whole-file byte identity. The
registry pins the original released artifact's complete-file SHA-256.
Existing release manifests remain unchanged until the new artifacts are
published and the coordinated input release is selected.
