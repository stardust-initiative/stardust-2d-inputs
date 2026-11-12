ISENTROPIC TENSOR GENERATORS

These two source files are preserved unchanged so their SHA-256 hashes can
be compared with generation metadata. Both contribute to the final dataset:
base monthly calculations supply model-pressure rows at or above 30 hPa;
graded calculations supply rows below 30 hPa. The base generator is therefore
needed even though only the final selected dataset is distributed.

Run --help for options. For example (substitute the desired year):
python -m stardust_2d_inputs.generators.transport.isentropic_sources.era5_iso_fast --mode year --year 2010 --output base_2010
python -m stardust_2d_inputs.generators.transport.isentropic_sources.era5_iso_fast_theta3000_graded --mode year --year 2010 --output graded_2010

These commands perform substantial computation using public ARCO-ERA5 data.
Use the per-month configuration metadata retained in the deposited NetCDF
when selecting the generation settings. Do not load untrusted restart pickle
files. The optional --mode validate path additionally needs the reference
implementation named in the source documentation; ordinary generation does not.

For the four model-pressure components retained by the public dataset, align
base and graded files on identical model_pressure and latitude coordinates,
then select graded values where model_pressure < 3000 Pa and base values
elsewhere. The boundary at exactly 3000 Pa belongs to the base calculation.
Preserve missing values; do not replace them or extrapolate during assembly.
Concatenate each calendar year's twelve months, retain year and month as
coordinates, and retain both generator configurations and hashes. The four
fields and lossless extraction are documented in the parent transport README.
The full diagnostic composite also contains theta-grid fields; these are not
needed for the published four-component transport input.
