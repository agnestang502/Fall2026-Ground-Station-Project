# Not FUNcube-1

These recordings were fetched by the old download script, which used an API filter
(`satellite__norad_cat_id`) that SatNOGS silently ignores, so they are passes of
**other satellites** (KNACKSAT-2, SEEDS, SAUDISAT 1C, OPTISAT, FRONTIERSAT, ...).
They were moved here on 2026-10-18 instead of being deleted. Do not use them for
FUNcube-1 analysis. Each `obs_<id>.json` (where present) names the real satellite
in `tle0` / `norad_cat_id`.
