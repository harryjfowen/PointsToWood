"""Region group definitions for training and distillation.

Maps --region argument to the set of file prefixes it includes.
File prefixes are the first 3 characters of each .ply filename (ISO 3166-1 alpha-3).

  global  → all files in the pool (no filter)
  eu      → European country prefixes only
  any other region (fin, pol, spa, cam ...) → that prefix only

Add new regions by extending REGION_GROUPS.
"""

# Standard ISO 3166-1 alpha-3 codes for European countries
_EU_PREFIXES = [
    'aut', 'bel', 'bgr', 'hrv', 'cyp', 'cze', 'dnk', 'est', 'fin',
    'fra', 'deu', 'grc', 'hun', 'irl', 'ita', 'lva', 'ltu', 'lux',
    'mlt', 'nld', 'nor', 'pol', 'prt', 'rou', 'svk', 'svn', 'esp',
    'swe', 'che', 'gbr', 'eng', 'sco', 'wal', 'alb', 'bih', 'mkd', 'mne', 'srb', 'isl',
]

REGION_GROUPS: dict = {
    'global': None,      # all files — no filter
    'eu':     _EU_PREFIXES,
}


def get_prefixes(region: str) -> list | None:
    """Return list of file prefixes for a region, or None for no filter (global).

    For known multi-prefix regions (global, eu) uses REGION_GROUPS.
    For anything else (fin, pol, spa, cam ...) uses the region name itself as the prefix.
    """
    region = region.lower().strip()
    if region in REGION_GROUPS:
        return REGION_GROUPS[region]  # None = all files
    return [region]  # single-prefix region e.g. 'fin' -> ['fin']


def filter_ply_files(files: list, region: str) -> list:
    """Filter a list of .ply file paths to those matching the region's prefixes."""
    import os
    prefixes = get_prefixes(region)
    if prefixes is None:
        return files
    return [f for f in files if os.path.basename(f).lower()[:3] in prefixes]
