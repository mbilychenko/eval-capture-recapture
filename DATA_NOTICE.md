# Data notice

## CUAD (contracts and clause labels)
`data/sample.json` and the cached agent outputs in `outputs/` contain contract text and labels from the **Contract Understanding Atticus Dataset (CUAD) v1**, by The Atticus Project. It is licensed under **Creative Commons Attribution 4.0 (CC BY 4.0)**: https://creativecommons.org/licenses/by/4.0/

> Hendrycks, D., Burns, C., Chen, A., & Ball, S. (2021). *CUAD: An Expert-Annotated NLP Dataset for Legal Contract Review.* NeurIPS Datasets and Benchmarks. https://arxiv.org/abs/2103.06268. Data: https://zenodo.org/records/4595826

Changes made here:
- a seeded subset of 150 contracts and 10 clause categories was taken;
- overlapping labels were merged;
- AI agents' quotes from the contracts were added (`outputs/raw/`).

## Forest counts (Act 1 of the article)
`scripts/05_forests.py` contains only aggregate counts from the author's own analysis of:
- **National Forest Inventory 2023**, Forestry Commission (Open Government Licence v3.0);
- **OS Open Names**, Ordnance Survey (Open Government Licence v3.0; contains OS data © Crown copyright and database right);
- **OpenStreetMap** © OpenStreetMap contributors (ODbL), https://www.openstreetmap.org/copyright.
