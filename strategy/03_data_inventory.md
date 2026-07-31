# Data Inventory

## Sources (external to repo, configured in configs/paths.yaml)

| Key | Path | Content |
|-----|------|---------|
| database | /media/guru/Elements/backup_11_2026/Worspace_Mahish/Phd/Database/database_24_06 | raw: 17 alloys x 86 conditions, each with OM/, Property/, XRD/ |
| training_data | /media/guru/Elements/backup_11_2026/Worspace_Mahish/Phd/Database/training_data | precomputed Acta-era descriptors, naming {alloy}_{T}_{v}_* |
| labels_xlsx | ./data_labels.xlsx | composition wt% per alloy (13 element columns) |
| genai_repo | ~/Worspace_Mahish/Phd/02_Microstrucure_Reconstruction/2D_reconstruction/GenAI/encoder-decoder-micro-reconstruction | ViT-FMDiT-1280 encoder code; checkpoint pending from remote |
| sampling_repo | ~/Worspace_Mahish/Phd/02_Microstrucure_Reconstruction/2D_reconstruction/Sampling | DREAM.3D copula pipeline + MTEX + odf_harmonics (116 files) |
| characterization_repo | ~/Worspace_Mahish/Phd/01_Microstructure_Characterization | binarization, n-point, Gram, GSH scripts |
| dimred_repo | ~/Worspace_Mahish/Phd/04_Dimensionality_Reduction | PCA/Isomap/AE batch scripts |

## Alloy x condition enumeration (from database_24_06)

Condition folder name encodes {T_ext degC}_{v_ext mm/s}.

| Alloy | # cond | Conditions (T_v) |
|-------|--------|------------------|
| AZ31_extruded | 9 | 200_0.6, 250_0.6, 300_0.6, 300_2.4, 350_0.6, 400_0.6, 450_0.6, 500_0.6, 500_6 |
| AZ31_extruded_heattreated | 8 | (HT twin of above minus 500_6) EXCLUDED |
| ME21_extruded | 19 | 300_{1.4,2.8,5.5,7.5}, 350_{0.75,1.4,2.8,5.5,7.5}, 400_{1.4,2.8,5.5,7.5}, 450_{0.75,1.4,2.8,5.5,7.5} |
| Mg-10Gd_extruded | 9 | {350,400,450}_{0.5,1,2} |
| Mg-10Gd-0.5Mn_extruded | 4 | 350_0.5, 450_{0.5,1,2} |
| Mg-10Gd-1Mn_extruded | 4 | 350_0.5, 450_{0.5,1,2} |
| Mg-2Gd_extruded | 9 | {350,400,450}_{0.5,1,2} |
| Mg-2Gd-0.5Mn_extruded | 4 | 350_0.5, 450_{0.5,1,2} |
| Mg-2Gd-1Mn_extruded | 4 | 350_0.5, 450_{0.5,1,2} |
| Mg-5Gd_extruded | 8 | 350_{0.5,2}, 400_{0.5,1,2}, 450_{0.5,1,2} |
| Mg-5Gd-0.5Mn_extruded | 4 | 350_0.5, 450_{0.5,1,2} |
| Mg-5Gd-1Mn_extruded | 4 | 350_0.5, 450_{0.5,1,2} |
| Z1_extruded | 9 | {250,300,400}_{2,5,7.5} |
| ZNd10_extruded | 11 | {250,300,400}_{2,5,7.5} + 350_0.6, 350_2.4 |
| ZNd10_extruded_heattreated | 3 | EXCLUDED |
| ZX10_extruded | 11 | {250,300,400}_{2,5,7.5} + 300_0.6, 300_2.4 |
| ZX10_extruded_heattreated | 2 | EXCLUDED |

Total (actual database walk, verified 2026-07): 121 conditions; **108 as-extruded
(used)**; 13 heat-treated (excluded by default, `include_heat_treated: false`).
Ground truth per-alloy counts from the walk: AZ31 9, ME21 18, Mg-10Gd 9,
Mg-10Gd-0.5Mn 4, Mg-10Gd-1Mn 4, Mg-2Gd 9, Mg-2Gd-0.5Mn 4, Mg-2Gd-1Mn 4, Mg-5Gd 8,
Mg-5Gd-0.5Mn 4, Mg-5Gd-1Mn 4, Z1 9, ZNd10 11, ZX10 11 (14 base alloys).
Rationale for HT exclusion: with the HT flag dropped from the label
set, HT conditions collide with as-extruded twins on identical labels but different
microstructures.

## Composition labels (data_labels.xlsx, sheet1)

Rows: AZ31, ME21, Mg-2Gd, Mg-2Gd-0.5Mn, Mg-2Gd-1Mn, Mg-5Gd, Mg-5Gd-0.5Mn,
Mg-5Gd-1Mn, Mg-10Gd, Mg-10Gd-0.5Mn, "Mg-10Gd-1Mn)" (typo, normalize), Z1,
"ZNd10 = Mg-Zn-Nd" (normalize), "ZX10 = Mg-Zn-Ca" (normalize), Mg-10Gd-0.8Mn (no
conditions in database, dropped), Mg-4Gd-0.5Mn (empty row, dropped).

Element columns: Mn, Al, Zn, Cu, Ni, Si, Fe, Ce, Nd, Y, Pr, Gd, Ca.
Predicted set (8): Al, Zn, Mn, Ce, Gd, Ca, Nd, Y. Excluded as impurity-level
(< 0.012 wt% everywhere): Cu, Ni, Si, Fe, Pr.

Note: heat-treated conditions inherit the composition of their base alloy.

## training_data descriptor folders

| Folder | Descriptor | Scales | Notes |
|--------|------------|--------|-------|
| GSH/ | generalized spherical harmonics (from XRD ODF) | per condition | {alloy}_{T}_{v}_XRD.npy |
| ODF/ | discretized ODF | per condition | _odf.npy + _odf.txt |
| gram/ | VGG Gram matrices | 90/100/120/150 | |
| 2point_pymks/, 2point_mcrpy/ | 2-point statistics | 90/100/120/150 + binary/final | |
| 2point_mcrpy_reduced(_normalized)/ | pre-reduced 2-pt | " | LEAKAGE: fit on all data; refit per fold |
| aspect_ratio/, eq_diameter/ | grain histograms | per scale | 30 bins each |
| properties/ | stress-strain json | per condition | not used here (targets of Acta paper) |
| stats/, mask_stats/, binary/ | masks + grain stats | per scale | GNN experimental graphs source |

## Descriptor dataset conventions (what our builders emit, under data/)

- data/labels/labels.csv: condition_id, alloy, alloy_class_idx, Al..Y wt%, T_ext, v_ext
- data/splits/loco_fold{0..4}.json, loao_{alloy}.json: {train: [condition_id], val: [...], test: [...]}
- data/conventional/{fold}/X_{split}.npy + feature_names.json (per-fold refit reducers)
- data/genai/latents.npz: per-image 16x80 latents + condition_id map
- data/gnn/{synthetic,experimental}/graphs/*.pt + manifest.csv
