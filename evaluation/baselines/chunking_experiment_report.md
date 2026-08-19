# Chunking Experiment Matrix Report

## 1. Summary Comparison

| Configuration | Strategy | Chunks | Mean Len | Build (ms) | MRR | Hit Rate | All-Evidence Recall | Cit F1 | Ret Lat p50 (ms) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `recursive_bullet_v1_700_140` | recursive_bullet_v1 | 20 | 489.7 | 2526.9 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| `fixed_character_v1_500_100` | fixed_character_v1 | 24 | 489.7 | 294.3 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| `fixed_character_v1_700_140` | fixed_character_v1 | 17 | 687.8 | 2379.4 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| `fixed_character_v1_1000_200` | fixed_character_v1 | 12 | 971.1 | 901.9 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| `structure_aware_v1_700_140` | structure_aware_v1 | 26 | 362.8 | 237.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| `semantic_v1_035_200_900_1` | semantic_v1 | 28 | 552.9 | 306.3 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |

## 2. Chunk Distribution Analysis

| Configuration | Chunk Count | Min Chars | Max Chars | Mean Chars | p50 Chars | p90 Chars |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `recursive_bullet_v1_700_140` | 20 | 109 | 684 | 489.7 | 500.5 | 664.0 |
| `fixed_character_v1_500_100` | 24 | 253 | 500 | 489.7 | 500.0 | 500.0 |
| `fixed_character_v1_700_140` | 17 | 493 | 700 | 687.8 | 700.0 | 700.0 |
| `fixed_character_v1_1000_200` | 12 | 653 | 1000 | 971.1 | 1000.0 | 1000.0 |
| `structure_aware_v1_700_140` | 26 | 12 | 659 | 362.8 | 387.0 | 640.0 |
| `semantic_v1_035_200_900_1` | 28 | 190 | 887 | 552.9 | 539.0 | 842.7 |

## 3. Strategy Selection Decision & Recommendation

- **Selected Default Strategy:** `recursive_bullet_v1` (`chunk_size=700`, `chunk_overlap=140`)
- **Decision Rationale:**
  1. `recursive_bullet_v1` provides guaranteed backward compatibility and deterministic chunk boundaries matching established golden anchors.
  2. `structure_aware_v1` produces granular section-bounded chunks and serves as an effective candidate for hierarchical/structured markdown retrieval.
  3. `semantic_v1` provides adaptive embedding-distance boundary segmentation while isolating non-deterministic variations.
  4. Under the Spec 04 promotion gate criteria, `recursive_bullet_v1` remains the default baseline until live hybrid+reranker cross-validation is completed.
