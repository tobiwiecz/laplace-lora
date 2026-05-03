# Dataset Split Sizes

| Dataset | Config | Train | Validation | Test | Test Labels |
|---|---|---|---|---|---|
| winogrande | winogrande_s | 640 | 1,267 | 1,767 | **Withheld** |
| winogrande | winogrande_m | 2,558 | 1,267 | 1,767 | **Withheld** |
| ai2_arc | ARC-Challenge | 1,119 | 299 | 1,172 | Available |
| ai2_arc | ARC-Easy | 2,251 | 570 | 2,376 | Available |
| openbookqa | main | 4,957 | 500 | 500 | Available |
| super_glue | boolq | 9,427 | 3,270 | 3,245 | **Withheld** |

## Notes

- **Winogrande**: All training-size configs (xs/s/m/l/xl) share the same validation (1,267) and test (1,767) sets. The test `answer` field is empty; evaluation requires a submission to the AllenAI leaderboard. In practice, the validation set is used for reporting.

- **ARC-Challenge / ARC-Easy**: The HuggingFace release includes `answerKey` in the test split — labels are publicly available.

- **OpenBookQA**: Same as ARC — `answerKey` is present in all splits including test.

- **BoolQ**: Part of SuperGLUE. Test labels are withheld (placeholder value -1). Most papers report results on the **validation set** (3,270 examples).
