# Fair Early Warning in Online Education with Dynamic Bayesian Networks

**Course:** ICT 5143: Algorithms Fairness and Graphical Models Laboratory (AFGML Capstone)
**Problem:** 3
**Team:** 3
**Submission date:** 7 November 2026
**Repository:** https://example.org/replace-with-repository
**Submission commit:** REPLACE

## Team

| Name | Roll number | Contribution |
|---|---|---|
| Anisha Chandra | 261090830001 | Understanding and code implementation |
| Dhanya Padasalgi | 261090830015 | Research and documentation |

## Question

Can a temporally regularized alert rule reduce disparities in early support while keeping predictive detection and a fixed support capacity?

**Decision context.** Stakeholders are students, tutors, disability-support teams, academic advisers and university learning-analytics teams. The outcome is Y = 1 for a final result of Fail or Withdrawn, and Y = 0 for Pass or Distinction. The decision is whether to offer voluntary support at checkpoint t ∈ {2, 4, 6, 8} weeks, with D_t = 1 meaning support was offered.

**Hypothesis.** At 20% capacity, the temporal rule has a lower cumulative coverage gap than independent per-checkpoint equalisation, and detection falls by at most 0.02. The hypothesis is refuted if the gap is not lower, or if detection falls by more than 0.02.

## Bounded contribution

The project proposes a threshold-offset alert rule with three parts:

- Group offsets (gender × disability) tuned jointly across the four checkpoints, rather than separately at each checkpoint.
- A re-alert discount that lowers priority for students already alerted.
- Smoothness penalties on changes in offsets across checkpoints.

The rule runs under a fixed capacity and is compared against the per-checkpoint equal-opportunity baseline (Hardt, Price and Srebro, 2016), which is reproduced as `indep_eopp`. The scope is one OULAD module (BBB) and one Kaggle copy of the dataset. The contribution is an evaluated method with ablations and robustness checks, not a causal claim or a deployable system.

## Headline findings

Results are for the test set at 20% capacity, mean over five student-level splits.

- **Predictive utility:** The three-state dynamic Bayesian network is within about 0.03 average precision of logistic regression at week 8 (0.780 vs 0.794). Its calibrated probabilities are shared by all policies.
- **Detection:** The temporal rule raises cumulative detection over the baseline in every split (0.683 vs 0.567 on test; +0.116 mean difference, non-inferior in 5 of 5 splits).
- **Coverage gap:** The mean gap is lower (0.095 vs 0.114 on test; 0.138 vs 0.253 on transfer). However, the test difference is significant in none of the five splits, so the reduction is not reliable on test.
- **False-alert exposure:** Exposure nearly doubles (0.300 vs 0.166 on test). The 0.05 non-inferiority margin holds in only 1 of 5 splits.
- **Against top-k:** The temporal rule does not reduce the gap relative to top-k, which ignores groups (0.095 vs 0.088 on test).
- **Mechanism:** Most of the detection gain comes from the re-alert discount. Removing it lowers detection to 0.570 and exposure to 0.170.

**Conclusion:** The fairness hypothesis is not supported. The temporal rule improves detection and lowers the mean gap relative to per-checkpoint equalisation, but the gap reduction is not reliable and the rule does not beat top-k on fairness.

## Data

- **Source:** Open University Learning Analytics Dataset (OULAD), module BBB (Kuzilek, Hlosta and Zdrahal, 2017), loaded from a Kaggle copy (`anlgrbz/student-demographics-online-education-dataset`).
- **Pool:** presentations 2013B, 2013J and 2014B, split 60/20/20 by student into train, validation and test, with five split seeds (2024, 1, 2, 3, 4).
- **Transfer:** presentation 2014J, with 81 students already seen in the pool removed.
- **Eligible students at week 2:** 947 (validation), 945 (test), 1,723 (transfer).
- **Test subgroup cells (gender × disability):** F|N 756; F|Y 76; M|N 99; M|Y 14. Cells with fewer than 30 positives or negatives are exploratory.

## Setup

- Google Colab is the reference environment.
- The notebook downloads the dataset with `kagglehub`. To use local CSV files instead, set the `OULAD_DIR` variable to the directory containing them.
- Configuration is in the `cfg` dictionary in Section 1 of the notebook.

## Reproduction

1. Open the notebook (`final_2_.ipynb`, referred to in the report as `final(2).ipynb`) in Google Colab.
2. Select **Runtime > Run all**, then run cells A–I.
3. Outputs are written to `results/`.

**Fallback:** If the run cannot be completed, a saved trace of the notebook is available at `results/finalnb.pdf`.

**Runtime and hardware:** About 3 minutes on Google Colab. The hardware configuration is not recorded in the project materials.

## Evidence

All outputs are in `results/`:

| File | Contents |
|---|---|
| `paired_bootstrap_all_splits.csv` | Paired student-level bootstrap results for H1–H4 across all five splits |
| `model_seeds.csv` | Results for the five DBN initialisation seeds on split 0 |
| `logloss_brier.csv` | Log loss and Brier scores for calibrated probabilities |
| `group_input_alert_changes.csv` | Alerts changed when group membership is given to the DBN |
| `alert_policy_simulation_test.csv` | Alert-policy simulator output on the test set, split 0 |
| `temporal_fairness.pdf` | Figures for the temporal fairness analysis |
| `finalnb.pdf` | Saved trace of the notebook (fallback) |

## Limitations

- One module and one Kaggle copy. The copy's version is not yet recorded.
- Several gender × disability cells are small, and the 55-and-over age band has no test students.
- Recorded clicks may reflect offline study, assistive technology, work commitments or inaccessible materials. Missing assessments may reflect circumstances rather than disengagement.
- Tuning budgets were not matched (200 evaluations for `indep_eopp`, 315 for `temporal`). This is a disclosed deviation.
- Five model seeds were run on split 0 only. The test results for the temporal rule are identical across seeds, but seed stability is not fully verified for all policies.
- The latent engagement state is a statistical summary, not ability or motivation. No causal claim is made.
- Repeated alerts may stigmatise students. Support must remain voluntary, with an opt-out and a route to contest an alert.

## References

- Hardt, M., Price, E., & Srebro, N. (2016). Equality of Opportunity in Supervised Learning. NeurIPS.
- Kuzilek, J., Hlosta, M., & Zdrahal, Z. (2017). Open University Learning Analytics Dataset. *Scientific Data*, 4, 170171.
- Koller, D., & Friedman, N. (2009). *Probabilistic Graphical Models*. MIT Press.
- Murphy, K. P. (2002). *Dynamic Bayesian Networks: Representation, Inference and Learning*. PhD thesis, UC Berkeley.
