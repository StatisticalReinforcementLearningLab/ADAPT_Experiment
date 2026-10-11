# Prior Mean and Variance Summary

`prior_variance` is the diagonal entry of the prior covariance. For `q_td_modify_joint`, the full covariance remains in `rl_priors.json`; this table reports marginal variances.

## PF Priors

| prior_family | model | block | index | feature | prior_mean | prior_variance |
| --- | --- | --- | --- | --- | --- | --- |
| PF | fourSC | theta | 0 | intercept | -0.047490 | 0.043173 |
| PF | fourSC | theta | 1 | stepCountNext4HourLag1 | 0.047787 | 0.061929 |
| PF | fourSC | theta | 2 | yesterdayStepCount | 0.638668 | 0.061199 |
| PF | fourSC | theta | 3 | prior2HourStepCount | 0.148178 | 0.043106 |
| PF | fourSC | theta | 4 | activitySuggestionsSentLast7Days | -0.001962 | 0.003658 |
| PF | fourSC | theta | 5 | activitySuggestionInteractLast7Days | 0.047875 | 0.048633 |
| PF | fourSC | theta | 6 | activeDaysLast7Days | 0.069972 | 0.063672 |
| PF | fourSC | theta | 7 | isWeekend | -0.001310 | 0.045217 |
| PF | fourSC | theta | 8 | decisionTimeSlot | -0.107634 | 0.135899 |
| PF | fourSC | theta | 9 | perceivedUtilityLastWeek | -0.001347 | 0.059779 |
| PF | fourSC | theta | 10 | caeAverageLastWeek | -0.022758 | 0.029210 |
| PF | fourSC | theta | 11 | Ah | 0.015409 | 0.041792 |
| PF | fourSC | theta | 12 | Ah*yesterdayStepCount | 0.152430 | 0.077767 |
| PF | fourSC | theta | 13 | Ah*prior2HourStepCount | -0.034753 | 0.025917 |
| PF | fourSC | theta | 14 | Ah*perceivedUtilityLastWeek | -0.063912 | 0.078488 |
| PF | fourSC | theta | 15 | Ah*caeAverageLastWeek | 0.015111 | 0.051599 |
| PF | fourSC | theta | 16 | Ah*decisionTimeSlot | 0.003800 | 0.024060 |
| PF | antic | theta | 0 | intercept | 0.287541 | 0.009923 |
| PF | antic | theta | 1 | anticipated_affect_yesterday | 0.211278 | 0.006582 |
| PF | antic | theta | 2 | active_status_fraction_7days | 0.113264 | 0.007070 |
| PF | antic | theta | 3 | is_weekend | -0.030918 | 0.005236 |
| PF | antic | theta | 4 | perceived_utility_lastweek | 0.048915 | 0.006660 |
| PF | antic | theta | 5 | CAE_avg_lastweek | 0.202025 | 0.007429 |
| PF | antic | theta | 6 | recent_burden | 0.009179 | 0.003478 |
| PF | antic | theta | 7 | A0_morning | -0.008123 | 0.001519 |
| PF | antic | theta | 8 | A1_afternoon | 0.047738 | 0.001041 |
| PF | antic | theta | 9 | A0_morning_by_perceived_utility_lastweek | 0.010791 | 0.003078 |
| PF | antic | theta | 10 | A1_afternoon_by_perceived_utility_lastweek | -0.025311 | 0.001856 |
| PF | antic | theta | 11 | A0_morning_by_CAE_avg_lastweek | -0.009347 | 0.006492 |
| PF | antic | theta | 12 | A1_afternoon_by_CAE_avg_lastweek | -0.018554 | 0.003637 |
| PF | CAE | theta | 0 | intercept | -0.094983 | 0.108848 |
| PF | CAE | theta | 1 | CAE_avg_lastweek | 0.884206 | 0.058809 |
| PF | CAE | theta | 2 | fourSC_ewma | -0.039165 | 0.020283 |
| PF | CAE | theta | 3 | anticipated_affect_ewma | 0.188655 | 0.024217 |
| PF | CAE_short | theta | 0 | intercept | 0.000498 | 0.067981 |
| PF | CAE_short | theta | 1 | caeAverage | 0.968416 | 0.060727 |

## RL Priors

| prior_family | model | block | index | feature | prior_mean | prior_variance |
| --- | --- | --- | --- | --- | --- | --- |
| RL | reward_shaping | eta | 0 | intercept | 0.013619 | 0.005287 |
| RL | reward_shaping | eta | 1 | weekday_vs_weekend | 0.002270 | 0.000147 |
| RL | reward_shaping | eta | 2 | slot_pm | 0.006809 | 0.001322 |
| RL | reward_shaping | eta | 3 | E_w | 0.046714 | 0.001441 |
| RL | reward_shaping | eta | 4 | b_hat | 0.058723 | 0.005192 |
| RL | reward_shaping | eta | 5 | b_tilde | 0.000000 | 1.000000 |
| RL | reward_shaping | eta | 6 | M_Y_anticipated_affect_ewma | 0.117814 | 0.005913 |
| RL | reward_shaping | eta | 7 | M_Y_fourSC_ewma | 0.010958 | 0.004984 |
| RL | reward_shaping | eta | 8 | M_E_pageview_ewma | 0.002294 | 0.000916 |
| RL | reward_shaping | eta | 9 | M_E_fitbit_wear_ewma | -0.031354 | 0.006469 |
| RL | reward_shaping | eta | 10 | M_E_survey_complete_ewma | 0.017111 | 0.005117 |
| RL | reward_shaping | eta | 11 | yesterday_step_count | -0.008537 | 0.009119 |
| RL | reward_shaping | eta | 12 | prior2hour_step_count | -0.021777 | 0.005765 |
| RL | reward_shaping | eta | 13 | active_status_fraction_7days | -0.022166 | 0.005038 |
| RL | reward_shaping | eta | 14 | recent_burden | -0.000878 | 0.000377 |
| RL | reward_shaping | eta | 15 | walk_interaction_7d | -0.018200 | 0.006499 |
| RL | redistribution_stage1_AA | eta | 0 | intercept | 0.169885 | 0.006115 |
| RL | redistribution_stage1_AA | eta | 1 | E_w | 0.012625 | 0.002205 |
| RL | redistribution_stage1_AA | eta | 2 | b_hat | 0.078694 | 0.002940 |
| RL | redistribution_stage1_AA | eta | 3 | b_tilde | 0.000000 | 1.000000 |
| RL | redistribution_stage1_AA | eta | 4 | M_Y_anticipated_affect_ewma | 0.168726 | 0.001832 |
| RL | redistribution_stage1_AA | eta | 5 | M_Y_fourSC_ewma | 0.002520 | 0.002625 |
| RL | redistribution_stage1_AA | eta | 6 | M_E_pageview_ewma | 0.013147 | 0.000415 |
| RL | redistribution_stage1_AA | eta | 7 | M_E_fitbit_wear_ewma | -0.030285 | 0.001411 |
| RL | redistribution_stage1_AA | eta | 8 | M_E_survey_complete_ewma | -0.023902 | 0.001534 |
| RL | redistribution_stage1_AA | eta | 9 | yesterday_step_count | -0.007861 | 0.000917 |
| RL | redistribution_stage1_AA | eta | 10 | prior2hour_step_count | 0.008109 | 0.000650 |
| RL | redistribution_stage1_AA | eta | 11 | active_status_fraction_7days | 0.080586 | 0.002559 |
| RL | redistribution_stage1_AA | eta | 12 | recent_burden | 0.001304 | 8.515867e-05 |
| RL | redistribution_stage1_AA | eta | 13 | walk_interaction_7d | 0.004430 | 0.002290 |
| RL | redistribution_stage1_AA | eta | 14 | A | 0.017973 | 0.000980 |
| RL | redistribution_stage1_AA | eta | 15 | A*E_w | -0.000525 | 0.001625 |
| RL | redistribution_stage1_AA | eta | 16 | A*b_hat | -0.007112 | 0.001748 |
| RL | redistribution_stage1_AA | eta | 17 | A*b_tilde | 0.000000 | 1.000000 |
| RL | redistribution_stage1_AA | eta | 18 | A*yesterday_step_count | -0.004086 | 0.002686 |
| RL | redistribution_stage1_AA | eta | 19 | A*prior2hour_step_count | 0.011693 | 0.000415 |
| RL | redistribution_stage1_AA | eta | 20 | A*active_status_fraction_7days | -0.022810 | 0.000886 |
| RL | redistribution_stage1_AA | eta | 21 | A*recent_burden | 0.001400 | 0.000323 |
| RL | redistribution_stage1_AA | eta | 22 | A*walk_interaction_7d | -0.004653 | 0.000840 |
| RL | redistribution_stage1_AA | eta | 23 | A*slot_pm | 0.012055 | 0.001704 |
| RL | redistribution_stage1_FW | eta | 0 | intercept | 0.358031 | 0.024058 |
| RL | redistribution_stage1_FW | eta | 1 | E_w | -0.002353 | 0.014559 |
| RL | redistribution_stage1_FW | eta | 2 | b_hat | 0.005472 | 0.012830 |
| RL | redistribution_stage1_FW | eta | 3 | b_tilde | 0.000000 | 1.000000 |
| RL | redistribution_stage1_FW | eta | 4 | M_Y_anticipated_affect_ewma | -0.128139 | 0.019858 |
| RL | redistribution_stage1_FW | eta | 5 | M_Y_fourSC_ewma | -0.000876 | 0.016672 |
| RL | redistribution_stage1_FW | eta | 6 | M_E_pageview_ewma | 0.029240 | 0.001384 |
| RL | redistribution_stage1_FW | eta | 7 | M_E_fitbit_wear_ewma | 0.165508 | 0.014643 |
| RL | redistribution_stage1_FW | eta | 8 | M_E_survey_complete_ewma | -0.006093 | 0.012851 |
| RL | redistribution_stage1_FW | eta | 9 | yesterday_step_count | -0.024042 | 0.006843 |
| RL | redistribution_stage1_FW | eta | 10 | prior2hour_step_count | 0.004501 | 0.002865 |
| RL | redistribution_stage1_FW | eta | 11 | active_status_fraction_7days | 0.067427 | 0.024304 |
| RL | redistribution_stage1_FW | eta | 12 | recent_burden | -0.011319 | 0.002882 |
| RL | redistribution_stage1_FW | eta | 13 | walk_interaction_7d | 0.032398 | 0.029074 |
| RL | redistribution_stage1_FW | eta | 14 | A | 0.006917 | 0.007730 |
| RL | redistribution_stage1_FW | eta | 15 | A*E_w | 0.012711 | 0.009281 |
| RL | redistribution_stage1_FW | eta | 16 | A*b_hat | -0.012993 | 0.011922 |
| RL | redistribution_stage1_FW | eta | 17 | A*b_tilde | 0.000000 | 1.000000 |
| RL | redistribution_stage1_FW | eta | 18 | A*yesterday_step_count | 0.005173 | 0.017876 |
| RL | redistribution_stage1_FW | eta | 19 | A*prior2hour_step_count | -0.006446 | 0.005030 |
| RL | redistribution_stage1_FW | eta | 20 | A*active_status_fraction_7days | -0.006417 | 0.013291 |
| RL | redistribution_stage1_FW | eta | 21 | A*recent_burden | 0.005882 | 0.004024 |
| RL | redistribution_stage1_FW | eta | 22 | A*walk_interaction_7d | -0.012983 | 0.024813 |
| RL | redistribution_stage1_FW | eta | 23 | A*slot_pm | -0.032067 | 0.011854 |
| RL | redistribution_stage1_PJ | eta | 0 | intercept | 0.145567 | 0.030327 |
| RL | redistribution_stage1_PJ | eta | 1 | E_w | -0.023189 | 0.010365 |
| RL | redistribution_stage1_PJ | eta | 2 | b_hat | 0.032344 | 0.018138 |
| RL | redistribution_stage1_PJ | eta | 3 | b_tilde | 0.000000 | 1.000000 |
| RL | redistribution_stage1_PJ | eta | 4 | M_Y_anticipated_affect_ewma | -0.145974 | 0.032670 |
| RL | redistribution_stage1_PJ | eta | 5 | M_Y_fourSC_ewma | -0.018960 | 0.027070 |
| RL | redistribution_stage1_PJ | eta | 6 | M_E_pageview_ewma | 0.013185 | 0.001979 |
| RL | redistribution_stage1_PJ | eta | 7 | M_E_fitbit_wear_ewma | 0.018803 | 0.008702 |
| RL | redistribution_stage1_PJ | eta | 8 | M_E_survey_complete_ewma | 0.164595 | 0.013127 |
| RL | redistribution_stage1_PJ | eta | 9 | yesterday_step_count | 0.009331 | 0.023374 |
| RL | redistribution_stage1_PJ | eta | 10 | prior2hour_step_count | 0.024827 | 0.003371 |
| RL | redistribution_stage1_PJ | eta | 11 | active_status_fraction_7days | 0.060458 | 0.023738 |
| RL | redistribution_stage1_PJ | eta | 12 | recent_burden | 0.012763 | 0.003885 |
| RL | redistribution_stage1_PJ | eta | 13 | walk_interaction_7d | 0.188715 | 0.031283 |
| RL | redistribution_stage1_PJ | eta | 14 | A | 0.009335 | 0.018664 |
| RL | redistribution_stage1_PJ | eta | 15 | A*E_w | 0.016780 | 0.012848 |
| RL | redistribution_stage1_PJ | eta | 16 | A*b_hat | 0.001017 | 0.024649 |
| RL | redistribution_stage1_PJ | eta | 17 | A*b_tilde | 0.000000 | 1.000000 |
| RL | redistribution_stage1_PJ | eta | 18 | A*yesterday_step_count | 0.004537 | 0.031573 |
| RL | redistribution_stage1_PJ | eta | 19 | A*prior2hour_step_count | -0.002982 | 0.011559 |
| RL | redistribution_stage1_PJ | eta | 20 | A*active_status_fraction_7days | -0.024859 | 0.017225 |
| RL | redistribution_stage1_PJ | eta | 21 | A*recent_burden | -0.007444 | 0.009943 |
| RL | redistribution_stage1_PJ | eta | 22 | A*walk_interaction_7d | -0.036459 | 0.027972 |
| RL | redistribution_stage1_PJ | eta | 23 | A*slot_pm | -0.022952 | 0.019138 |
| RL | redistribution_stage1_SC | eta | 0 | intercept | 0.006338 | 0.182818 |
| RL | redistribution_stage1_SC | eta | 1 | weekday_vs_weekend | -0.014838 | 0.026119 |
| RL | redistribution_stage1_SC | eta | 2 | slot_pm | -0.119766 | 0.094844 |
| RL | redistribution_stage1_SC | eta | 3 | E_w | -0.040761 | 0.082417 |
| RL | redistribution_stage1_SC | eta | 4 | b_hat | -0.011649 | 0.031309 |
| RL | redistribution_stage1_SC | eta | 5 | b_tilde | 0.000000 | 1.000000 |
| RL | redistribution_stage1_SC | eta | 6 | yesterday_step_count | 0.827761 | 0.023758 |
| RL | redistribution_stage1_SC | eta | 7 | prior2hour_step_count | 0.123453 | 0.023002 |
| RL | redistribution_stage1_SC | eta | 8 | active_status_fraction_7days | 0.014706 | 0.073790 |
| RL | redistribution_stage1_SC | eta | 9 | recent_burden | -0.008060 | 0.001935 |
| RL | redistribution_stage1_SC | eta | 10 | walk_interaction_7d | 0.060700 | 0.028629 |
| RL | redistribution_stage1_SC | eta | 11 | A | -0.051890 | 0.017448 |
| RL | redistribution_stage1_SC | eta | 12 | A*slot_pm | 0.013683 | 0.024041 |
| RL | redistribution_stage1_PV | eta | 0 | intercept | -2.645482 | 0.461050 |
| RL | redistribution_stage1_PV | eta | 1 | weekday_vs_weekend | -0.171159 | 0.177857 |
| RL | redistribution_stage1_PV | eta | 2 | slot_pm | -0.165188 | 0.166583 |
| RL | redistribution_stage1_PV | eta | 3 | E_w | -0.007368 | 0.276919 |
| RL | redistribution_stage1_PV | eta | 4 | b_hat | 0.087344 | 0.307636 |
| RL | redistribution_stage1_PV | eta | 5 | b_tilde | 0.000000 | 1.000000 |
| RL | redistribution_stage1_PV | eta | 6 | yesterday_step_count | 0.115700 | 0.084981 |
| RL | redistribution_stage1_PV | eta | 7 | prior2hour_step_count | 0.072069 | 0.032670 |
| RL | redistribution_stage1_PV | eta | 8 | active_status_fraction_7days | -0.293292 | 0.458600 |
| RL | redistribution_stage1_PV | eta | 9 | recent_burden | -0.041914 | 0.032874 |
| RL | redistribution_stage1_PV | eta | 10 | walk_interaction_7d | 1.039181 | 0.416168 |
| RL | redistribution_stage1_PV | eta | 11 | A | 0.562862 | 0.563983 |
| RL | redistribution_stage1_PV | eta | 12 | A*slot_pm | 0.111326 | 0.139227 |
| RL | redistribution_stage2_v4 | eta | 0 | redistribution_stage2_v4_coef_0 | 0.051277 | 0.001984 |
| RL | redistribution_stage2_v4 | eta | 1 | redistribution_stage2_v4_coef_1 | 0.008546 | 5.510984e-05 |
| RL | redistribution_stage2_v4 | eta | 2 | redistribution_stage2_v4_coef_2 | -0.034917 | 0.002079 |
| RL | redistribution_stage2_v4 | eta | 3 | redistribution_stage2_v4_coef_3 | 0.061750 | 0.003049 |
| RL | redistribution_stage2_v4 | eta | 4 | redistribution_stage2_v4_coef_4 | 0.000000 | 1.000000 |
| RL | redistribution_stage2_v4 | eta | 5 | redistribution_stage2_v4_coef_5 | 0.130071 | 0.003361 |
| RL | redistribution_stage2_v4 | eta | 6 | redistribution_stage2_v4_coef_6 | 0.013180 | 0.004908 |
| RL | redistribution_stage2_v4 | eta | 7 | redistribution_stage2_v4_coef_7 | 0.014456 | 0.003609 |
| RL | redistribution_stage2_v4 | eta | 8 | redistribution_stage2_v4_coef_8 | -0.019206 | 0.003630 |
| RL | redistribution_stage2_v4 | eta | 9 | redistribution_stage2_v4_coef_9 | 0.011288 | 0.004388 |
| RL | redistribution_stage2_v4 | eta | 10 | redistribution_stage2_v4_coef_10 | -0.019931 | 0.005001 |
| RL | redistribution_stage2_v4 | eta | 11 | redistribution_stage2_v4_coef_11 | -0.023091 | 0.003270 |
| RL | redistribution_stage2_v4 | eta | 12 | redistribution_stage2_v4_coef_12 | -0.013084 | 0.002347 |
| RL | redistribution_stage2_v4 | eta | 13 | redistribution_stage2_v4_coef_13 | -0.001966 | 0.000319 |
| RL | redistribution_stage2_v4 | eta | 14 | redistribution_stage2_v4_coef_14 | -0.018845 | 0.004260 |
| RL | redistribution_stage2_v4 | eta | 15 | redistribution_stage2_v4_coef_15 | -0.010534 | 0.004112 |
| RL | redistribution_stage2_v4 | eta | 16 | redistribution_stage2_v4_coef_16 | -0.050886 | 0.000295 |
| RL | redistribution_stage2_v4 | eta | 17 | redistribution_stage2_v4_coef_17 | -0.106553 | 0.002806 |
| RL | redistribution_stage2_v4 | eta | 18 | redistribution_stage2_v4_coef_18 | 0.035300 | 0.002037 |
| RL | redistribution_stage2_v4 | eta | 19 | redistribution_stage2_v4_coef_19 | 0.009558 | 0.001120 |
| RL | q_no_td_modify | beta | 0 | intercept | 0.036085 | 0.093236 |
| RL | q_no_td_modify | beta | 1 | weekday_vs_weekend | -0.069053 | 0.095240 |
| RL | q_no_td_modify | beta | 2 | slot_pm | -0.007348 | 0.002961 |
| RL | q_no_td_modify | beta | 3 | E_w | 0.004395 | 0.043851 |
| RL | q_no_td_modify | beta | 4 | b_hat | 1.143277 | 0.055746 |
| RL | q_no_td_modify | beta | 5 | b_tilde | 0.000000 | 1.000000 |
| RL | q_no_td_modify | beta | 6 | M_Y_anticipated_affect_ewma | 0.286361 | 0.006822 |
| RL | q_no_td_modify | beta | 7 | M_Y_fourSC_ewma | -0.010173 | 0.009743 |
| RL | q_no_td_modify | beta | 8 | M_E_pageview_ewma | 0.001547 | 0.000233 |
| RL | q_no_td_modify | beta | 9 | M_E_fitbit_wear_ewma | -0.130661 | 0.004281 |
| RL | q_no_td_modify | beta | 10 | M_E_survey_complete_ewma | -0.034674 | 0.002796 |
| RL | q_no_td_modify | beta | 11 | yesterday_step_count | -0.019786 | 0.002634 |
| RL | q_no_td_modify | beta | 12 | prior2hour_step_count | -0.008717 | 0.000458 |
| RL | q_no_td_modify | beta | 13 | active_status_fraction_7days | 0.116305 | 0.031147 |
| RL | q_no_td_modify | beta | 14 | recent_burden | -0.012247 | 0.003094 |
| RL | q_no_td_modify | beta | 15 | walk_interaction_7d | -0.020215 | 0.025460 |
| RL | q_no_td_modify | beta | 16 | A | 0.010500 | 0.001662 |
| RL | q_no_td_modify | beta | 17 | A*E_w | -0.010543 | 0.003207 |
| RL | q_no_td_modify | beta | 18 | A*b_hat | 0.003598 | 0.001316 |
| RL | q_no_td_modify | beta | 19 | A*b_tilde | 0.000000 | 1.000000 |
| RL | q_no_td_modify | beta | 20 | A*yesterday_step_count | -0.000145 | 0.003294 |
| RL | q_no_td_modify | beta | 21 | A*prior2hour_step_count | 0.004467 | 0.000824 |
| RL | q_no_td_modify | beta | 22 | A*active_status_fraction_7days | -0.006360 | 0.001330 |
| RL | q_no_td_modify | beta | 23 | A*recent_burden | -0.004188 | 0.001129 |
| RL | q_no_td_modify | beta | 24 | A*walk_interaction_7d | -0.001039 | 0.001174 |
| RL | q_no_td_modify | beta | 25 | A*weekday_vs_weekend | 0.042453 | 0.004602 |
| RL | q_no_td_modify | beta | 26 | A*slot_pm | 0.008520 | 0.002164 |
| RL | q_no_td_modify_g09 | beta | 0 | intercept | 0.029317 | 0.160400 |
| RL | q_no_td_modify_g09 | beta | 1 | weekday_vs_weekend | -0.063153 | 0.244419 |
| RL | q_no_td_modify_g09 | beta | 2 | slot_pm | -0.005814 | 0.008811 |
| RL | q_no_td_modify_g09 | beta | 3 | E_w | 0.003146 | 0.087636 |
| RL | q_no_td_modify_g09 | beta | 4 | b_hat | 1.544966 | 0.093949 |
| RL | q_no_td_modify_g09 | beta | 5 | b_tilde | 0.000000 | 1.000000 |
| RL | q_no_td_modify_g09 | beta | 6 | M_Y_anticipated_affect_ewma | 0.471885 | 0.016809 |
| RL | q_no_td_modify_g09 | beta | 7 | M_Y_fourSC_ewma | -0.016245 | 0.017682 |
| RL | q_no_td_modify_g09 | beta | 8 | M_E_pageview_ewma | 0.004341 | 0.000452 |
| RL | q_no_td_modify_g09 | beta | 9 | M_E_fitbit_wear_ewma | -0.205951 | 0.009013 |
| RL | q_no_td_modify_g09 | beta | 10 | M_E_survey_complete_ewma | -0.048950 | 0.004850 |
| RL | q_no_td_modify_g09 | beta | 11 | yesterday_step_count | -0.026052 | 0.005068 |
| RL | q_no_td_modify_g09 | beta | 12 | prior2hour_step_count | -0.015272 | 0.000731 |
| RL | q_no_td_modify_g09 | beta | 13 | active_status_fraction_7days | 0.209767 | 0.052913 |
| RL | q_no_td_modify_g09 | beta | 14 | recent_burden | -0.017846 | 0.007730 |
| RL | q_no_td_modify_g09 | beta | 15 | walk_interaction_7d | -0.012273 | 0.048301 |
| RL | q_no_td_modify_g09 | beta | 16 | A | 0.022164 | 0.002792 |
| RL | q_no_td_modify_g09 | beta | 17 | A*E_w | -0.016764 | 0.005835 |
| RL | q_no_td_modify_g09 | beta | 18 | A*b_hat | 0.005941 | 0.002894 |
| RL | q_no_td_modify_g09 | beta | 19 | A*b_tilde | 0.000000 | 1.000000 |
| RL | q_no_td_modify_g09 | beta | 20 | A*yesterday_step_count | -0.001576 | 0.007165 |
| RL | q_no_td_modify_g09 | beta | 21 | A*prior2hour_step_count | 0.007476 | 0.001321 |
| RL | q_no_td_modify_g09 | beta | 22 | A*active_status_fraction_7days | -0.019676 | 0.003004 |
| RL | q_no_td_modify_g09 | beta | 23 | A*recent_burden | -0.006546 | 0.002275 |
| RL | q_no_td_modify_g09 | beta | 24 | A*walk_interaction_7d | 0.000902 | 0.002290 |
| RL | q_no_td_modify_g09 | beta | 25 | A*weekday_vs_weekend | 0.058788 | 0.006079 |
| RL | q_no_td_modify_g09 | beta | 26 | A*slot_pm | 0.014459 | 0.005477 |
| RL | q_no_td_modify_g097 | beta | 0 | intercept | 0.026409 | 0.176966 |
| RL | q_no_td_modify_g097 | beta | 1 | weekday_vs_weekend | -0.060182 | 0.289471 |
| RL | q_no_td_modify_g097 | beta | 2 | slot_pm | -0.005226 | 0.010748 |
| RL | q_no_td_modify_g097 | beta | 3 | E_w | 0.002592 | 0.100226 |
| RL | q_no_td_modify_g097 | beta | 4 | b_hat | 1.632866 | 0.104001 |
| RL | q_no_td_modify_g097 | beta | 5 | b_tilde | 0.000000 | 1.000000 |
| RL | q_no_td_modify_g097 | beta | 6 | M_Y_anticipated_affect_ewma | 0.515668 | 0.020155 |
| RL | q_no_td_modify_g097 | beta | 7 | M_Y_fourSC_ewma | -0.017655 | 0.019526 |
| RL | q_no_td_modify_g097 | beta | 8 | M_E_pageview_ewma | 0.005098 | 0.000516 |
| RL | q_no_td_modify_g097 | beta | 9 | M_E_fitbit_wear_ewma | -0.223370 | 0.010942 |
| RL | q_no_td_modify_g097 | beta | 10 | M_E_survey_complete_ewma | -0.052192 | 0.005650 |
| RL | q_no_td_modify_g097 | beta | 11 | yesterday_step_count | -0.027342 | 0.005754 |
| RL | q_no_td_modify_g097 | beta | 12 | prior2hour_step_count | -0.016888 | 0.000811 |
| RL | q_no_td_modify_g097 | beta | 13 | active_status_fraction_7days | 0.231758 | 0.058484 |
| RL | q_no_td_modify_g097 | beta | 14 | recent_burden | -0.019142 | 0.009304 |
| RL | q_no_td_modify_g097 | beta | 15 | walk_interaction_7d | -0.009724 | 0.054429 |
| RL | q_no_td_modify_g097 | beta | 16 | A | 0.025240 | 0.003131 |
| RL | q_no_td_modify_g097 | beta | 17 | A*E_w | -0.018281 | 0.006593 |
| RL | q_no_td_modify_g097 | beta | 18 | A*b_hat | 0.006489 | 0.003414 |
| RL | q_no_td_modify_g097 | beta | 19 | A*b_tilde | 0.000000 | 1.000000 |
| RL | q_no_td_modify_g097 | beta | 20 | A*yesterday_step_count | -0.001921 | 0.008285 |
| RL | q_no_td_modify_g097 | beta | 21 | A*prior2hour_step_count | 0.008191 | 0.001469 |
| RL | q_no_td_modify_g097 | beta | 22 | A*active_status_fraction_7days | -0.023210 | 0.003514 |
| RL | q_no_td_modify_g097 | beta | 23 | A*recent_burden | -0.007138 | 0.002599 |
| RL | q_no_td_modify_g097 | beta | 24 | A*walk_interaction_7d | 0.001352 | 0.002688 |
| RL | q_no_td_modify_g097 | beta | 25 | A*weekday_vs_weekend | 0.062399 | 0.006515 |
| RL | q_no_td_modify_g097 | beta | 26 | A*slot_pm | 0.015893 | 0.006671 |
| RL | q_redistribution_v3 | beta | 0 | intercept | 0.580491 | 0.175965 |
| RL | q_redistribution_v3 | beta | 1 | weekday_vs_weekend | -0.163542 | 0.240515 |
| RL | q_redistribution_v3 | beta | 2 | slot_pm | -0.037872 | 0.009528 |
| RL | q_redistribution_v3 | beta | 3 | E_w | -0.552254 | 0.149532 |
| RL | q_redistribution_v3 | beta | 4 | b_hat | 1.640608 | 0.123015 |
| RL | q_redistribution_v3 | beta | 5 | b_tilde | 0.000000 | 1.000000 |
| RL | q_redistribution_v3 | beta | 6 | M_Y_anticipated_affect_ewma | 0.476822 | 0.017961 |
| RL | q_redistribution_v3 | beta | 7 | M_Y_fourSC_ewma | -0.020061 | 0.019013 |
| RL | q_redistribution_v3 | beta | 8 | M_E_pageview_ewma | 0.001486 | 0.000629 |
| RL | q_redistribution_v3 | beta | 9 | M_E_fitbit_wear_ewma | -0.244261 | 0.012896 |
| RL | q_redistribution_v3 | beta | 10 | M_E_survey_complete_ewma | -0.034539 | 0.006908 |
| RL | q_redistribution_v3 | beta | 11 | yesterday_step_count | -0.035393 | 0.007148 |
| RL | q_redistribution_v3 | beta | 12 | prior2hour_step_count | -0.016373 | 0.000898 |
| RL | q_redistribution_v3 | beta | 13 | active_status_fraction_7days | 0.093174 | 0.074636 |
| RL | q_redistribution_v3 | beta | 14 | recent_burden | -0.008216 | 0.014356 |
| RL | q_redistribution_v3 | beta | 15 | walk_interaction_7d | -0.062384 | 0.039370 |
| RL | q_redistribution_v3 | beta | 16 | A | 0.023560 | 0.003560 |
| RL | q_redistribution_v3 | beta | 17 | A*E_w | -0.018989 | 0.007692 |
| RL | q_redistribution_v3 | beta | 18 | A*b_hat | 0.006250 | 0.005660 |
| RL | q_redistribution_v3 | beta | 19 | A*b_tilde | 0.000000 | 1.000000 |
| RL | q_redistribution_v3 | beta | 20 | A*yesterday_step_count | -0.002091 | 0.007318 |
| RL | q_redistribution_v3 | beta | 21 | A*prior2hour_step_count | 0.009026 | 0.001649 |
| RL | q_redistribution_v3 | beta | 22 | A*active_status_fraction_7days | -0.026292 | 0.003606 |
| RL | q_redistribution_v3 | beta | 23 | A*recent_burden | -0.006698 | 0.002203 |
| RL | q_redistribution_v3 | beta | 24 | A*walk_interaction_7d | 0.008254 | 0.002868 |
| RL | q_redistribution_v3 | beta | 25 | A*weekday_vs_weekend | 0.071233 | 0.011675 |
| RL | q_redistribution_v3 | beta | 26 | A*slot_pm | 0.024161 | 0.006679 |
| RL | q_redistribution_v4 | beta | 0 | intercept | 0.459319 | 0.306327 |
| RL | q_redistribution_v4 | beta | 1 | weekday_vs_weekend | 0.005414 | 0.018439 |
| RL | q_redistribution_v4 | beta | 2 | slot_pm | 0.011373 | 0.001803 |
| RL | q_redistribution_v4 | beta | 3 | E_w | -0.555200 | 0.159613 |
| RL | q_redistribution_v4 | beta | 4 | b_hat | 1.667747 | 0.127861 |
| RL | q_redistribution_v4 | beta | 5 | b_tilde | 0.000000 | 1.000000 |
| RL | q_redistribution_v4 | beta | 6 | M_Y_anticipated_affect_ewma | 0.309215 | 0.022321 |
| RL | q_redistribution_v4 | beta | 7 | M_Y_fourSC_ewma | -0.021080 | 0.021908 |
| RL | q_redistribution_v4 | beta | 8 | M_E_pageview_ewma | 0.014990 | 0.006805 |
| RL | q_redistribution_v4 | beta | 9 | M_E_fitbit_wear_ewma | -0.157214 | 0.027519 |
| RL | q_redistribution_v4 | beta | 10 | M_E_survey_complete_ewma | 0.039060 | 0.029114 |
| RL | q_redistribution_v4 | beta | 11 | yesterday_step_count | -0.060919 | 0.031728 |
| RL | q_redistribution_v4 | beta | 12 | prior2hour_step_count | -0.011346 | 0.003202 |
| RL | q_redistribution_v4 | beta | 13 | active_status_fraction_7days | 0.220005 | 0.124181 |
| RL | q_redistribution_v4 | beta | 14 | recent_burden | -0.026243 | 0.015101 |
| RL | q_redistribution_v4 | beta | 15 | walk_interaction_7d | 0.026688 | 0.069136 |
| RL | q_redistribution_v4 | beta | 16 | A | 0.057131 | 0.004360 |
| RL | q_redistribution_v4 | beta | 17 | A*E_w | 0.009093 | 0.008005 |
| RL | q_redistribution_v4 | beta | 18 | A*b_hat | 0.002395 | 0.002408 |
| RL | q_redistribution_v4 | beta | 19 | A*b_tilde | 0.000000 | 1.000000 |
| RL | q_redistribution_v4 | beta | 20 | A*yesterday_step_count | 0.007631 | 0.001987 |
| RL | q_redistribution_v4 | beta | 21 | A*prior2hour_step_count | 0.019184 | 0.001217 |
| RL | q_redistribution_v4 | beta | 22 | A*active_status_fraction_7days | -0.079881 | 0.004447 |
| RL | q_redistribution_v4 | beta | 23 | A*recent_burden | 0.006594 | 0.002380 |
| RL | q_redistribution_v4 | beta | 24 | A*walk_interaction_7d | -0.070118 | 0.005233 |
| RL | q_redistribution_v4 | beta | 25 | A*weekday_vs_weekend | 0.023393 | 0.013194 |
| RL | q_redistribution_v4 | beta | 26 | A*slot_pm | 0.003816 | 0.006752 |
| RL | q_redistribution_v4_resid | beta | 0 | intercept | 0.460467 | 0.309715 |
| RL | q_redistribution_v4_resid | beta | 1 | weekday_vs_weekend | 0.004421 | 0.018659 |
| RL | q_redistribution_v4_resid | beta | 2 | slot_pm | 0.011868 | 0.001920 |
| RL | q_redistribution_v4_resid | beta | 3 | E_w | -0.561221 | 0.160074 |
| RL | q_redistribution_v4_resid | beta | 4 | b_hat | 1.666915 | 0.129617 |
| RL | q_redistribution_v4_resid | beta | 5 | b_tilde | 0.000000 | 1.000000 |
| RL | q_redistribution_v4_resid | beta | 6 | M_Y_anticipated_affect_ewma | 0.311179 | 0.023459 |
| RL | q_redistribution_v4_resid | beta | 7 | M_Y_fourSC_ewma | -0.020883 | 0.022383 |
| RL | q_redistribution_v4_resid | beta | 8 | M_E_pageview_ewma | 0.015075 | 0.006726 |
| RL | q_redistribution_v4_resid | beta | 9 | M_E_fitbit_wear_ewma | -0.157121 | 0.028587 |
| RL | q_redistribution_v4_resid | beta | 10 | M_E_survey_complete_ewma | 0.037620 | 0.030544 |
| RL | q_redistribution_v4_resid | beta | 11 | yesterday_step_count | -0.060443 | 0.031969 |
| RL | q_redistribution_v4_resid | beta | 12 | prior2hour_step_count | -0.011751 | 0.003279 |
| RL | q_redistribution_v4_resid | beta | 13 | active_status_fraction_7days | 0.233027 | 0.125443 |
| RL | q_redistribution_v4_resid | beta | 14 | recent_burden | -0.028844 | 0.014865 |
| RL | q_redistribution_v4_resid | beta | 15 | walk_interaction_7d | 0.027566 | 0.071261 |
| RL | q_redistribution_v4_resid | beta | 16 | A | 0.052684 | 0.004469 |
| RL | q_redistribution_v4_resid | beta | 17 | A*E_w | 0.010089 | 0.007826 |
| RL | q_redistribution_v4_resid | beta | 18 | A*b_hat | 0.002383 | 0.002472 |
| RL | q_redistribution_v4_resid | beta | 19 | A*b_tilde | 0.000000 | 1.000000 |
| RL | q_redistribution_v4_resid | beta | 20 | A*yesterday_step_count | 0.007200 | 0.002141 |
| RL | q_redistribution_v4_resid | beta | 21 | A*prior2hour_step_count | 0.020670 | 0.001285 |
| RL | q_redistribution_v4_resid | beta | 22 | A*active_status_fraction_7days | -0.075873 | 0.004372 |
| RL | q_redistribution_v4_resid | beta | 23 | A*recent_burden | 0.006409 | 0.002418 |
| RL | q_redistribution_v4_resid | beta | 24 | A*walk_interaction_7d | -0.069575 | 0.005112 |
| RL | q_redistribution_v4_resid | beta | 25 | A*weekday_vs_weekend | 0.025050 | 0.013124 |
| RL | q_redistribution_v4_resid | beta | 26 | A*slot_pm | 0.002905 | 0.006955 |

## Joint Modified-TD Priors

| prior_family | model | block | index | feature | prior_mean | prior_variance |
| --- | --- | --- | --- | --- | --- | --- |
| Joint Modified TD | q_td_modify_joint | eta | 0 | eta_intercept | 0.554407 | 0.179593 |
| Joint Modified TD | q_td_modify_joint | eta | 1 | eta_E_w | 0.059705 | 0.111881 |
| Joint Modified TD | q_td_modify_joint | eta | 2 | eta_b_hat | -0.521469 | 0.066424 |
| Joint Modified TD | q_td_modify_joint | eta | 3 | eta_b_tilde | 0.000000 | 1.000001 |
| Joint Modified TD | q_td_modify_joint | beta | 4 | beta_intercept | 0.540189 | 0.123330 |
| Joint Modified TD | q_td_modify_joint | beta | 5 | beta_weekday_vs_weekend | -0.094527 | 0.197179 |
| Joint Modified TD | q_td_modify_joint | beta | 6 | beta_slot_pm | 0.095629 | 0.005312 |
| Joint Modified TD | q_td_modify_joint | beta | 7 | beta_E_w | 0.067651 | 0.070639 |
| Joint Modified TD | q_td_modify_joint | beta | 8 | beta_b_hat | -0.053442 | 0.055738 |
| Joint Modified TD | q_td_modify_joint | beta | 9 | beta_b_tilde | 0.000000 | 1.000001 |
| Joint Modified TD | q_td_modify_joint | beta | 10 | beta_M_Y_anticipated_affect_ewma | 0.703035 | 0.017593 |
| Joint Modified TD | q_td_modify_joint | beta | 11 | beta_M_Y_fourSC_ewma | -0.006101 | 0.013504 |
| Joint Modified TD | q_td_modify_joint | beta | 12 | beta_M_E_pageview_ewma | 0.008166 | 0.000810 |
| Joint Modified TD | q_td_modify_joint | beta | 13 | beta_M_E_fitbit_wear_ewma | -0.341099 | 0.006665 |
| Joint Modified TD | q_td_modify_joint | beta | 14 | beta_M_E_survey_complete_ewma | -0.041315 | 0.005075 |
| Joint Modified TD | q_td_modify_joint | beta | 15 | beta_yesterday_step_count | 0.012540 | 0.003205 |
| Joint Modified TD | q_td_modify_joint | beta | 16 | beta_prior2hour_step_count | -0.007502 | 0.000437 |
| Joint Modified TD | q_td_modify_joint | beta | 17 | beta_active_status_fraction_7days | -0.078415 | 0.036198 |
| Joint Modified TD | q_td_modify_joint | beta | 18 | beta_recent_burden | -0.000914 | 0.001667 |
| Joint Modified TD | q_td_modify_joint | beta | 19 | beta_walk_interaction_7d | -0.003007 | 0.017286 |
| Joint Modified TD | q_td_modify_joint | beta | 20 | beta_A | 0.032329 | 0.001910 |
| Joint Modified TD | q_td_modify_joint | beta | 21 | beta_A*E_w | -0.012319 | 0.004087 |
| Joint Modified TD | q_td_modify_joint | beta | 22 | beta_A*b_hat | -0.067210 | 0.002813 |
| Joint Modified TD | q_td_modify_joint | beta | 23 | beta_A*b_tilde | 0.000000 | 1.000001 |
| Joint Modified TD | q_td_modify_joint | beta | 24 | beta_A*yesterday_step_count | -0.004083 | 0.003798 |
| Joint Modified TD | q_td_modify_joint | beta | 25 | beta_A*prior2hour_step_count | 0.001353 | 0.000996 |
| Joint Modified TD | q_td_modify_joint | beta | 26 | beta_A*active_status_fraction_7days | -0.010598 | 0.001994 |
| Joint Modified TD | q_td_modify_joint | beta | 27 | beta_A*recent_burden | -0.003772 | 0.001928 |
| Joint Modified TD | q_td_modify_joint | beta | 28 | beta_A*walk_interaction_7d | 0.010154 | 0.002569 |
| Joint Modified TD | q_td_modify_joint | beta | 29 | beta_A*weekday_vs_weekend | -0.002960 | 0.005862 |
| Joint Modified TD | q_td_modify_joint | beta | 30 | beta_A*slot_pm | -0.009039 | 0.005255 |

## Pooled PF Regression Coefficients

Coefficients are from the all-user stacked ridge fits used as PF prior means. `p_value` uses ridge sandwich standard errors and is approximate because ridge shrinks coefficients. `identified=False` marks zero-variance design columns; SE / p-values are omitted for those rows.

| model | outcome | index | feature | coefficient | identified | feature_std | std_error | t_stat | p_value | significant_0.05 | significant_0.01 | n_obs | n_features | ridge_alpha | residual_sigma2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fourSC | 4hour_step_norm | 0 | intercept | -0.047490 | False | 0.000000 |  |  |  | False | False | 2890 | 17 | 1.000000 | 0.462795 |
| fourSC | 4hour_step_norm | 1 | stepCountNext4HourLag1 | 0.047787 | True | 0.958607 | 0.024391 | 1.959242 | 0.050181 | False | False | 2890 | 17 | 1.000000 | 0.462795 |
| fourSC | 4hour_step_norm | 2 | yesterdayStepCount | 0.638668 | True | 0.851149 | 0.033768 | 18.913532 | 2.752532e-75 | True | True | 2890 | 17 | 1.000000 | 0.462795 |
| fourSC | 4hour_step_norm | 3 | prior2HourStepCount | 0.148178 | True | 0.934680 | 0.019867 | 7.458406 | 1.153372e-13 | True | True | 2890 | 17 | 1.000000 | 0.462795 |
| fourSC | 4hour_step_norm | 4 | activitySuggestionsSentLast7Days | -0.001962 | True | 1.009565 | 0.012864 | -0.152555 | 0.878760 | False | False | 2890 | 17 | 1.000000 | 0.462795 |
| fourSC | 4hour_step_norm | 5 | activitySuggestionInteractLast7Days | 0.047875 | True | 0.355593 | 0.037002 | 1.293831 | 0.195828 | False | False | 2890 | 17 | 1.000000 | 0.462795 |
| fourSC | 4hour_step_norm | 6 | activeDaysLast7Days | 0.069972 | True | 0.206850 | 0.070150 | 0.997460 | 0.318625 | False | False | 2890 | 17 | 1.000000 | 0.462795 |
| fourSC | 4hour_step_norm | 7 | isWeekend | -0.001310 | True | 0.369659 | 0.034228 | -0.038275 | 0.969471 | False | False | 2890 | 17 | 1.000000 | 0.462795 |
| fourSC | 4hour_step_norm | 8 | decisionTimeSlot | -0.107634 | True | 0.497078 | 0.036672 | -2.935072 | 0.003361 | True | True | 2890 | 17 | 1.000000 | 0.462795 |
| fourSC | 4hour_step_norm | 9 | perceivedUtilityLastWeek | -0.001347 | True | 0.591507 | 0.032982 | -0.040847 | 0.967421 | False | False | 2890 | 17 | 1.000000 | 0.462795 |
| fourSC | 4hour_step_norm | 10 | caeAverageLastWeek | -0.022758 | True | 0.945021 | 0.021012 | -1.083071 | 0.278868 | False | False | 2890 | 17 | 1.000000 | 0.462795 |
| fourSC | 4hour_step_norm | 11 | Ah | 0.015409 | True | 0.499925 | 0.064404 | 0.239262 | 0.810920 | False | False | 2890 | 17 | 1.000000 | 0.462795 |
| fourSC | 4hour_step_norm | 12 | Ah*yesterdayStepCount | 0.152430 | True | 0.692269 | 0.033105 | 4.604469 | 4.314884e-06 | True | True | 2890 | 17 | 1.000000 | 0.462795 |
| fourSC | 4hour_step_norm | 13 | Ah*prior2HourStepCount | -0.034753 | True | 0.655143 | 0.027960 | -1.242991 | 0.213973 | False | False | 2890 | 17 | 1.000000 | 0.462795 |
| fourSC | 4hour_step_norm | 14 | Ah*perceivedUtilityLastWeek | -0.063912 | True | 0.722372 | 0.045548 | -1.403187 | 0.160669 | False | False | 2890 | 17 | 1.000000 | 0.462795 |
| fourSC | 4hour_step_norm | 15 | Ah*caeAverageLastWeek | 0.015111 | True | 0.642740 | 0.028753 | 0.525551 | 0.599241 | False | False | 2890 | 17 | 1.000000 | 0.462795 |
| fourSC | 4hour_step_norm | 16 | Ah*decisionTimeSlot | 0.003800 | True | 0.413354 | 0.051379 | 0.073968 | 0.941041 | False | False | 2890 | 17 | 1.000000 | 0.462795 |
| antic | anticipated_affect_norm | 0 | intercept | 0.287541 | False | 0.000000 |  |  |  | False | False | 817 | 13 | 1.000000 | 0.040192 |
| antic | anticipated_affect_norm | 1 | anticipated_affect_yesterday | 0.211278 | True | 0.329197 | 0.024941 | 8.470951 | 1.151622e-16 | True | True | 817 | 13 | 1.000000 | 0.040192 |
| antic | anticipated_affect_norm | 2 | active_status_fraction_7days | 0.113264 | True | 0.204925 | 0.036695 | 3.086590 | 0.002094 | True | True | 817 | 13 | 1.000000 | 0.040192 |
| antic | anticipated_affect_norm | 3 | is_weekend | -0.030918 | True | 0.370289 | 0.019015 | -1.626024 | 0.104336 | False | False | 817 | 13 | 1.000000 | 0.040192 |
| antic | anticipated_affect_norm | 4 | perceived_utility_lastweek | 0.048915 | True | 0.553492 | 0.022910 | 2.135049 | 0.033059 | True | False | 817 | 13 | 1.000000 | 0.040192 |
| antic | anticipated_affect_norm | 5 | CAE_avg_lastweek | 0.202025 | True | 0.893200 | 0.015530 | 13.008418 | 3.011785e-35 | True | True | 817 | 13 | 1.000000 | 0.040192 |
| antic | anticipated_affect_norm | 6 | recent_burden | 0.009179 | True | 0.996914 | 0.007277 | 1.261409 | 0.207527 | False | False | 817 | 13 | 1.000000 | 0.040192 |
| antic | anticipated_affect_norm | 7 | A0_morning | -0.008123 | True | 0.499865 | 0.034160 | -0.237791 | 0.812104 | False | False | 817 | 13 | 1.000000 | 0.040192 |
| antic | anticipated_affect_norm | 8 | A1_afternoon | 0.047738 | True | 0.499937 | 0.033893 | 1.408478 | 0.159376 | False | False | 817 | 13 | 1.000000 | 0.040192 |
| antic | anticipated_affect_norm | 9 | A0_morning_by_perceived_utility_lastweek | 0.010791 | True | 0.695927 | 0.027194 | 0.396835 | 0.691594 | False | False | 817 | 13 | 1.000000 | 0.040192 |
| antic | anticipated_affect_norm | 10 | A1_afternoon_by_perceived_utility_lastweek | -0.025311 | True | 0.708424 | 0.027037 | -0.936175 | 0.349464 | False | False | 817 | 13 | 1.000000 | 0.040192 |
| antic | anticipated_affect_norm | 11 | A0_morning_by_CAE_avg_lastweek | -0.009347 | True | 0.639909 | 0.017349 | -0.538761 | 0.590201 | False | False | 817 | 13 | 1.000000 | 0.040192 |
| antic | anticipated_affect_norm | 12 | A1_afternoon_by_CAE_avg_lastweek | -0.018554 | True | 0.614166 | 0.017351 | -1.069310 | 0.285251 | False | False | 817 | 13 | 1.000000 | 0.040192 |
| CAE | CAE_avg_norm | 0 | intercept | -0.094983 | False | 0.000000 |  |  |  | False | False | 213 | 4 | 1.000000 | 0.163622 |
| CAE | CAE_avg_norm | 1 | CAE_avg_lastweek | 0.884206 | True | 0.977703 | 0.037994 | 23.272212 | 6.120177e-60 | True | True | 213 | 4 | 1.000000 | 0.163622 |
| CAE | CAE_avg_norm | 2 | fourSC_ewma | -0.039165 | True | 0.524645 | 0.052670 | -0.743587 | 0.457961 | False | False | 213 | 4 | 1.000000 | 0.163622 |
| CAE | CAE_avg_norm | 3 | anticipated_affect_ewma | 0.188655 | True | 0.334064 | 0.105242 | 1.792593 | 0.074484 | False | False | 213 | 4 | 1.000000 | 0.163622 |

## Pooled PF GEE Coefficients

Population-averaged GEE fits on the same stacked PF designs, clustered by `ParticipantIdentifier`. `p_value` uses GEE sandwich standard errors with exchangeable working correlation. `identified=False` marks zero-variance design columns; SE / p-values are omitted for those rows. These are for inference/audit only; PF priors still use ridge.

| coefficient | robust_se | z_stat | p_value | significant_0.05 | significant_0.01 | model | outcome | index | feature | identified | feature_std | n_obs | n_clusters | n_features | working_correlation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| -0.151181 |  |  |  | False | False | fourSC | 4hour_step_norm | 0 | intercept | False | 0.000000 | 2890 | 28 | 17 | exchangeable |
| -0.134993 | 0.039300 | -3.434947 | 0.000593 | True | True | fourSC | 4hour_step_norm | 1 | stepCountNext4HourLag1 | True | 0.958607 | 2890 | 28 | 17 | exchangeable |
| 0.191600 | 0.058633 | 3.267789 | 0.001084 | True | True | fourSC | 4hour_step_norm | 2 | yesterdayStepCount | True | 0.851149 | 2890 | 28 | 17 | exchangeable |
| 0.148872 | 0.032902 | 4.524732 | 6.047197e-06 | True | True | fourSC | 4hour_step_norm | 3 | prior2HourStepCount | True | 0.934680 | 2890 | 28 | 17 | exchangeable |
| 0.016536 | 0.012303 | 1.344035 | 0.178937 | False | False | fourSC | 4hour_step_norm | 4 | activitySuggestionsSentLast7Days | True | 1.009565 | 2890 | 28 | 17 | exchangeable |
| 0.118059 | 0.058338 | 2.023716 | 0.042999 | True | False | fourSC | 4hour_step_norm | 5 | activitySuggestionInteractLast7Days | True | 0.355593 | 2890 | 28 | 17 | exchangeable |
| 0.205419 | 0.110951 | 1.851435 | 0.064107 | False | False | fourSC | 4hour_step_norm | 6 | activeDaysLast7Days | True | 0.206850 | 2890 | 28 | 17 | exchangeable |
| 0.000334 | 0.042056 | 0.007939 | 0.993666 | False | False | fourSC | 4hour_step_norm | 7 | isWeekend | True | 0.369659 | 2890 | 28 | 17 | exchangeable |
| -0.114037 | 0.076840 | -1.484083 | 0.137787 | False | False | fourSC | 4hour_step_norm | 8 | decisionTimeSlot | True | 0.497078 | 2890 | 28 | 17 | exchangeable |
| -0.061578 | 0.034059 | -1.808003 | 0.070606 | False | False | fourSC | 4hour_step_norm | 9 | perceivedUtilityLastWeek | True | 0.591507 | 2890 | 28 | 17 | exchangeable |
| 0.029923 | 0.029966 | 0.998580 | 0.317998 | False | False | fourSC | 4hour_step_norm | 10 | caeAverageLastWeek | True | 0.945021 | 2890 | 28 | 17 | exchangeable |
| 0.034879 | 0.044136 | 0.790276 | 0.429367 | False | False | fourSC | 4hour_step_norm | 11 | Ah | True | 0.499925 | 2890 | 28 | 17 | exchangeable |
| 0.057715 | 0.043854 | 1.316077 | 0.188148 | False | False | fourSC | 4hour_step_norm | 12 | Ah*yesterdayStepCount | True | 0.692269 | 2890 | 28 | 17 | exchangeable |
| -0.030740 | 0.030877 | -0.995570 | 0.319459 | False | False | fourSC | 4hour_step_norm | 13 | Ah*prior2HourStepCount | True | 0.655143 | 2890 | 28 | 17 | exchangeable |
| -0.047165 | 0.028151 | -1.675446 | 0.093847 | False | False | fourSC | 4hour_step_norm | 14 | Ah*perceivedUtilityLastWeek | True | 0.722372 | 2890 | 28 | 17 | exchangeable |
| -0.003061 | 0.027716 | -0.110446 | 0.912055 | False | False | fourSC | 4hour_step_norm | 15 | Ah*caeAverageLastWeek | True | 0.642740 | 2890 | 28 | 17 | exchangeable |
| -0.014545 | 0.052604 | -0.276509 | 0.782157 | False | False | fourSC | 4hour_step_norm | 16 | Ah*decisionTimeSlot | True | 0.413354 | 2890 | 28 | 17 | exchangeable |
| 0.361611 |  |  |  | False | False | antic | anticipated_affect_norm | 0 | intercept | False | 0.000000 | 817 | 28 | 13 | exchangeable |
| 0.079386 | 0.033822 | 2.347160 | 0.018917 | True | False | antic | anticipated_affect_norm | 1 | anticipated_affect_yesterday | True | 0.329197 | 817 | 28 | 13 | exchangeable |
| 0.121866 | 0.037352 | 3.262616 | 0.001104 | True | True | antic | anticipated_affect_norm | 2 | active_status_fraction_7days | True | 0.204925 | 817 | 28 | 13 | exchangeable |
| -0.008364 | 0.021677 | -0.385850 | 0.699608 | False | False | antic | anticipated_affect_norm | 3 | is_weekend | True | 0.370289 | 817 | 28 | 13 | exchangeable |
| 0.033201 | 0.030718 | 1.080832 | 0.279772 | False | False | antic | anticipated_affect_norm | 4 | perceived_utility_lastweek | True | 0.553492 | 817 | 28 | 13 | exchangeable |
| 0.121161 | 0.035692 | 3.394637 | 0.000687 | True | True | antic | anticipated_affect_norm | 5 | CAE_avg_lastweek | True | 0.893200 | 817 | 28 | 13 | exchangeable |
| 0.001948 | 0.006112 | 0.318794 | 0.749882 | False | False | antic | anticipated_affect_norm | 6 | recent_burden | True | 0.996914 | 817 | 28 | 13 | exchangeable |
| -0.017103 | 0.022058 | -0.775368 | 0.438122 | False | False | antic | anticipated_affect_norm | 7 | A0_morning | True | 0.499865 | 817 | 28 | 13 | exchangeable |
| 0.020683 | 0.022532 | 0.917948 | 0.358646 | False | False | antic | anticipated_affect_norm | 8 | A1_afternoon | True | 0.499937 | 817 | 28 | 13 | exchangeable |
| 0.016041 | 0.019003 | 0.844128 | 0.398598 | False | False | antic | anticipated_affect_norm | 9 | A0_morning_by_perceived_utility_lastweek | True | 0.695927 | 817 | 28 | 13 | exchangeable |
| -0.003412 | 0.019387 | -0.175986 | 0.860305 | False | False | antic | anticipated_affect_norm | 10 | A1_afternoon_by_perceived_utility_lastweek | True | 0.708424 | 817 | 28 | 13 | exchangeable |
| 0.008639 | 0.012833 | 0.673213 | 0.500812 | False | False | antic | anticipated_affect_norm | 11 | A0_morning_by_CAE_avg_lastweek | True | 0.639909 | 817 | 28 | 13 | exchangeable |
| -0.014215 | 0.015537 | -0.914936 | 0.360225 | False | False | antic | anticipated_affect_norm | 12 | A1_afternoon_by_CAE_avg_lastweek | True | 0.614166 | 817 | 28 | 13 | exchangeable |
| -0.076974 |  |  |  | False | False | CAE | CAE_avg_norm | 0 | intercept | False | 0.000000 | 213 | 28 | 4 | exchangeable |
| 0.931634 | 0.040771 | 22.850182 | 1.455311e-115 | True | True | CAE | CAE_avg_norm | 1 | CAE_avg_lastweek | True | 0.977703 | 213 | 28 | 4 | exchangeable |
| -0.033553 | 0.030416 | -1.103125 | 0.269973 | False | False | CAE | CAE_avg_norm | 2 | fourSC_ewma | True | 0.524645 | 213 | 28 | 4 | exchangeable |
| 0.137923 | 0.116704 | 1.181814 | 0.237279 | False | False | CAE | CAE_avg_norm | 3 | anticipated_affect_ewma | True | 0.334064 | 213 | 28 | 4 | exchangeable |

## Pooled RL Q GEE Coefficients

Gaussian GEE on the final fitted-Q regression from pooled FQI (``phi_obs`` vs bootstrap targets), clustered by participant. ``identified=False`` marks structurally unused features with zero design variance (e.g. ``b_tilde``, masked day-6 mediators); SE / p-values are omitted for those rows. RL priors still use ridge-FQI for ``mu_0_micro``; all-zero design columns get ``UNIDENTIFIED_PRIOR_VAR`` instead of the ``MIN_SIGMA2`` floor.

| coefficient | robust_se | z_stat | p_value | significant_0.05 | significant_0.01 | model | outcome | index | feature | identified | feature_std | n_obs | n_clusters | n_features | working_correlation | fqi_iters | ridge_alpha | residual_sigma2 | block |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.048799 |  |  |  | False | False | q_no_td_modify | fqi_target | 0 | intercept | False | 0.000000 | 3360 | 28 | 27 | exchangeable | 25 | 1.000000 | 0.029763 | beta |
| -0.069117 | 0.036789 | -1.878730 | 0.060281 | False | False | q_no_td_modify | fqi_target | 1 | weekday_vs_weekend | True | 0.372678 | 3360 | 28 | 27 | exchangeable | 25 | 1.000000 | 0.029763 | beta |
| -0.008293 | 0.014193 | -0.584278 | 0.559033 | False | False | q_no_td_modify | fqi_target | 2 | slot_pm | True | 0.500000 | 3360 | 28 | 27 | exchangeable | 25 | 1.000000 | 0.029763 | beta |
| 0.001532 | 0.005767 | 0.265653 | 0.790507 | False | False | q_no_td_modify | fqi_target | 3 | E_w | True | 0.577290 | 3360 | 28 | 27 | exchangeable | 25 | 1.000000 | 0.029763 | beta |
| 1.153230 | 0.005921 | 194.777712 | 0.000000 | True | True | q_no_td_modify | fqi_target | 4 | b_hat | True | 0.919943 | 3360 | 28 | 27 | exchangeable | 25 | 1.000000 | 0.029763 | beta |
| 1.749831e-16 |  |  |  | False | False | q_no_td_modify | fqi_target | 5 | b_tilde | False | 0.000000 | 3360 | 28 | 27 | exchangeable | 25 | 1.000000 | 0.029763 | beta |
| 0.285147 | 0.015756 | 18.097723 | 3.321149e-73 | True | True | q_no_td_modify | fqi_target | 6 | M_Y_anticipated_affect_ewma | True | 0.311549 | 3360 | 28 | 27 | exchangeable | 25 | 1.000000 | 0.029763 | beta |
| -0.008936 | 0.005985 | -1.493012 | 0.135434 | False | False | q_no_td_modify | fqi_target | 7 | M_Y_fourSC_ewma | True | 0.969806 | 3360 | 28 | 27 | exchangeable | 25 | 1.000000 | 0.029763 | beta |
| 0.000210 | 0.001653 | 0.127036 | 0.898912 | False | False | q_no_td_modify | fqi_target | 8 | M_E_pageview_ewma | True | 1.107588 | 3360 | 28 | 27 | exchangeable | 25 | 1.000000 | 0.029763 | beta |
| -0.129825 | 0.008077 | -16.073383 | 3.920922e-58 | True | True | q_no_td_modify | fqi_target | 9 | M_E_fitbit_wear_ewma | True | 0.412251 | 3360 | 28 | 27 | exchangeable | 25 | 1.000000 | 0.029763 | beta |
| -0.037168 | 0.005934 | -6.263528 | 3.763642e-10 | True | True | q_no_td_modify | fqi_target | 10 | M_E_survey_complete_ewma | True | 0.411996 | 3360 | 28 | 27 | exchangeable | 25 | 1.000000 | 0.029763 | beta |
| -0.020089 | 0.006440 | -3.119470 | 0.001812 | True | True | q_no_td_modify | fqi_target | 11 | yesterday_step_count | True | 1.002896 | 3360 | 28 | 27 | exchangeable | 25 | 1.000000 | 0.029763 | beta |
| -0.012522 | 0.004540 | -2.758442 | 0.005808 | True | True | q_no_td_modify | fqi_target | 12 | prior2hour_step_count | True | 0.923123 | 3360 | 28 | 27 | exchangeable | 25 | 1.000000 | 0.029763 | beta |
| 0.102694 | 0.013401 | 7.662943 | 1.817202e-14 | True | True | q_no_td_modify | fqi_target | 13 | active_status_fraction_7days | True | 0.209797 | 3360 | 28 | 27 | exchangeable | 25 | 1.000000 | 0.029763 | beta |
| -0.013011 | 0.003923 | -3.316672 | 0.000911 | True | True | q_no_td_modify | fqi_target | 14 | recent_burden | True | 0.970901 | 3360 | 28 | 27 | exchangeable | 25 | 1.000000 | 0.029763 | beta |
| -0.023584 | 0.009733 | -2.423190 | 0.015385 | True | False | q_no_td_modify | fqi_target | 15 | walk_interaction_7d | True | 0.356911 | 3360 | 28 | 27 | exchangeable | 25 | 1.000000 | 0.029763 | beta |
| 0.010832 | 0.028296 | 0.382803 | 0.701866 | False | False | q_no_td_modify | fqi_target | 16 | A | True | 0.499915 | 3360 | 28 | 27 | exchangeable | 25 | 1.000000 | 0.029763 | beta |
| -0.010089 | 0.013404 | -0.752680 | 0.451642 | False | False | q_no_td_modify | fqi_target | 17 | A*E_w | True | 0.734168 | 3360 | 28 | 27 | exchangeable | 25 | 1.000000 | 0.029763 | beta |
| 0.003416 | 0.011957 | 0.285686 | 0.775119 | False | False | q_no_td_modify | fqi_target | 18 | A*b_hat | True | 0.633764 | 3360 | 28 | 27 | exchangeable | 25 | 1.000000 | 0.029763 | beta |
| 1.251922e-16 |  |  |  | False | False | q_no_td_modify | fqi_target | 19 | A*b_tilde | False | 0.000000 | 3360 | 28 | 27 | exchangeable | 25 | 1.000000 | 0.029763 | beta |
| 0.000922 | 0.004701 | 0.196114 | 0.844521 | False | False | q_no_td_modify | fqi_target | 20 | A*yesterday_step_count | True | 0.771463 | 3360 | 28 | 27 | exchangeable | 25 | 1.000000 | 0.029763 | beta |
| 0.003934 | 0.005033 | 0.781603 | 0.434448 | False | False | q_no_td_modify | fqi_target | 21 | A*prior2hour_step_count | True | 0.643886 | 3360 | 28 | 27 | exchangeable | 25 | 1.000000 | 0.029763 | beta |
| -0.007643 | 0.027425 | -0.278695 | 0.780479 | False | False | q_no_td_modify | fqi_target | 22 | A*active_status_fraction_7days | True | 0.364389 | 3360 | 28 | 27 | exchangeable | 25 | 1.000000 | 0.029763 | beta |
| -0.003336 | 0.005508 | -0.605647 | 0.544749 | False | False | q_no_td_modify | fqi_target | 23 | A*recent_burden | True | 0.689166 | 3360 | 28 | 27 | exchangeable | 25 | 1.000000 | 0.029763 | beta |
| -0.001706 | 0.017955 | -0.094992 | 0.924321 | False | False | q_no_td_modify | fqi_target | 24 | A*walk_interaction_7d | True | 0.336223 | 3360 | 28 | 27 | exchangeable | 25 | 1.000000 | 0.029763 | beta |
| 0.042514 | 0.031186 | 1.363237 | 0.172808 | False | False | q_no_td_modify | fqi_target | 25 | A*weekday_vs_weekend | True | 0.281695 | 3360 | 28 | 27 | exchangeable | 25 | 1.000000 | 0.029763 | beta |
| 0.008346 | 0.010927 | 0.763816 | 0.444977 | False | False | q_no_td_modify | fqi_target | 26 | A*slot_pm | True | 0.432496 | 3360 | 28 | 27 | exchangeable | 25 | 1.000000 | 0.029763 | beta |
