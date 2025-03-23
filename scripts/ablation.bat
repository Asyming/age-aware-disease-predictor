# TODO: 确定好到底用ad还是ad_new
# full (8 + 6 + 5)
python scripts/train4.py --exp_name full --mode teacher --teacher_type MLP --age_threshold 65 --data_dir ./data/ad;
python scripts/train4.py --exp_name full --mode student --teacher_type MLP --student_type MLP --age_threshold 65 --data_dir ./data/ad;
python scripts/train4.py --exp_name full --mode teacher --teacher_type AgeAwareMLP1 --age_threshold 65 --data_dir ./data/ad;
python scripts/train4.py --exp_name full --mode student --teacher_type AgeAwareMLP1 --student_type AgeAwareMLP1 --age_threshold 65 --data_dir ./data/ad;
python scripts/train4.py --exp_name full --mode student --teacher_type AgeAwareMLP1 --student_type MLP --age_threshold 65 --data_dir ./data/ad;
python scripts/train4.py --exp_name full --mode teacher --teacher_type AgeAwareMLP2 --age_threshold 65 --data_dir ./data/ad;
python scripts/train4.py --exp_name full --mode student --teacher_type AgeAwareMLP2 --student_type AgeAwareMLP2 --age_threshold 65 --data_dir ./data/ad;
python scripts/train4.py --exp_name full --mode student --teacher_type AgeAwareMLP2 --student_type MLP --age_threshold 65 --data_dir ./data/ad;

python scripts/train4.py --exp_name full --mode teacher --teacher_type UGP_v1 --data_dir ./data/ad_new;
python scripts/train4.py --exp_name full --mode student --teacher_type UGP_v1 --student_type UGP_v1 --data_dir ./data/ad_new;
python scripts/train4.py --exp_name full --mode student --teacher_type UGP_v1 --student_type MLP --data_dir ./data/ad_new;
python scripts/train4.py --exp_name full --mode teacher --teacher_type UGP_v2 --data_dir ./data/ad_new;
python scripts/train4.py --exp_name full --mode student --teacher_type UGP_v2 --student_type UGP_v2 --data_dir ./data/ad_new;
python scripts/train4.py --exp_name full --mode student --teacher_type UGP_v2 --student_type MLP --data_dir ./data/ad_new;

python scripts/train_classical.py --exp_name full --mode teacher --model_type LinearRegression --data_dir ./data/ad;
python scripts/train_classical.py --exp_name full --mode teacher --model_type XGBoostClassifier --data_dir ./data/ad;
python scripts/train_classical.py --exp_name full --mode teacher --model_type XGBoostRegressor --data_dir ./data/ad;
python scripts/train_classical.py --exp_name full --mode teacher --model_type LightGBMClassifier --data_dir ./data/ad;
python scripts/train_classical.py --exp_name full --mode teacher --model_type LightGBMRegressor --data_dir ./data/ad;


# zero (6)
python scripts/train4.py --exp_name zero --mode teacher --teacher_type AgeAwareMLP1 --age_threshold 65 --data_dir ./data/ad --age1_use_consist false --age1_use_adversarial false;
python scripts/train4.py --exp_name zero --mode student --teacher_type AgeAwareMLP1 --student_type AgeAwareMLP1 --age_threshold 65 --data_dir ./data/ad --age1_use_consist false --age1_use_adversarial false;
python scripts/train4.py --exp_name zero --mode student --teacher_type AgeAwareMLP1 --student_type MLP --age_threshold 65 --data_dir ./data/ad --age1_use_consist false --age1_use_adversarial false;
python scripts/train4.py --exp_name zero --mode teacher --teacher_type AgeAwareMLP2 --age_threshold 65 --data_dir ./data/ad --age2_use_consist false --age2_use_ageloss false --age2_use_disentangle false;
python scripts/train4.py --exp_name zero --mode student --teacher_type AgeAwareMLP2 --student_type AgeAwareMLP2 --age_threshold 65 --data_dir ./data/ad --age2_use_consist false --age2_use_ageloss false --age2_use_disentangle false;
python scripts/train4.py --exp_name zero --mode student --teacher_type AgeAwareMLP2 --student_type MLP --age_threshold 65 --data_dir ./data/ad --age2_use_consist false --age2_use_ageloss false --age2_use_disentangle false;

# ab_age1_adv (3)
python scripts/train4.py --exp_name ab_age1_adv --mode teacher --teacher_type AgeAwareMLP1 --age_threshold 65 --data_dir ./data/ad --age1_use_adversarial false;
python scripts/train4.py --exp_name ab_age1_adv --mode student --teacher_type AgeAwareMLP1 --student_type AgeAwareMLP1 --age_threshold 65 --data_dir ./data/ad --age1_use_adversarial false;
python scripts/train4.py --exp_name ab_age1_adv --mode student --teacher_type AgeAwareMLP1 --student_type MLP --age_threshold 65 --data_dir ./data/ad --age1_use_adversarial false;

# ab_age1_consist (3)
python scripts/train4.py --exp_name ab_age1_consist --mode teacher --teacher_type AgeAwareMLP1 --age_threshold 65 --data_dir ./data/ad --age1_use_consist false;
python scripts/train4.py --exp_name ab_age1_consist --mode student --teacher_type AgeAwareMLP1 --student_type AgeAwareMLP1 --age_threshold 65 --data_dir ./data/ad --age1_use_consist false;
python scripts/train4.py --exp_name ab_age1_consist --mode student --teacher_type AgeAwareMLP1 --student_type MLP --age_threshold 65 --data_dir ./data/ad --age1_use_consist false;

# ab_age2_ageloss (3)
python scripts/train4.py --exp_name ab_age2_ageloss --mode teacher --teacher_type AgeAwareMLP2 --age_threshold 65 --data_dir ./data/ad --age2_use_ageloss false;
python scripts/train4.py --exp_name ab_age2_ageloss --mode student --teacher_type AgeAwareMLP2 --student_type AgeAwareMLP2 --age_threshold 65 --data_dir ./data/ad --age2_use_ageloss false;
python scripts/train4.py --exp_name ab_age2_ageloss --mode student --teacher_type AgeAwareMLP2 --student_type MLP --age_threshold 65 --data_dir ./data/ad --age2_use_ageloss false;

# ab_age2_disentangle (3)
python scripts/train4.py --exp_name ab_age2_disentangle --mode teacher --teacher_type AgeAwareMLP2 --age_threshold 65 --data_dir ./data/ad --age2_use_disentangle false;
python scripts/train4.py --exp_name ab_age2_disentangle --mode student --teacher_type AgeAwareMLP2 --student_type AgeAwareMLP2 --age_threshold 65 --data_dir ./data/ad --age2_use_disentangle false;
python scripts/train4.py --exp_name ab_age2_disentangle --mode student --teacher_type AgeAwareMLP2 --student_type MLP --age_threshold 65 --data_dir ./data/ad --age2_use_disentangle false;

# ab_age2_consist (3)
python scripts/train4.py --exp_name ab_age2_consist --mode teacher --teacher_type AgeAwareMLP2 --age_threshold 65 --data_dir ./data/ad --age2_use_consist false;
python scripts/train4.py --exp_name ab_age2_consist --mode student --teacher_type AgeAwareMLP2 --student_type AgeAwareMLP2 --age_threshold 65 --data_dir ./data/ad --age2_use_consist false;
python scripts/train4.py --exp_name ab_age2_consist --mode student --teacher_type AgeAwareMLP2 --student_type MLP --age_threshold 65 --data_dir ./data/ad --age2_use_consist false;

# ab_age2_ageloss_disentangle (3)
python scripts/train4.py --exp_name ab_age2_ageloss_disentangle --mode teacher --teacher_type AgeAwareMLP2 --age_threshold 65 --data_dir ./data/ad --age2_use_ageloss false --age2_use_disentangle false;
python scripts/train4.py --exp_name ab_age2_ageloss_disentangle --mode student --teacher_type AgeAwareMLP2 --student_type AgeAwareMLP2 --age_threshold 65 --data_dir ./data/ad --age2_use_ageloss false --age2_use_disentangle false;
python scripts/train4.py --exp_name ab_age2_ageloss_disentangle --mode student --teacher_type AgeAwareMLP2 --student_type MLP --age_threshold 65 --data_dir ./data/ad --age2_use_ageloss false --age2_use_disentangle false;

# student_enhanced_lr3 (5 + 4)
python scripts/train4.py --exp_name student_enhanced_lr3 --mode student --teacher_type MLP --student_type MLP --age_threshold 65 --data_dir ./data/ad --teacher_model_exp_name full --lr 0.001;
python scripts/train4.py --exp_name student_enhanced_lr3 --mode student --teacher_type AgeAwareMLP1 --student_type AgeAwareMLP1 --age_threshold 65 --data_dir ./data/ad --teacher_model_exp_name full --lr 0.001;
python scripts/train4.py --exp_name student_enhanced_lr3 --mode student --teacher_type AgeAwareMLP1 --student_type MLP --age_threshold 65 --data_dir ./data/ad --teacher_model_exp_name full --lr 0.001;
python scripts/train4.py --exp_name student_enhanced_lr3 --mode student --teacher_type AgeAwareMLP2 --student_type AgeAwareMLP2 --age_threshold 65 --data_dir ./data/ad --teacher_model_exp_name full --lr 0.001;
python scripts/train4.py --exp_name student_enhanced_lr3 --mode student --teacher_type AgeAwareMLP2 --student_type MLP --age_threshold 65 --data_dir ./data/ad --teacher_model_exp_name full --lr 0.001;

python scripts/train4.py --exp_name student_enhanced_lr3 --mode student --teacher_type UGP_v1 --student_type UGP_v1 --data_dir ./data/ad_new --teacher_model_exp_name full --lr 0.001;
python scripts/train4.py --exp_name student_enhanced_lr3 --mode student --teacher_type UGP_v1 --student_type MLP --data_dir ./data/ad_new --teacher_model_exp_name full --lr 0.001;
python scripts/train4.py --exp_name student_enhanced_lr3 --mode student --teacher_type UGP_v2 --student_type UGP_v2 --data_dir ./data/ad_new --teacher_model_exp_name full --lr 0.001;
python scripts/train4.py --exp_name student_enhanced_lr3 --mode student --teacher_type UGP_v2 --student_type MLP --data_dir ./data/ad_new --teacher_model_exp_name full --lr 0.001;

# test_cumulative_rate (6)
python scripts/train4.py --exp_name test_cumulative_rate --mode teacher --teacher_type AgeAwareMLP1 --age_threshold 65 --data_dir ./data/ad --use_cumulative_rate;
python scripts/train4.py --exp_name test_cumulative_rate --mode student --teacher_type AgeAwareMLP1 --student_type AgeAwareMLP1 --age_threshold 65 --data_dir ./data/ad --use_cumulative_rate;
python scripts/train4.py --exp_name test_cumulative_rate --mode student --teacher_type AgeAwareMLP1 --student_type MLP --age_threshold 65 --data_dir ./data/ad --use_cumulative_rate;
python scripts/train4.py --exp_name test_cumulative_rate --mode teacher --teacher_type AgeAwareMLP2 --age_threshold 65 --data_dir ./data/ad --use_cumulative_rate;
python scripts/train4.py --exp_name test_cumulative_rate --mode student --teacher_type AgeAwareMLP2 --student_type AgeAwareMLP2 --age_threshold 65 --data_dir ./data/ad --use_cumulative_rate;
python scripts/train4.py --exp_name test_cumulative_rate --mode student --teacher_type AgeAwareMLP2 --student_type MLP --age_threshold 65 --data_dir ./data/ad --use_cumulative_rate;

# test_cumulative_rate_lr3 (6)
python scripts/train4.py --exp_name test_cumulative_rate_lr3 --mode teacher --teacher_type AgeAwareMLP1 --age_threshold 65 --data_dir ./data/ad --use_cumulative_rate --lr 0.001;
python scripts/train4.py --exp_name test_cumulative_rate_lr3 --mode student --teacher_type AgeAwareMLP1 --student_type AgeAwareMLP1 --age_threshold 65 --data_dir ./data/ad --use_cumulative_rate --lr 0.001;
python scripts/train4.py --exp_name test_cumulative_rate_lr3 --mode student --teacher_type AgeAwareMLP1 --student_type MLP --age_threshold 65 --data_dir ./data/ad --use_cumulative_rate --lr 0.001;
python scripts/train4.py --exp_name test_cumulative_rate_lr3 --mode teacher --teacher_type AgeAwareMLP2 --age_threshold 65 --data_dir ./data/ad --use_cumulative_rate --lr 0.001;
python scripts/train4.py --exp_name test_cumulative_rate_lr3 --mode student --teacher_type AgeAwareMLP2 --student_type AgeAwareMLP2 --age_threshold 65 --data_dir ./data/ad --use_cumulative_rate --lr 0.001;
python scripts/train4.py --exp_name test_cumulative_rate_lr3 --mode student --teacher_type AgeAwareMLP2 --student_type MLP --age_threshold 65 --data_dir ./data/ad --use_cumulative_rate --lr 0.001;

# test_classical (6) # teacher: classical, student: classical
python scripts/train_classical.py --mode teacher --data_dir ./data/ad --model_type XGBoost --exp_name test_classical;
python scripts/train_classical.py --mode student --data_dir ./data/ad --model_type XGBoost --teacher_type XGBoost --teacher_model_exp_name test_classical --exp_name test_classical;
python scripts/train_classical.py --mode teacher --data_dir ./data/ad --model_type LinearRegression --exp_name test_classical;
python scripts/train_classical.py --mode student --data_dir ./data/ad --model_type LinearRegression --teacher_type LinearRegression --teacher_model_exp_name test_classical --exp_name test_classical;
python scripts/train_classical.py --mode teacher --data_dir ./data/ad --model_type LightGBM --exp_name test_classical;
python scripts/train_classical.py --mode student --data_dir ./data/ad --model_type LightGBM --teacher_type LightGBM --teacher_model_exp_name test_classical --exp_name test_classical;

# test_MLP_classical (3) # teacher: MLP, student: classical
python scripts/train_classical.py --mode student --data_dir ./data/ad --model_type XGBoost --train4_teacher_exp_name full --exp_name test_MLP_classical --train4_teacher_type MLP;
python scripts/train_classical.py --mode student --data_dir ./data/ad --model_type LinearRegression --train4_teacher_exp_name full --exp_name test_MLP_classical --train4_teacher_type MLP;
python scripts/train_classical.py --mode student --data_dir ./data/ad --model_type LightGBM --train4_teacher_exp_name full --exp_name test_MLP_classical --train4_teacher_type MLP;

# test_AgeAwareMLP1_classical (3) # teacher: AgeAwareMLP1, student: classical
python scripts/train_classical.py --mode student --data_dir ./data/ad --model_type XGBoost --train4_teacher_exp_name full --exp_name test_AgeAwareMLP1_classical --train4_teacher_type AgeAwareMLP1;
python scripts/train_classical.py --mode student --data_dir ./data/ad --model_type LinearRegression --train4_teacher_exp_name full --exp_name test_AgeAwareMLP1_classical --train4_teacher_type AgeAwareMLP1;
python scripts/train_classical.py --mode student --data_dir ./data/ad --model_type LightGBM --train4_teacher_exp_name full --exp_name test_AgeAwareMLP1_classical --train4_teacher_type AgeAwareMLP1;

# test_AgeAwareMLP2_classical (3) # teacher: AgeAwareMLP2, student: classical
python scripts/train_classical.py --mode student --data_dir ./data/ad --model_type XGBoost --train4_teacher_exp_name full --exp_name test_AgeAwareMLP2_classical --train4_teacher_type AgeAwareMLP2;
python scripts/train_classical.py --mode student --data_dir ./data/ad --model_type LinearRegression --train4_teacher_exp_name full --exp_name test_AgeAwareMLP2_classical --train4_teacher_type AgeAwareMLP2;
python scripts/train_classical.py --mode student --data_dir ./data/ad --model_type LightGBM --train4_teacher_exp_name full --exp_name test_AgeAwareMLP2_classical --train4_teacher_type AgeAwareMLP2;