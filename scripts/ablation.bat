# full (8)
python scripts/train4.py --exp_name full --mode teacher --teacher_type MLP --age_threshold 65 --data_dir ./data/ad;
python scripts/train4.py --exp_name full --mode student --teacher_type MLP --student_type MLP --age_threshold 65 --data_dir ./data/ad;
python scripts/train4.py --exp_name full --mode teacher --teacher_type AgeAwareMLP1 --age_threshold 65 --data_dir ./data/ad;
python scripts/train4.py --exp_name full --mode student --teacher_type AgeAwareMLP1 --student_type AgeAwareMLP1 --age_threshold 65 --data_dir ./data/ad;
python scripts/train4.py --exp_name full --mode student --teacher_type AgeAwareMLP1 --student_type MLP --age_threshold 65 --data_dir ./data/ad;
python scripts/train4.py --exp_name full --mode teacher --teacher_type AgeAwareMLP2 --age_threshold 65 --data_dir ./data/ad;
python scripts/train4.py --exp_name full --mode student --teacher_type AgeAwareMLP2 --student_type AgeAwareMLP2 --age_threshold 65 --data_dir ./data/ad;
python scripts/train4.py --exp_name full --mode student --teacher_type AgeAwareMLP2 --student_type MLP --age_threshold 65 --data_dir ./data/ad;

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

# student_enhanced_lr3 (5)
python scripts/train4.py --exp_name student_enhanced_lr3 --mode student --teacher_type MLP --student_type MLP --age_threshold 65 --data_dir ./data/ad --teacher_model_exp_name full --lr 0.001;
python scripts/train4.py --exp_name student_enhanced_lr3 --mode student --teacher_type AgeAwareMLP1 --student_type AgeAwareMLP1 --age_threshold 65 --data_dir ./data/ad --teacher_model_exp_name full --lr 0.001;
python scripts/train4.py --exp_name student_enhanced_lr3 --mode student --teacher_type AgeAwareMLP1 --student_type MLP --age_threshold 65 --data_dir ./data/ad --teacher_model_exp_name full --lr 0.001;
python scripts/train4.py --exp_name student_enhanced_lr3 --mode student --teacher_type AgeAwareMLP2 --student_type AgeAwareMLP2 --age_threshold 65 --data_dir ./data/ad --teacher_model_exp_name full --lr 0.001;
python scripts/train4.py --exp_name student_enhanced_lr3 --mode student --teacher_type AgeAwareMLP2 --student_type MLP --age_threshold 65 --data_dir ./data/ad --teacher_model_exp_name full --lr 0.001;

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

