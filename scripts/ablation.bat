python scripts/train4.py --exp_name full --mode teacher --teacher_type MLP --age_threshold 65 --data_dir ./data/ad;
python scripts/train4.py --exp_name full --mode student --teacher_type MLP --student_type MLP --age_threshold 65 --data_dir ./data/ad;
python scripts/train4.py --exp_name full --mode teacher --teacher_type AgeAwareMLP1 --age_threshold 65 --data_dir ./data/ad;
python scripts/train4.py --exp_name full --mode student --teacher_type AgeAwareMLP1 --student_type AgeAwareMLP1 --age_threshold 65 --data_dir ./data/ad;
python scripts/train4.py --exp_name full --mode student --teacher_type AgeAwareMLP1 --student_type MLP --age_threshold 65 --data_dir ./data/ad;
python scripts/train4.py --exp_name full --mode teacher --teacher_type AgeAwareMLP2 --age_threshold 65 --data_dir ./data/ad;
python scripts/train4.py --exp_name full --mode student --teacher_type AgeAwareMLP2 --student_type AgeAwareMLP2 --age_threshold 65 --data_dir ./data/ad;
python scripts/train4.py --exp_name full --mode student --teacher_type AgeAwareMLP2 --student_type MLP --age_threshold 65 --data_dir ./data/ad;


python scripts/train4.py --exp_name zero --mode teacher --teacher_type AgeAwareMLP1 --age_threshold 65 --data_dir ./data/ad --age1_use_consist False --age1_use_adversarial False;
python scripts/train4.py --exp_name zero --mode student --teacher_type AgeAwareMLP1 --student_type AgeAwareMLP1 --age_threshold 65 --data_dir ./data/ad --age1_use_consist False --age1_use_adversarial False;
python scripts/train4.py --exp_name zero --mode student --teacher_type AgeAwareMLP1 --student_type MLP --age_threshold 65 --data_dir ./data/ad --age1_use_consist False --age1_use_adversarial False;
python scripts/train4.py --exp_name zero --mode teacher --teacher_type AgeAwareMLP2 --age_threshold 65 --data_dir ./data/ad --age2_use_consist False --age2_use_ageloss False --age2_use_disentangle False;
python scripts/train4.py --exp_name zero --mode student --teacher_type AgeAwareMLP2 --student_type AgeAwareMLP2 --age_threshold 65 --data_dir ./data/ad --age2_use_consist False --age2_use_ageloss False --age2_use_disentangle False;
python scripts/train4.py --exp_name zero --mode student --teacher_type AgeAwareMLP2 --student_type MLP --age_threshold 65 --data_dir ./data/ad --age2_use_consist False --age2_use_ageloss False --age2_use_disentangle False


python scripts/train4.py --exp_name ab_age1_adv --mode teacher --teacher_type AgeAwareMLP1 --age_threshold 65 --data_dir ./data/ad --age1_use_adversarial False;
python scripts/train4.py --exp_name ab_age1_adv --mode student --teacher_type AgeAwareMLP1 --student_type AgeAwareMLP1 --age_threshold 65 --data_dir ./data/ad --age1_use_adversarial False;
python scripts/train4.py --exp_name ab_age1_adv --mode student --teacher_type AgeAwareMLP1 --student_type MLP --age_threshold 65 --data_dir ./data/ad --age1_use_adversarial False


python scripts/train4.py --exp_name ab_age1_consist --mode teacher --teacher_type AgeAwareMLP1 --age_threshold 65 --data_dir ./data/ad --age1_use_consist False;
python scripts/train4.py --exp_name ab_age1_consist --mode student --teacher_type AgeAwareMLP1 --student_type AgeAwareMLP1 --age_threshold 65 --data_dir ./data/ad --age1_use_consist False;
python scripts/train4.py --exp_name ab_age1_consist --mode student --teacher_type AgeAwareMLP1 --student_type MLP --age_threshold 65 --data_dir ./data/ad --age1_use_consist False


python scripts/train4.py --exp_name ab_age2_ageloss --mode teacher --teacher_type AgeAwareMLP2 --age_threshold 65 --data_dir ./data/ad --age2_use_ageloss False;
python scripts/train4.py --exp_name ab_age2_ageloss --mode student --teacher_type AgeAwareMLP2 --student_type AgeAwareMLP2 --age_threshold 65 --data_dir ./data/ad --age2_use_ageloss False;
python scripts/train4.py --exp_name ab_age2_ageloss --mode student --teacher_type AgeAwareMLP2 --student_type MLP --age_threshold 65 --data_dir ./data/ad --age2_use_ageloss False


python scripts/train4.py --exp_name ab_age2_disentangle --mode teacher --teacher_type AgeAwareMLP2 --age_threshold 65 --data_dir ./data/ad --age2_use_disentangle False;
python scripts/train4.py --exp_name ab_age2_disentangle --mode student --teacher_type AgeAwareMLP2 --student_type AgeAwareMLP2 --age_threshold 65 --data_dir ./data/ad --age2_use_disentangle False;
python scripts/train4.py --exp_name ab_age2_disentangle --mode student --teacher_type AgeAwareMLP2 --student_type MLP --age_threshold 65 --data_dir ./data/ad --age2_use_disentangle False


python scripts/train4.py --exp_name ab_age2_consist --mode teacher --teacher_type AgeAwareMLP2 --age_threshold 65 --data_dir ./data/ad --age2_use_consist False;
python scripts/train4.py --exp_name ab_age2_consist --mode student --teacher_type AgeAwareMLP2 --student_type AgeAwareMLP2 --age_threshold 65 --data_dir ./data/ad --age2_use_consist False;
python scripts/train4.py --exp_name ab_age2_consist --mode student --teacher_type AgeAwareMLP2 --student_type MLP --age_threshold 65 --data_dir ./data/ad --age2_use_consist False


python scripts/train4.py --exp_name ab_age2_ageloss_disentangle --mode teacher --teacher_type AgeAwareMLP2 --age_threshold 65 --data_dir ./data/ad --age2_use_ageloss False --age2_use_disentangle False;
python scripts/train4.py --exp_name ab_age2_ageloss_disentangle --mode student --teacher_type AgeAwareMLP2 --student_type AgeAwareMLP2 --age_threshold 65 --data_dir ./data/ad --age2_use_ageloss False --age2_use_disentangle False;
python scripts/train4.py --exp_name ab_age2_ageloss_disentangle --mode student --teacher_type AgeAwareMLP2 --student_type MLP --age_threshold 65 --data_dir ./data/ad --age2_use_ageloss False --age2_use_disentangle False

