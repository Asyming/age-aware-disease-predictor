
# test_new_data_s1
CUDA_VISIBLE_DEVICES=0 python scripts/train4.py --exp_name test_new_data_s1 --mode student --teacher_type UGP_v3 --student_type ctrMLP --teacher_model_exp_name test_new_data_2 --teacher_model_lr 1e-5 --lr 1e-4 --use_label_correction --data_dir ./data/ad
CUDA_VISIBLE_DEVICES=1 python scripts/train4.py --exp_name test_new_data_s1 --mode student --teacher_type UGP_v3 --student_type ctrMLP --teacher_model_exp_name test_new_data_2 --teacher_model_lr 1e-5 --lr 1e-4 --use_label_correction --data_dir ./data/af
CUDA_VISIBLE_DEVICES=2 python scripts/train4.py --exp_name test_new_data_s1 --mode student --teacher_type UGP_v3 --student_type ctrMLP --teacher_model_exp_name test_new_data_2 --teacher_model_lr 1e-5 --lr 1e-4 --use_label_correction --data_dir ./data/ms
CUDA_VISIBLE_DEVICES=3 python scripts/train4.py --exp_name test_new_data_s1 --mode student --teacher_type UGP_v3 --student_type ctrMLP --teacher_model_exp_name test_new_data_2 --teacher_model_lr 1e-5 --lr 1e-4 --use_label_correction --data_dir ./data/uc

# test_new_data_s2 (alpha=0.0, T=2)
CUDA_VISIBLE_DEVICES=0 python scripts/train4.py --exp_name test_new_data_s2 --mode student --teacher_type UGP_v3 --student_type ctrMLP --teacher_model_exp_name test_new_data_2 --teacher_model_lr 1e-5 --lr 1e-4 --use_label_correction --alpha 0.0 --data_dir ./data/ad
CUDA_VISIBLE_DEVICES=1 python scripts/train4.py --exp_name test_new_data_s2 --mode student --teacher_type UGP_v3 --student_type ctrMLP --teacher_model_exp_name test_new_data_2 --teacher_model_lr 1e-5 --lr 1e-4 --use_label_correction --alpha 0.0 --data_dir ./data/af
CUDA_VISIBLE_DEVICES=2 python scripts/train4.py --exp_name test_new_data_s2 --mode student --teacher_type UGP_v3 --student_type ctrMLP --teacher_model_exp_name test_new_data_2 --teacher_model_lr 1e-5 --lr 1e-4 --use_label_correction --alpha 0.0 --data_dir ./data/ms
CUDA_VISIBLE_DEVICES=3 python scripts/train4.py --exp_name test_new_data_s2 --mode student --teacher_type UGP_v3 --student_type ctrMLP --teacher_model_exp_name test_new_data_2 --teacher_model_lr 1e-5 --lr 1e-4 --use_label_correction --alpha 0.0 --data_dir ./data/uc

# test_new_data_s3 (alpha=0.05, T=2)
CUDA_VISIBLE_DEVICES=0 python scripts/train4.py --exp_name test_new_data_s3 --mode student --teacher_type UGP_v3 --student_type ctrMLP --teacher_model_exp_name test_new_data_2 --teacher_model_lr 1e-5 --lr 1e-4 --use_label_correction --alpha 0.05 --data_dir ./data/ad
CUDA_VISIBLE_DEVICES=1 python scripts/train4.py --exp_name test_new_data_s3 --mode student --teacher_type UGP_v3 --student_type ctrMLP --teacher_model_exp_name test_new_data_2 --teacher_model_lr 1e-5 --lr 1e-4 --use_label_correction --alpha 0.05 --data_dir ./data/af
CUDA_VISIBLE_DEVICES=2 python scripts/train4.py --exp_name test_new_data_s3 --mode student --teacher_type UGP_v3 --student_type ctrMLP --teacher_model_exp_name test_new_data_2 --teacher_model_lr 1e-5 --lr 1e-4 --use_label_correction --alpha 0.05 --data_dir ./data/ms
CUDA_VISIBLE_DEVICES=3 python scripts/train4.py --exp_name test_new_data_s3 --mode student --teacher_type UGP_v3 --student_type ctrMLP --teacher_model_exp_name test_new_data_2 --teacher_model_lr 1e-5 --lr 1e-4 --use_label_correction --alpha 0.05 --data_dir ./data/uc

# test_new_data_s4 (alpha=0.1, T=1)
CUDA_VISIBLE_DEVICES=0 python scripts/train4.py --exp_name test_new_data_s4 --mode student --teacher_type UGP_v3 --student_type ctrMLP --teacher_model_exp_name test_new_data_2 --teacher_model_lr 1e-5 --lr 1e-4 --use_label_correction --temperature 1 --data_dir ./data/ad
CUDA_VISIBLE_DEVICES=1 python scripts/train4.py --exp_name test_new_data_s4 --mode student --teacher_type UGP_v3 --student_type ctrMLP --teacher_model_exp_name test_new_data_2 --teacher_model_lr 1e-5 --lr 1e-4 --use_label_correction --temperature 1 --data_dir ./data/af
CUDA_VISIBLE_DEVICES=2 python scripts/train4.py --exp_name test_new_data_s4 --mode student --teacher_type UGP_v3 --student_type ctrMLP --teacher_model_exp_name test_new_data_2 --teacher_model_lr 1e-5 --lr 1e-4 --use_label_correction --temperature 1 --data_dir ./data/ms
CUDA_VISIBLE_DEVICES=3 python scripts/train4.py --exp_name test_new_data_s4 --mode student --teacher_type UGP_v3 --student_type ctrMLP --teacher_model_exp_name test_new_data_2 --teacher_model_lr 1e-5 --lr 1e-4 --use_label_correction --temperature 1 --data_dir ./data/uc

# test_new_data_s5 (alpha=0.1, T=0.5)
CUDA_VISIBLE_DEVICES=0 python scripts/train4.py --exp_name test_new_data_s5 --mode student --teacher_type UGP_v3 --student_type ctrMLP --teacher_model_exp_name test_new_data_2 --teacher_model_lr 1e-5 --lr 1e-4 --use_label_correction --temperature 1 --data_dir ./data/ad
CUDA_VISIBLE_DEVICES=1 python scripts/train4.py --exp_name test_new_data_s5 --mode student --teacher_type UGP_v3 --student_type ctrMLP --teacher_model_exp_name test_new_data_2 --teacher_model_lr 1e-5 --lr 1e-4 --use_label_correction --temperature 1 --data_dir ./data/af
CUDA_VISIBLE_DEVICES=2 python scripts/train4.py --exp_name test_new_data_s5 --mode student --teacher_type UGP_v3 --student_type ctrMLP --teacher_model_exp_name test_new_data_2 --teacher_model_lr 1e-5 --lr 1e-4 --use_label_correction --temperature 1 --data_dir ./data/ms
CUDA_VISIBLE_DEVICES=3 python scripts/train4.py --exp_name test_new_data_s5 --mode student --teacher_type UGP_v3 --student_type ctrMLP --teacher_model_exp_name test_new_data_2 --teacher_model_lr 1e-5 --lr 1e-4 --use_label_correction --temperature 1 --data_dir ./data/uc

@REM # test_new_data_s6 (alpha=0.05, T=1.0)
@REM CUDA_VISIBLE_DEVICES=0 python scripts/train4.py --exp_name test_new_data_s6 --mode student --teacher_type UGP_v3 --student_type ctrMLP --teacher_model_exp_name test_new_data_2 --teacher_model_lr 1e-5 --lr 1e-4 --use_label_correction --temperature 1 --alpha 0.05 --data_dir ./data/ad
@REM CUDA_VISIBLE_DEVICES=1 python scripts/train4.py --exp_name test_new_data_s6 --mode student --teacher_type UGP_v3 --student_type ctrMLP --teacher_model_exp_name test_new_data_2 --teacher_model_lr 1e-5 --lr 1e-4 --use_label_correction --temperature 1 --alpha 0.05 --data_dir ./data/af
@REM CUDA_VISIBLE_DEVICES=2 python scripts/train4.py --exp_name test_new_data_s6 --mode student --teacher_type UGP_v3 --student_type ctrMLP --teacher_model_exp_name test_new_data_2 --teacher_model_lr 1e-5 --lr 1e-4 --use_label_correction --temperature 1 --alpha 0.05 --data_dir ./data/ms
@REM CUDA_VISIBLE_DEVICES=3 python scripts/train4.py --exp_name test_new_data_s6 --mode student --teacher_type UGP_v3 --student_type ctrMLP --teacher_model_exp_name test_new_data_2 --teacher_model_lr 1e-5 --lr 1e-4 --use_label_correction --temperature 1 --alpha 0.05 --data_dir ./data/uc

# test_new_data_s7 (alpha=0.05, T=1.0)
CUDA_VISIBLE_DEVICES=0 python scripts/train4.py --exp_name test_new_data_s7 --mode student --teacher_type UGP_v3 --student_type ctrMLP --teacher_model_exp_name test_new_data_2 --teacher_model_lr 1e-5 --lr 1e-4 --use_label_correction --temperature 1 --alpha 0.05 --data_dir ./data/ad
CUDA_VISIBLE_DEVICES=1 python scripts/train4.py --exp_name test_new_data_s7 --mode student --teacher_type UGP_v3 --student_type ctrMLP --teacher_model_exp_name test_new_data_2 --teacher_model_lr 1e-5 --lr 1e-4 --use_label_correction --temperature 1 --alpha 0.05 --data_dir ./data/af
CUDA_VISIBLE_DEVICES=2 python scripts/train4.py --exp_name test_new_data_s7 --mode student --teacher_type UGP_v3 --student_type ctrMLP --teacher_model_exp_name test_new_data_2 --teacher_model_lr 1e-5 --lr 1e-4 --use_label_correction --temperature 1 --alpha 0.05 --data_dir ./data/ms
CUDA_VISIBLE_DEVICES=3 python scripts/train4.py --exp_name test_new_data_s7 --mode student --teacher_type UGP_v3 --student_type ctrMLP --teacher_model_exp_name test_new_data_2 --teacher_model_lr 1e-5 --lr 1e-4 --use_label_correction --temperature 1 --alpha 0.05 --data_dir ./data/uc

# test_new_data_s8 (alpha=0.0, T=1.0, lamb = 10.0)
CUDA_VISIBLE_DEVICES=0 python scripts/train4.py --exp_name test_new_data_s8 --mode student --teacher_type UGP_v3 --student_type ctrMLP --teacher_model_exp_name test_new_data_2 --teacher_model_lr 1e-5 --lr 1e-4 --use_label_correction --temperature 1 --alpha 0.0 --lamb 10.0 --data_dir ./data/ad
CUDA_VISIBLE_DEVICES=1 python scripts/train4.py --exp_name test_new_data_s8 --mode student --teacher_type UGP_v3 --student_type ctrMLP --teacher_model_exp_name test_new_data_2 --teacher_model_lr 1e-5 --lr 1e-4 --use_label_correction --temperature 1 --alpha 0.0 --lamb 10.0 --data_dir ./data/af
CUDA_VISIBLE_DEVICES=2 python scripts/train4.py --exp_name test_new_data_s8 --mode student --teacher_type UGP_v3 --student_type ctrMLP --teacher_model_exp_name test_new_data_2 --teacher_model_lr 1e-5 --lr 1e-4 --use_label_correction --temperature 1 --alpha 0.0 --lamb 10.0 --data_dir ./data/ms
CUDA_VISIBLE_DEVICES=3 python scripts/train4.py --exp_name test_new_data_s8 --mode student --teacher_type UGP_v3 --student_type ctrMLP --teacher_model_exp_name test_new_data_2 --teacher_model_lr 1e-5 --lr 1e-4 --use_label_correction --temperature 1 --alpha 0.0 --lamb 10.0 --data_dir ./data/uc

# test_new_data_s9 (alpha=0.05, T=1.0, pos_tau=0.999, neg_tau=0.800)
CUDA_VISIBLE_DEVICES=0 python scripts/train4.py --exp_name test_new_data_s9 --mode student --teacher_type UGP_v3 --student_type ctrMLP --teacher_model_exp_name test_new_data_2 --teacher_model_lr 1e-5 --lr 1e-4 --use_label_correction --temperature 1 --alpha 0.05 --pos_tau 0.999 --neg_tau 0.800 --data_dir ./data/ad
CUDA_VISIBLE_DEVICES=1 python scripts/train4.py --exp_name test_new_data_s9 --mode student --teacher_type UGP_v3 --student_type ctrMLP --teacher_model_exp_name test_new_data_2 --teacher_model_lr 1e-5 --lr 1e-4 --use_label_correction --temperature 1 --alpha 0.05 --pos_tau 0.999 --neg_tau 0.800 --data_dir ./data/af
CUDA_VISIBLE_DEVICES=2 python scripts/train4.py --exp_name test_new_data_s9 --mode student --teacher_type UGP_v3 --student_type ctrMLP --teacher_model_exp_name test_new_data_2 --teacher_model_lr 1e-5 --lr 1e-4 --use_label_correction --temperature 1 --alpha 0.05 --pos_tau 0.999 --neg_tau 0.800 --data_dir ./data/ms
CUDA_VISIBLE_DEVICES=3 python scripts/train4.py --exp_name test_new_data_s9 --mode student --teacher_type UGP_v3 --student_type ctrMLP --teacher_model_exp_name test_new_data_2 --teacher_model_lr 1e-5 --lr 1e-4 --use_label_correction --temperature 1 --alpha 0.05 --pos_tau 0.999 --neg_tau 0.800 --data_dir ./data/uc

# test_new_data_s10 (alpha=0.05, T=2.0, pos_tau=0.999, neg_tau=0.800)
CUDA_VISIBLE_DEVICES=0 python scripts/train4.py --exp_name test_new_data_s10 --mode student --teacher_type UGP_v3 --student_type ctrMLP --teacher_model_exp_name test_new_data_2 --teacher_model_lr 1e-5 --lr 1e-4 --use_label_correction --temperature 2 --alpha 0.05 --pos_tau 0.999 --neg_tau 0.800 --data_dir ./data/ad
CUDA_VISIBLE_DEVICES=1 python scripts/train4.py --exp_name test_new_data_s10 --mode student --teacher_type UGP_v3 --student_type ctrMLP --teacher_model_exp_name test_new_data_2 --teacher_model_lr 1e-5 --lr 1e-4 --use_label_correction --temperature 2 --alpha 0.05 --pos_tau 0.999 --neg_tau 0.800 --data_dir ./data/af
CUDA_VISIBLE_DEVICES=2 python scripts/train4.py --exp_name test_new_data_s10 --mode student --teacher_type UGP_v3 --student_type ctrMLP --teacher_model_exp_name test_new_data_2 --teacher_model_lr 1e-5 --lr 1e-4 --use_label_correction --temperature 2 --alpha 0.05 --pos_tau 0.999 --neg_tau 0.800 --data_dir ./data/ms
CUDA_VISIBLE_DEVICES=3 python scripts/train4.py --exp_name test_new_data_s10 --mode student --teacher_type UGP_v3 --student_type ctrMLP --teacher_model_exp_name test_new_data_2 --teacher_model_lr 1e-5 --lr 1e-4 --use_label_correction --temperature 2 --alpha 0.05 --pos_tau 0.999 --neg_tau 0.800 --data_dir ./data/uc

# test_ugp_random
CUDA_VISIBLE_DEVICES=0 python scripts/train4.py --exp_name test_new_data_lc_5 --mode teacher --teacher_type UGP_v3 --data_dir ./data/ad --lr 1e-5;
CUDA_VISIBLE_DEVICES=0 python scripts/train4.py --exp_name test_new_data_lc_5 --mode teacher --teacher_type UGP_v3 --data_dir ./data/ad --lr 1e-5;
CUDA_VISIBLE_DEVICES=0 python scripts/train4.py --exp_name test_new_data_lc_5 --mode teacher --teacher_type UGP_v3 --data_dir ./data/ad --lr 1e-5;
CUDA_VISIBLE_DEVICES=0 python scripts/train4.py --exp_name test_new_data_lc_5 --mode teacher --teacher_type UGP_v3 --data_dir ./data/ad --lr 1e-5;

# test_lc
CUDA_VISIBLE_DEVICES=0 python scripts/train4.py --exp_name test_lc --mode teacher --teacher_type MLP --data_dir ./data/ad --lr 1e-4 --use_label_correction;
CUDA_VISIBLE_DEVICES=0 python scripts/train4.py --exp_name test_lc --mode teacher --teacher_type ctrMLP --data_dir ./data/ad --lr 1e-4 --use_label_correction;
CUDA_VISIBLE_DEVICES=0 python scripts/train4.py --exp_name test_lc --mode teacher --teacher_type UGP_v3 --data_dir ./data/ad --lr 1e-5 --use_label_correction;
CUDA_VISIBLE_DEVICES=0 python scripts/train4.py --exp_name test_lc --mode teacher --teacher_type ctrUGP_v1 --data_dir ./data/ad --lr 1e-5 --use_label_correction;

CUDA_VISIBLE_DEVICES=0 python scripts/train4.py --exp_name test_lc --mode teacher --teacher_type AgeUGP_v1 --data_dir ./data/ad --lr 1e-5 --use_label_correction;
CUDA_VISIBLE_DEVICES=0 python scripts/train4.py --exp_name test_lc --mode teacher --teacher_type AgeUGP_v2 --data_dir ./data/ad --lr 1e-5 --use_label_correction;
CUDA_VISIBLE_DEVICES=0 python scripts/train4.py --exp_name test_lc --mode teacher --teacher_type AgeAwareMLP1 --data_dir ./data/ad --lr 1e-4 --use_label_correction;
CUDA_VISIBLE_DEVICES=0 python scripts/train4.py --exp_name test_lc --mode teacher --teacher_type AgeAwareMLP2 --data_dir ./data/ad --lr 1e-4 --use_label_correction;

# test_ugp
python scripts/train4.py --exp_name test_ugp --mode teacher --teacher_type UGP_v3 --data_dir ./data/ad --lr 1e-5;
python scripts/train4.py --exp_name test_ugp --mode student --teacher_type UGP_v3 --student_type UGP_v3 --data_dir ./data/ad --lr 1e-5;
python scripts/train4.py --exp_name test_ugp --mode student --teacher_type UGP_v3 --student_type MLP --data_dir ./data/ad --lr 1e-5;

# test_new_data
CUDA_VISIBLE_DEVICES=0 python scripts/train4.py --exp_name test_new_data_2 --mode teacher --teacher_type MLP --data_dir ./data/ad --lr 1e-4;
CUDA_VISIBLE_DEVICES=0 python scripts/train4.py --exp_name test_new_data_2 --mode teacher --teacher_type ctrMLP --data_dir ./data/ad --lr 1e-4;
CUDA_VISIBLE_DEVICES=0 python scripts/train4.py --exp_name test_new_data_2 --mode teacher --teacher_type UGP_v3 --data_dir ./data/ad --lr 1e-5;
CUDA_VISIBLE_DEVICES=0 python scripts/train4.py --exp_name test_new_data_2 --mode teacher --teacher_type ctrUGP_v1 --data_dir ./data/ad --lr 1e-5;

CUDA_VISIBLE_DEVICES=1 python scripts/train4.py --exp_name test_new_data_2 --mode teacher --teacher_type MLP --data_dir ./data/ms --lr 1e-4;
CUDA_VISIBLE_DEVICES=1 python scripts/train4.py --exp_name test_new_data_2 --mode teacher --teacher_type ctrMLP --data_dir ./data/ms --lr 1e-4;
CUDA_VISIBLE_DEVICES=1 python scripts/train4.py --exp_name test_new_data_2 --mode teacher --teacher_type UGP_v3 --data_dir ./data/ms --lr 1e-5;
CUDA_VISIBLE_DEVICES=1 python scripts/train4.py --exp_name test_new_data_2 --mode teacher --teacher_type ctrUGP_v1 --data_dir ./data/ms --lr 1e-5;

CUDA_VISIBLE_DEVICES=2 python scripts/train4.py --exp_name test_new_data_2 --mode teacher --teacher_type MLP --data_dir ./data/uc --lr 1e-4;
CUDA_VISIBLE_DEVICES=2 python scripts/train4.py --exp_name test_new_data_2 --mode teacher --teacher_type ctrMLP --data_dir ./data/uc --lr 1e-4;
CUDA_VISIBLE_DEVICES=2 python scripts/train4.py --exp_name test_new_data_2 --mode teacher --teacher_type UGP_v3 --data_dir ./data/uc --lr 1e-5;
CUDA_VISIBLE_DEVICES=2 python scripts/train4.py --exp_name test_new_data_2 --mode teacher --teacher_type ctrUGP_v1 --data_dir ./data/uc --lr 1e-5;

CUDA_VISIBLE_DEVICES=3 python scripts/train4.py --exp_name test_new_data_2 --mode teacher --teacher_type MLP --data_dir ./data/af --lr 1e-4;
CUDA_VISIBLE_DEVICES=3 python scripts/train4.py --exp_name test_new_data_2 --mode teacher --teacher_type ctrMLP --data_dir ./data/af --lr 1e-4;
CUDA_VISIBLE_DEVICES=3 python scripts/train4.py --exp_name test_new_data_2 --mode teacher --teacher_type UGP_v3 --data_dir ./data/af --lr 1e-5;
CUDA_VISIBLE_DEVICES=3 python scripts/train4.py --exp_name test_new_data_2 --mode teacher --teacher_type ctrUGP_v1 --data_dir ./data/af --lr 1e-5;

#test_new_data_2 (no_lc)
CUDA_VISIBLE_DEVICES=0 python scripts/train4.py --exp_name test_new_data_2 --mode teacher --teacher_type MLP --data_dir ./data/ad --lr 1e-4;
CUDA_VISIBLE_DEVICES=0 python scripts/train4.py --exp_name test_new_data_2 --mode teacher --teacher_type ctrMLP --data_dir ./data/ad --lr 1e-4;
CUDA_VISIBLE_DEVICES=0 python scripts/train4.py --exp_name test_new_data_2 --mode teacher --teacher_type UGP_v3 --data_dir ./data/ad --lr 1e-5;
CUDA_VISIBLE_DEVICES=0 python scripts/train4.py --exp_name test_new_data_2 --mode teacher --teacher_type ctrUGP_v1 --data_dir ./data/ad --lr 1e-5;

CUDA_VISIBLE_DEVICES=1 python scripts/train4.py --exp_name test_new_data_2 --mode teacher --teacher_type MLP --data_dir ./data/ms --lr 1e-4;
CUDA_VISIBLE_DEVICES=1 python scripts/train4.py --exp_name test_new_data_2 --mode teacher --teacher_type ctrMLP --data_dir ./data/ms --lr 1e-4;
CUDA_VISIBLE_DEVICES=1 python scripts/train4.py --exp_name test_new_data_2 --mode teacher --teacher_type UGP_v3 --data_dir ./data/ms --lr 1e-5;
CUDA_VISIBLE_DEVICES=1 python scripts/train4.py --exp_name test_new_data_2 --mode teacher --teacher_type ctrUGP_v1 --data_dir ./data/ms --lr 1e-5;

CUDA_VISIBLE_DEVICES=2 python scripts/train4.py --exp_name test_new_data_2 --mode teacher --teacher_type MLP --data_dir ./data/uc --lr 1e-4;
CUDA_VISIBLE_DEVICES=2 python scripts/train4.py --exp_name test_new_data_2 --mode teacher --teacher_type ctrMLP --data_dir ./data/uc --lr 1e-4;
CUDA_VISIBLE_DEVICES=2 python scripts/train4.py --exp_name test_new_data_2 --mode teacher --teacher_type UGP_v3 --data_dir ./data/uc --lr 1e-5;
CUDA_VISIBLE_DEVICES=2 python scripts/train4.py --exp_name test_new_data_2 --mode teacher --teacher_type ctrUGP_v1 --data_dir ./data/uc --lr 1e-5;

CUDA_VISIBLE_DEVICES=3 python scripts/train4.py --exp_name test_new_data_2 --mode teacher --teacher_type MLP --data_dir ./data/af --lr 1e-4;
CUDA_VISIBLE_DEVICES=3 python scripts/train4.py --exp_name test_new_data_2 --mode teacher --teacher_type ctrMLP --data_dir ./data/af --lr 1e-4;
CUDA_VISIBLE_DEVICES=3 python scripts/train4.py --exp_name test_new_data_2 --mode teacher --teacher_type UGP_v3 --data_dir ./data/af --lr 1e-5;
CUDA_VISIBLE_DEVICES=3 python scripts/train4.py --exp_name test_new_data_2 --mode teacher --teacher_type ctrUGP_v1 --data_dir ./data/af --lr 1e-5;

#test_new_data_gce_1 (q=0.6)
CUDA_VISIBLE_DEVICES=0 python scripts/train4.py --exp_name test_new_data_gce_1 --mode teacher --teacher_type MLP --data_dir ./data/ad --lr 1e-4 --use_bgce;
CUDA_VISIBLE_DEVICES=0 python scripts/train4.py --exp_name test_new_data_gce_1 --mode teacher --teacher_type UGP_v3 --data_dir ./data/ad --lr 1e-5 --use_bgce;

CUDA_VISIBLE_DEVICES=1 python scripts/train4.py --exp_name test_new_data_gce_1 --mode teacher --teacher_type MLP --data_dir ./data/ms --lr 1e-4 --use_bgce;
CUDA_VISIBLE_DEVICES=1 python scripts/train4.py --exp_name test_new_data_gce_1 --mode teacher --teacher_type UGP_v3 --data_dir ./data/ms --lr 1e-5 --use_bgce;

CUDA_VISIBLE_DEVICES=2 python scripts/train4.py --exp_name test_new_data_gce_1 --mode teacher --teacher_type MLP --data_dir ./data/uc --lr 1e-4 --use_bgce;
CUDA_VISIBLE_DEVICES=2 python scripts/train4.py --exp_name test_new_data_gce_1 --mode teacher --teacher_type UGP_v3 --data_dir ./data/uc --lr 1e-5 --use_bgce;

CUDA_VISIBLE_DEVICES=3 python scripts/train4.py --exp_name test_new_data_gce_1 --mode teacher --teacher_type MLP --data_dir ./data/af --lr 1e-4 --use_bgce;
CUDA_VISIBLE_DEVICES=3 python scripts/train4.py --exp_name test_new_data_gce_1 --mode teacher --teacher_type UGP_v3 --data_dir ./data/af --lr 1e-5 --use_bgce;

#test_new_data_gce_2 (lc, q=0.6)
CUDA_VISIBLE_DEVICES=0 python scripts/train4.py --exp_name test_new_data_gce_2 --mode teacher --teacher_type MLP --data_dir ./data/ad --lr 1e-4 --use_bgce --use_label_correction;
CUDA_VISIBLE_DEVICES=0 python scripts/train4.py --exp_name test_new_data_gce_2 --mode teacher --teacher_type UGP_v3 --data_dir ./data/ad --lr 1e-5 --use_bgce --use_label_correction;

CUDA_VISIBLE_DEVICES=1 python scripts/train4.py --exp_name test_new_data_gce_2 --mode teacher --teacher_type MLP --data_dir ./data/ms --lr 1e-4 --use_bgce --use_label_correction;
CUDA_VISIBLE_DEVICES=1 python scripts/train4.py --exp_name test_new_data_gce_2 --mode teacher --teacher_type UGP_v3 --data_dir ./data/ms --lr 1e-5 --use_bgce --use_label_correction;

CUDA_VISIBLE_DEVICES=2 python scripts/train4.py --exp_name test_new_data_gce_2 --mode teacher --teacher_type MLP --data_dir ./data/uc --lr 1e-4 --use_bgce --use_label_correction;
CUDA_VISIBLE_DEVICES=2 python scripts/train4.py --exp_name test_new_data_gce_2 --mode teacher --teacher_type UGP_v3 --data_dir ./data/uc --lr 1e-5 --use_bgce --use_label_correction;

CUDA_VISIBLE_DEVICES=3 python scripts/train4.py --exp_name test_new_data_gce_2 --mode teacher --teacher_type MLP --data_dir ./data/af --lr 1e-4 --use_bgce --use_label_correction;
CUDA_VISIBLE_DEVICES=3 python scripts/train4.py --exp_name test_new_data_gce_2 --mode teacher --teacher_type UGP_v3 --data_dir ./data/af --lr 1e-5 --use_bgce --use_label_correction;

#test_new_data_gce_3 (mixup, q=0.6, alpha=0.2)
CUDA_VISIBLE_DEVICES=0 python scripts/train4.py --exp_name test_new_data_gce_3 --mode teacher --teacher_type MLP --data_dir ./data/ad --lr 1e-4 --use_bgce --use_mixup;
CUDA_VISIBLE_DEVICES=0 python scripts/train4.py --exp_name test_new_data_gce_3 --mode teacher --teacher_type UGP_v3 --data_dir ./data/ad --lr 1e-5 --use_bgce --use_mixup;

CUDA_VISIBLE_DEVICES=1 python scripts/train4.py --exp_name test_new_data_gce_3 --mode teacher --teacher_type MLP --data_dir ./data/ms --lr 1e-4 --use_bgce --use_mixup;
CUDA_VISIBLE_DEVICES=1 python scripts/train4.py --exp_name test_new_data_gce_3 --mode teacher --teacher_type UGP_v3 --data_dir ./data/ms --lr 1e-5 --use_bgce --use_mixup;

CUDA_VISIBLE_DEVICES=2 python scripts/train4.py --exp_name test_new_data_gce_3 --mode teacher --teacher_type MLP --data_dir ./data/uc --lr 1e-4 --use_bgce --use_mixup;
CUDA_VISIBLE_DEVICES=2 python scripts/train4.py --exp_name test_new_data_gce_3 --mode teacher --teacher_type UGP_v3 --data_dir ./data/uc --lr 1e-5 --use_bgce --use_mixup;

CUDA_VISIBLE_DEVICES=3 python scripts/train4.py --exp_name test_new_data_gce_3 --mode teacher --teacher_type MLP --data_dir ./data/af --lr 1e-4 --use_bgce --use_mixup;
CUDA_VISIBLE_DEVICES=3 python scripts/train4.py --exp_name test_new_data_gce_3 --mode teacher --teacher_type UGP_v3 --data_dir ./data/af --lr 1e-5 --use_bgce --use_mixup;

#test_new_data_gce_4 (mixup, lc, q=0.6, alpha=0.2)
CUDA_VISIBLE_DEVICES=0 python scripts/train4.py --exp_name test_new_data_gce_4 --mode teacher --teacher_type MLP --data_dir ./data/ad --lr 1e-4 --use_bgce --use_mixup --use_label_correction;
CUDA_VISIBLE_DEVICES=0 python scripts/train4.py --exp_name test_new_data_gce_4 --mode teacher --teacher_type UGP_v3 --data_dir ./data/ad --lr 1e-5 --use_bgce --use_mixup --use_label_correction;

CUDA_VISIBLE_DEVICES=1 python scripts/train4.py --exp_name test_new_data_gce_4 --mode teacher --teacher_type MLP --data_dir ./data/ms --lr 1e-4 --use_bgce --use_mixup --use_label_correction;
CUDA_VISIBLE_DEVICES=1 python scripts/train4.py --exp_name test_new_data_gce_4 --mode teacher --teacher_type UGP_v3 --data_dir ./data/ms --lr 1e-5 --use_bgce --use_mixup --use_label_correction;

CUDA_VISIBLE_DEVICES=2 python scripts/train4.py --exp_name test_new_data_gce_4 --mode teacher --teacher_type MLP --data_dir ./data/uc --lr 1e-4 --use_bgce --use_mixup --use_label_correction;
CUDA_VISIBLE_DEVICES=2 python scripts/train4.py --exp_name test_new_data_gce_4 --mode teacher --teacher_type UGP_v3 --data_dir ./data/uc --lr 1e-5 --use_bgce --use_mixup --use_label_correction;

CUDA_VISIBLE_DEVICES=3 python scripts/train4.py --exp_name test_new_data_gce_4 --mode teacher --teacher_type MLP --data_dir ./data/af --lr 1e-4 --use_bgce --use_mixup --use_label_correction;
CUDA_VISIBLE_DEVICES=1 python scripts/train4.py --exp_name test_new_data_gce_4 --mode teacher --teacher_type UGP_v3 --data_dir ./data/af --lr 1e-5 --use_bgce --use_mixup --use_label_correction;

#test_new_data_3 (simple ctrUGP)
CUDA_VISIBLE_DEVICES=0 python scripts/train4.py --exp_name test_new_data_3 --mode teacher --teacher_type MLP --data_dir ./data/ad --lr 1e-5;
CUDA_VISIBLE_DEVICES=0 python scripts/train4.py --exp_name test_new_data_3 --mode teacher --teacher_type ctrMLP --data_dir ./data/ad --lr 1e-5;
CUDA_VISIBLE_DEVICES=0 python scripts/train4.py --exp_name test_new_data_3 --mode teacher --teacher_type UGP_v3 --data_dir ./data/ad --lr 1e-5;
CUDA_VISIBLE_DEVICES=0 python scripts/train4.py --exp_name test_new_data_3 --mode teacher --teacher_type ctrUGP_v1 --data_dir ./data/ad --lr 1e-5;

CUDA_VISIBLE_DEVICES=1 python scripts/train4.py --exp_name test_new_data_3 --mode teacher --teacher_type MLP --data_dir ./data/ms --lr 1e-5;
CUDA_VISIBLE_DEVICES=1 python scripts/train4.py --exp_name test_new_data_3 --mode teacher --teacher_type ctrMLP --data_dir ./data/ms --lr 1e-5;
CUDA_VISIBLE_DEVICES=1 python scripts/train4.py --exp_name test_new_data_3 --mode teacher --teacher_type UGP_v3 --data_dir ./data/ms --lr 1e-5;
CUDA_VISIBLE_DEVICES=1 python scripts/train4.py --exp_name test_new_data_3 --mode teacher --teacher_type ctrUGP_v1 --data_dir ./data/ms --lr 1e-5;

CUDA_VISIBLE_DEVICES=2 python scripts/train4.py --exp_name test_new_data_3 --mode teacher --teacher_type MLP --data_dir ./data/uc --lr 1e-5;
CUDA_VISIBLE_DEVICES=2 python scripts/train4.py --exp_name test_new_data_3 --mode teacher --teacher_type ctrMLP --data_dir ./data/uc --lr 1e-5;
CUDA_VISIBLE_DEVICES=2 python scripts/train4.py --exp_name test_new_data_3 --mode teacher --teacher_type UGP_v3 --data_dir ./data/uc --lr 1e-5;
CUDA_VISIBLE_DEVICES=2 python scripts/train4.py --exp_name test_new_data_3 --mode teacher --teacher_type ctrUGP_v1 --data_dir ./data/uc --lr 1e-5;

CUDA_VISIBLE_DEVICES=3 python scripts/train4.py --exp_name test_new_data_3 --mode teacher --teacher_type MLP --data_dir ./data/af --lr 1e-5;
CUDA_VISIBLE_DEVICES=3 python scripts/train4.py --exp_name test_new_data_3 --mode teacher --teacher_type ctrMLP --data_dir ./data/af --lr 1e-5;
CUDA_VISIBLE_DEVICES=3 python scripts/train4.py --exp_name test_new_data_3 --mode teacher --teacher_type UGP_v3 --data_dir ./data/af --lr 1e-5;
CUDA_VISIBLE_DEVICES=3 python scripts/train4.py --exp_name test_new_data_3 --mode teacher --teacher_type ctrUGP_v1 --data_dir ./data/af --lr 1e-5;

#test_new_data_mixup_1 (mixup, alpha = 0.2)
CUDA_VISIBLE_DEVICES=0 python scripts/train4.py --exp_name test_new_data_mixup_1 --mode teacher --teacher_type MLP --data_dir ./data/ad --lr 1e-4 --use_mixup;
CUDA_VISIBLE_DEVICES=0 python scripts/train4.py --exp_name test_new_data_mixup_1 --mode teacher --teacher_type UGP_v3 --data_dir ./data/ad --lr 1e-5 --use_mixup;

CUDA_VISIBLE_DEVICES=1 python scripts/train4.py --exp_name test_new_data_mixup_1 --mode teacher --teacher_type MLP --data_dir ./data/ms --lr 1e-4 --use_mixup;
CUDA_VISIBLE_DEVICES=1 python scripts/train4.py --exp_name test_new_data_mixup_1 --mode teacher --teacher_type UGP_v3 --data_dir ./data/ms --lr 1e-5 --use_mixup;

CUDA_VISIBLE_DEVICES=2 python scripts/train4.py --exp_name test_new_data_mixup_1 --mode teacher --teacher_type MLP --data_dir ./data/uc --lr 1e-4 --use_mixup;
CUDA_VISIBLE_DEVICES=2 python scripts/train4.py --exp_name test_new_data_mixup_1 --mode teacher --teacher_type UGP_v3 --data_dir ./data/uc --lr 1e-5 --use_mixup;

CUDA_VISIBLE_DEVICES=3 python scripts/train4.py --exp_name test_new_data_mixup_1 --mode teacher --teacher_type MLP --data_dir ./data/af --lr 1e-4 --use_mixup;
CUDA_VISIBLE_DEVICES=3 python scripts/train4.py --exp_name test_new_data_mixup_1 --mode teacher --teacher_type UGP_v3 --data_dir ./data/af --lr 1e-5 --use_mixup;

#test_new_data_mixup_2 (mixup, lc, alpha = 0.2, eta = 1.0)
CUDA_VISIBLE_DEVICES=0 python scripts/train4.py --exp_name test_new_data_mixup_2 --mode teacher --teacher_type MLP --data_dir ./data/ad --lr 1e-4 --use_mixup --use_label_correction;
CUDA_VISIBLE_DEVICES=0 python scripts/train4.py --exp_name test_new_data_mixup_2 --mode teacher --teacher_type UGP_v3 --data_dir ./data/ad --lr 1e-5 --use_mixup --use_label_correction;

CUDA_VISIBLE_DEVICES=1 python scripts/train4.py --exp_name test_new_data_mixup_2 --mode teacher --teacher_type MLP --data_dir ./data/ms --lr 1e-4 --use_mixup --use_label_correction;
CUDA_VISIBLE_DEVICES=1 python scripts/train4.py --exp_name test_new_data_mixup_2 --mode teacher --teacher_type UGP_v3 --data_dir ./data/ms --lr 1e-5 --use_mixup --use_label_correction;

CUDA_VISIBLE_DEVICES=2 python scripts/train4.py --exp_name test_new_data_mixup_2 --mode teacher --teacher_type MLP --data_dir ./data/uc --lr 1e-4 --use_mixup --use_label_correction;
CUDA_VISIBLE_DEVICES=2 python scripts/train4.py --exp_name test_new_data_mixup_2 --mode teacher --teacher_type UGP_v3 --data_dir ./data/uc --lr 1e-5 --use_mixup --use_label_correction;

CUDA_VISIBLE_DEVICES=3 python scripts/train4.py --exp_name test_new_data_mixup_2 --mode teacher --teacher_type MLP --data_dir ./data/af --lr 1e-4 --use_mixup --use_label_correction;
CUDA_VISIBLE_DEVICES=3 python scripts/train4.py --exp_name test_new_data_mixup_2 --mode teacher --teacher_type UGP_v3 --data_dir ./data/af --lr 1e-5 --use_mixup --use_label_correction;

#test_new_data_lc_5
CUDA_VISIBLE_DEVICES=0 python scripts/train4.py --exp_name test_new_data_lc_5 --mode teacher --teacher_type MLP --use_label_correction --untrusted_age_threshold 55 --pseudo_label_interval 20 --pseudo_label_start_step 300 --confidence_threshold 0.6 --data_dir ./data/ad --lr 1e-4;
CUDA_VISIBLE_DEVICES=0 python scripts/train4.py --exp_name test_new_data_lc_5 --mode teacher --teacher_type MLP --use_label_correction --untrusted_age_threshold 55 --pseudo_label_interval 20 --pseudo_label_start_step 300 --confidence_threshold 0.6 --data_dir ./data/af --lr 1e-4;
CUDA_VISIBLE_DEVICES=0 python scripts/train4.py --exp_name test_new_data_lc_5 --mode teacher --teacher_type MLP --use_label_correction --untrusted_age_threshold 55 --pseudo_label_interval 20 --pseudo_label_start_step 300 --confidence_threshold 0.6 --data_dir ./data/ms --lr 1e-4;
CUDA_VISIBLE_DEVICES=0 python scripts/train4.py --exp_name test_new_data_lc_5 --mode teacher --teacher_type MLP --use_label_correction --untrusted_age_threshold 55 --pseudo_label_interval 20 --pseudo_label_start_step 300 --confidence_threshold 0.6 --data_dir ./data/uc --lr 1e-4;

CUDA_VISIBLE_DEVICES=0 python scripts/train4.py --exp_name test_new_data_lc_5 --mode teacher --teacher_type UGP_v3 --use_label_correction --untrusted_age_threshold 55 --pseudo_label_interval 20 --pseudo_label_start_step 300 --confidence_threshold 0.6 --data_dir ./data/ad --lr 1e-5;
CUDA_VISIBLE_DEVICES=0 python scripts/train4.py --exp_name test_new_data_lc_5 --mode teacher --teacher_type UGP_v3 --use_label_correction --untrusted_age_threshold 55 --pseudo_label_interval 20 --pseudo_label_start_step 300 --confidence_threshold 0.6 --data_dir ./data/af --lr 1e-5;
CUDA_VISIBLE_DEVICES=0 python scripts/train4.py --exp_name test_new_data_lc_5 --mode teacher --teacher_type UGP_v3 --use_label_correction --untrusted_age_threshold 55 --pseudo_label_interval 20 --pseudo_label_start_step 300 --confidence_threshold 0.6 --data_dir ./data/ms --lr 1e-5;
CUDA_VISIBLE_DEVICES=0 python scripts/train4.py --exp_name test_new_data_lc_5 --mode teacher --teacher_type UGP_v3 --use_label_correction --untrusted_age_threshold 55 --pseudo_label_interval 20 --pseudo_label_start_step 300 --confidence_threshold 0.6 --data_dir ./data/uc --lr 1e-5;

#test_new_data_lc_11
CUDA_VISIBLE_DEVICES=1 python scripts/train4.py --exp_name test_new_data_lc_11 --mode teacher --teacher_type MLP --use_label_correction --untrusted_age_threshold 55 --pseudo_label_interval 10 --pseudo_label_start_step 400 --confidence_threshold 0.7 --data_dir ./data/ad --lr 1e-4;
CUDA_VISIBLE_DEVICES=1 python scripts/train4.py --exp_name test_new_data_lc_11 --mode teacher --teacher_type MLP --use_label_correction --untrusted_age_threshold 55 --pseudo_label_interval 10 --pseudo_label_start_step 400 --confidence_threshold 0.7 --data_dir ./data/af --lr 1e-4;
CUDA_VISIBLE_DEVICES=1 python scripts/train4.py --exp_name test_new_data_lc_11 --mode teacher --teacher_type MLP --use_label_correction --untrusted_age_threshold 55 --pseudo_label_interval 10 --pseudo_label_start_step 400 --confidence_threshold 0.7 --data_dir ./data/ms --lr 1e-4;
CUDA_VISIBLE_DEVICES=1 python scripts/train4.py --exp_name test_new_data_lc_11 --mode teacher --teacher_type MLP --use_label_correction --untrusted_age_threshold 55 --pseudo_label_interval 10 --pseudo_label_start_step 400 --confidence_threshold 0.7 --data_dir ./data/uc --lr 1e-4;

CUDA_VISIBLE_DEVICES=1 python scripts/train4.py --exp_name test_new_data_lc_11 --mode teacher --teacher_type UGP_v3 --use_label_correction --untrusted_age_threshold 55 --pseudo_label_interval 10 --pseudo_label_start_step 400 --confidence_threshold 0.7 --data_dir ./data/ad --lr 1e-5;
CUDA_VISIBLE_DEVICES=1 python scripts/train4.py --exp_name test_new_data_lc_11 --mode teacher --teacher_type UGP_v3 --use_label_correction --untrusted_age_threshold 55 --pseudo_label_interval 10 --pseudo_label_start_step 400 --confidence_threshold 0.7 --data_dir ./data/af --lr 1e-5;
CUDA_VISIBLE_DEVICES=1 python scripts/train4.py --exp_name test_new_data_lc_11 --mode teacher --teacher_type UGP_v3 --use_label_correction --untrusted_age_threshold 55 --pseudo_label_interval 10 --pseudo_label_start_step 400 --confidence_threshold 0.7 --data_dir ./data/ms --lr 1e-5;
CUDA_VISIBLE_DEVICES=1 python scripts/train4.py --exp_name test_new_data_lc_11 --mode teacher --teacher_type UGP_v3 --use_label_correction --untrusted_age_threshold 55 --pseudo_label_interval 10 --pseudo_label_start_step 400 --confidence_threshold 0.7 --data_dir ./data/uc --lr 1e-5;

#test_new_data_lc_12
CUDA_VISIBLE_DEVICES=2 python scripts/train4.py --exp_name test_new_data_lc_12 --mode teacher --teacher_type MLP --use_label_correction --untrusted_age_threshold 60 --pseudo_label_interval 10 --pseudo_label_start_step 500 --confidence_threshold 0.7 --data_dir ./data/ad --lr 1e-4;
CUDA_VISIBLE_DEVICES=2 python scripts/train4.py --exp_name test_new_data_lc_12 --mode teacher --teacher_type MLP --use_label_correction --untrusted_age_threshold 60 --pseudo_label_interval 10 --pseudo_label_start_step 500 --confidence_threshold 0.7 --data_dir ./data/af --lr 1e-4;
CUDA_VISIBLE_DEVICES=2 python scripts/train4.py --exp_name test_new_data_lc_12 --mode teacher --teacher_type MLP --use_label_correction --untrusted_age_threshold 60 --pseudo_label_interval 10 --pseudo_label_start_step 500 --confidence_threshold 0.7 --data_dir ./data/ms --lr 1e-4;
CUDA_VISIBLE_DEVICES=2 python scripts/train4.py --exp_name test_new_data_lc_12 --mode teacher --teacher_type MLP --use_label_correction --untrusted_age_threshold 60 --pseudo_label_interval 10 --pseudo_label_start_step 500 --confidence_threshold 0.7 --data_dir ./data/uc --lr 1e-4;

CUDA_VISIBLE_DEVICES=2 python scripts/train4.py --exp_name test_new_data_lc_12 --mode teacher --teacher_type UGP_v3 --use_label_correction --untrusted_age_threshold 60 --pseudo_label_interval 10 --pseudo_label_start_step 500 --confidence_threshold 0.7 --data_dir ./data/ad --lr 1e-5;
CUDA_VISIBLE_DEVICES=1 python scripts/train4.py --exp_name test_new_data_lc_12 --mode teacher --teacher_type UGP_v3 --use_label_correction --untrusted_age_threshold 60 --pseudo_label_interval 10 --pseudo_label_start_step 500 --confidence_threshold 0.7 --data_dir ./data/af --lr 1e-5;
CUDA_VISIBLE_DEVICES=2 python scripts/train4.py --exp_name test_new_data_lc_12 --mode teacher --teacher_type UGP_v3 --use_label_correction --untrusted_age_threshold 60 --pseudo_label_interval 10 --pseudo_label_start_step 500 --confidence_threshold 0.7 --data_dir ./data/ms --lr 1e-5;
CUDA_VISIBLE_DEVICES=2 python scripts/train4.py --exp_name test_new_data_lc_12 --mode teacher --teacher_type UGP_v3 --use_label_correction --untrusted_age_threshold 60 --pseudo_label_interval 10 --pseudo_label_start_step 500 --confidence_threshold 0.7 --data_dir ./data/uc --lr 1e-5;

#test_new_data_lc_13
CUDA_VISIBLE_DEVICES=3 python scripts/train4.py --exp_name test_new_data_lc_13 --mode teacher --teacher_type MLP --use_label_correction --untrusted_age_threshold 65 --pseudo_label_interval 20 --pseudo_label_start_step 400 --confidence_threshold 0.7 --data_dir ./data/ad --lr 1e-4;
CUDA_VISIBLE_DEVICES=3 python scripts/train4.py --exp_name test_new_data_lc_13 --mode teacher --teacher_type MLP --use_label_correction --untrusted_age_threshold 65 --pseudo_label_interval 20 --pseudo_label_start_step 400 --confidence_threshold 0.7 --data_dir ./data/af --lr 1e-4;
CUDA_VISIBLE_DEVICES=3 python scripts/train4.py --exp_name test_new_data_lc_13 --mode teacher --teacher_type MLP --use_label_correction --untrusted_age_threshold 65 --pseudo_label_interval 20 --pseudo_label_start_step 400 --confidence_threshold 0.7 --data_dir ./data/ms --lr 1e-4;
CUDA_VISIBLE_DEVICES=3 python scripts/train4.py --exp_name test_new_data_lc_13 --mode teacher --teacher_type MLP --use_label_correction --untrusted_age_threshold 65 --pseudo_label_interval 20 --pseudo_label_start_step 400 --confidence_threshold 0.7 --data_dir ./data/uc --lr 1e-4;

CUDA_VISIBLE_DEVICES=3 python scripts/train4.py --exp_name test_new_data_lc_13 --mode teacher --teacher_type UGP_v3 --use_label_correction --untrusted_age_threshold 65 --pseudo_label_interval 20 --pseudo_label_start_step 400 --confidence_threshold 0.7 --data_dir ./data/ad --lr 1e-5;
CUDA_VISIBLE_DEVICES=1 python scripts/train4.py --exp_name test_new_data_lc_13 --mode teacher --teacher_type UGP_v3 --use_label_correction --untrusted_age_threshold 65 --pseudo_label_interval 20 --pseudo_label_start_step 400 --confidence_threshold 0.7 --data_dir ./data/af --lr 1e-5;
CUDA_VISIBLE_DEVICES=3 python scripts/train4.py --exp_name test_new_data_lc_13 --mode teacher --teacher_type UGP_v3 --use_label_correction --untrusted_age_threshold 65 --pseudo_label_interval 20 --pseudo_label_start_step 400 --confidence_threshold 0.7 --data_dir ./data/ms --lr 1e-5;
CUDA_VISIBLE_DEVICES=3 python scripts/train4.py --exp_name test_new_data_lc_13 --mode teacher --teacher_type UGP_v3 --use_label_correction --untrusted_age_threshold 65 --pseudo_label_interval 20 --pseudo_label_start_step 400 --confidence_threshold 0.7 --data_dir ./data/uc --lr 1e-5;

#test_new_data_lc_14
CUDA_VISIBLE_DEVICES=0 python scripts/train4.py --exp_name test_new_data_lc_14 --mode teacher --teacher_type MLP --use_label_correction --untrusted_age_threshold 65 --pseudo_label_interval 20 --pseudo_label_start_step 200 --confidence_threshold 0.6 --data_dir ./data/ad --lr 1e-4;
CUDA_VISIBLE_DEVICES=0 python scripts/train4.py --exp_name test_new_data_lc_14 --mode teacher --teacher_type MLP --use_label_correction --untrusted_age_threshold 65 --pseudo_label_interval 20 --pseudo_label_start_step 200 --confidence_threshold 0.6 --data_dir ./data/af --lr 1e-4;
CUDA_VISIBLE_DEVICES=0 python scripts/train4.py --exp_name test_new_data_lc_14 --mode teacher --teacher_type MLP --use_label_correction --untrusted_age_threshold 65 --pseudo_label_interval 20 --pseudo_label_start_step 200 --confidence_threshold 0.6 --data_dir ./data/ms --lr 1e-4;
CUDA_VISIBLE_DEVICES=0 python scripts/train4.py --exp_name test_new_data_lc_14 --mode teacher --teacher_type MLP --use_label_correction --untrusted_age_threshold 65 --pseudo_label_interval 20 --pseudo_label_start_step 200 --confidence_threshold 0.6 --data_dir ./data/uc --lr 1e-4;

CUDA_VISIBLE_DEVICES=2 python scripts/train4.py --exp_name test_new_data_lc_14 --mode teacher --teacher_type UGP_v3 --use_label_correction --untrusted_age_threshold 65 --pseudo_label_interval 20 --pseudo_label_start_step 200 --confidence_threshold 0.6 --data_dir ./data/ad --lr 1e-5;
CUDA_VISIBLE_DEVICES=2 python scripts/train4.py --exp_name test_new_data_lc_14 --mode teacher --teacher_type UGP_v3 --use_label_correction --untrusted_age_threshold 65 --pseudo_label_interval 20 --pseudo_label_start_step 200 --confidence_threshold 0.6 --data_dir ./data/af --lr 1e-5;
CUDA_VISIBLE_DEVICES=2 python scripts/train4.py --exp_name test_new_data_lc_14 --mode teacher --teacher_type UGP_v3 --use_label_correction --untrusted_age_threshold 65 --pseudo_label_interval 20 --pseudo_label_start_step 200 --confidence_threshold 0.6 --data_dir ./data/ms --lr 1e-5;
CUDA_VISIBLE_DEVICES=2 python scripts/train4.py --exp_name test_new_data_lc_14 --mode teacher --teacher_type UGP_v3 --use_label_correction --untrusted_age_threshold 65 --pseudo_label_interval 20 --pseudo_label_start_step 200 --confidence_threshold 0.6 --data_dir ./data/uc --lr 1e-5;

#test_new_data_lc_15
CUDA_VISIBLE_DEVICES=1 python scripts/train4.py --exp_name test_new_data_lc_15 --mode teacher --teacher_type MLP --use_label_correction --untrusted_age_threshold 65 --pseudo_label_interval 10 --pseudo_label_start_step 200 --confidence_threshold 0.7 --data_dir ./data/ad --lr 1e-4;
CUDA_VISIBLE_DEVICES=1 python scripts/train4.py --exp_name test_new_data_lc_15 --mode teacher --teacher_type MLP --use_label_correction --untrusted_age_threshold 65 --pseudo_label_interval 10 --pseudo_label_start_step 200 --confidence_threshold 0.7 --data_dir ./data/af --lr 1e-4;
CUDA_VISIBLE_DEVICES=1 python scripts/train4.py --exp_name test_new_data_lc_15 --mode teacher --teacher_type MLP --use_label_correction --untrusted_age_threshold 65 --pseudo_label_interval 10 --pseudo_label_start_step 200 --confidence_threshold 0.7 --data_dir ./data/ms --lr 1e-4;
CUDA_VISIBLE_DEVICES=1 python scripts/train4.py --exp_name test_new_data_lc_15 --mode teacher --teacher_type MLP --use_label_correction --untrusted_age_threshold 65 --pseudo_label_interval 10 --pseudo_label_start_step 200 --confidence_threshold 0.7 --data_dir ./data/uc --lr 1e-4;

CUDA_VISIBLE_DEVICES=3 python scripts/train4.py --exp_name test_new_data_lc_15 --mode teacher --teacher_type UGP_v3 --use_label_correction --untrusted_age_threshold 65 --pseudo_label_interval 10 --pseudo_label_start_step 200 --confidence_threshold 0.7 --data_dir ./data/ad --lr 1e-5;
CUDA_VISIBLE_DEVICES=3 python scripts/train4.py --exp_name test_new_data_lc_15 --mode teacher --teacher_type UGP_v3 --use_label_correction --untrusted_age_threshold 65 --pseudo_label_interval 10 --pseudo_label_start_step 200 --confidence_threshold 0.7 --data_dir ./data/af --lr 1e-5;
CUDA_VISIBLE_DEVICES=3 python scripts/train4.py --exp_name test_new_data_lc_15 --mode teacher --teacher_type UGP_v3 --use_label_correction --untrusted_age_threshold 65 --pseudo_label_interval 10 --pseudo_label_start_step 200 --confidence_threshold 0.7 --data_dir ./data/ms --lr 1e-5;
CUDA_VISIBLE_DEVICES=3 python scripts/train4.py --exp_name test_new_data_lc_15 --mode teacher --teacher_type UGP_v3 --use_label_correction --untrusted_age_threshold 65 --pseudo_label_interval 10 --pseudo_label_start_step 200 --confidence_threshold 0.7 --data_dir ./data/uc --lr 1e-5;

#test_new_data_lc_16
CUDA_VISIBLE_DEVICES=2 python scripts/train4.py --exp_name test_new_data_lc_16 --mode teacher --teacher_type MLP --use_label_correction --untrusted_age_threshold 65 --pseudo_label_interval 10 --pseudo_label_start_step 200 --confidence_threshold 0.6 --data_dir ./data/ad --lr 1e-4;
CUDA_VISIBLE_DEVICES=2 python scripts/train4.py --exp_name test_new_data_lc_16 --mode teacher --teacher_type MLP --use_label_correction --untrusted_age_threshold 65 --pseudo_label_interval 10 --pseudo_label_start_step 200 --confidence_threshold 0.6 --data_dir ./data/af --lr 1e-4;
CUDA_VISIBLE_DEVICES=2 python scripts/train4.py --exp_name test_new_data_lc_16 --mode teacher --teacher_type MLP --use_label_correction --untrusted_age_threshold 65 --pseudo_label_interval 10 --pseudo_label_start_step 200 --confidence_threshold 0.6 --data_dir ./data/ms --lr 1e-4;
CUDA_VISIBLE_DEVICES=2 python scripts/train4.py --exp_name test_new_data_lc_16 --mode teacher --teacher_type MLP --use_label_correction --untrusted_age_threshold 65 --pseudo_label_interval 10 --pseudo_label_start_step 200 --confidence_threshold 0.6 --data_dir ./data/uc --lr 1e-4;

CUDA_VISIBLE_DEVICES=2 python scripts/train4.py --exp_name test_new_data_lc_16 --mode teacher --teacher_type UGP_v3 --use_label_correction --untrusted_age_threshold 65 --pseudo_label_interval 10 --pseudo_label_start_step 200 --confidence_threshold 0.6 --data_dir ./data/ad --lr 1e-5;
CUDA_VISIBLE_DEVICES=2 python scripts/train4.py --exp_name test_new_data_lc_16 --mode teacher --teacher_type UGP_v3 --use_label_correction --untrusted_age_threshold 65 --pseudo_label_interval 10 --pseudo_label_start_step 200 --confidence_threshold 0.6 --data_dir ./data/af --lr 1e-5;
CUDA_VISIBLE_DEVICES=2 python scripts/train4.py --exp_name test_new_data_lc_16 --mode teacher --teacher_type UGP_v3 --use_label_correction --untrusted_age_threshold 65 --pseudo_label_interval 10 --pseudo_label_start_step 200 --confidence_threshold 0.6 --data_dir ./data/ms --lr 1e-5;
CUDA_VISIBLE_DEVICES=2 python scripts/train4.py --exp_name test_new_data_lc_16 --mode teacher --teacher_type UGP_v3 --use_label_correction --untrusted_age_threshold 65 --pseudo_label_interval 10 --pseudo_label_start_step 200 --confidence_threshold 0.6 --data_dir ./data/uc --lr 1e-5;

#test_new_data_lc_17
CUDA_VISIBLE_DEVICES=3 python scripts/train4.py --exp_name test_new_data_lc_17 --mode teacher --teacher_type MLP --use_label_correction --untrusted_age_threshold 65 --pseudo_label_interval 20 --pseudo_label_start_step 300 --confidence_threshold 0.6 --data_dir ./data/ad --lr 1e-4;
CUDA_VISIBLE_DEVICES=3 python scripts/train4.py --exp_name test_new_data_lc_17 --mode teacher --teacher_type MLP --use_label_correction --untrusted_age_threshold 65 --pseudo_label_interval 20 --pseudo_label_start_step 300 --confidence_threshold 0.6 --data_dir ./data/af --lr 1e-4;
CUDA_VISIBLE_DEVICES=3 python scripts/train4.py --exp_name test_new_data_lc_17 --mode teacher --teacher_type MLP --use_label_correction --untrusted_age_threshold 65 --pseudo_label_interval 20 --pseudo_label_start_step 300 --confidence_threshold 0.6 --data_dir ./data/ms --lr 1e-4;
CUDA_VISIBLE_DEVICES=3 python scripts/train4.py --exp_name test_new_data_lc_17 --mode teacher --teacher_type MLP --use_label_correction --untrusted_age_threshold 65 --pseudo_label_interval 20 --pseudo_label_start_step 300 --confidence_threshold 0.6 --data_dir ./data/uc --lr 1e-4;

CUDA_VISIBLE_DEVICES=3 python scripts/train4.py --exp_name test_new_data_lc_17 --mode teacher --teacher_type UGP_v3 --use_label_correction --untrusted_age_threshold 65 --pseudo_label_interval 20 --pseudo_label_start_step 300 --confidence_threshold 0.6 --data_dir ./data/ad --lr 1e-5;
CUDA_VISIBLE_DEVICES=3 python scripts/train4.py --exp_name test_new_data_lc_17 --mode teacher --teacher_type UGP_v3 --use_label_correction --untrusted_age_threshold 65 --pseudo_label_interval 20 --pseudo_label_start_step 300 --confidence_threshold 0.6 --data_dir ./data/af --lr 1e-5;
CUDA_VISIBLE_DEVICES=3 python scripts/train4.py --exp_name test_new_data_lc_17 --mode teacher --teacher_type UGP_v3 --use_label_correction --untrusted_age_threshold 65 --pseudo_label_interval 20 --pseudo_label_start_step 300 --confidence_threshold 0.6 --data_dir ./data/ms --lr 1e-5;
CUDA_VISIBLE_DEVICES=3 python scripts/train4.py --exp_name test_new_data_lc_17 --mode teacher --teacher_type UGP_v3 --use_label_correction --untrusted_age_threshold 65 --pseudo_label_interval 20 --pseudo_label_start_step 300 --confidence_threshold 0.6 --data_dir ./data/uc --lr 1e-5;

#test_new_data_lc_new1 (hard loss)
CUDA_VISIBLE_DEVICES=2 python scripts/train4.py --exp_name test_new_data_lc_new1 --mode teacher --teacher_type UGP_v3 --use_label_correction --pseudo_label_interval 20 --data_dir ./data/ad --lr 1e-5;
CUDA_VISIBLE_DEVICES=3 python scripts/train4.py --exp_name test_new_data_lc_new1 --mode teacher --teacher_type UGP_v3 --use_label_correction --pseudo_label_interval 20 --data_dir ./data/af --lr 1e-5;
CUDA_VISIBLE_DEVICES=1 python scripts/train4.py --exp_name test_new_data_lc_new1 --mode teacher --teacher_type UGP_v3 --use_label_correction --pseudo_label_interval 20 --data_dir ./data/ms --lr 1e-5;
CUDA_VISIBLE_DEVICES=0 python scripts/train4.py --exp_name test_new_data_lc_new1 --mode teacher --teacher_type UGP_v3 --use_label_correction --pseudo_label_interval 20 --data_dir ./data/uc --lr 1e-5;

#test_new_data_lc_new2 (use soft loss)
CUDA_VISIBLE_DEVICES=1 python scripts/train4.py --exp_name test_new_data_lc_new2 --mode teacher --teacher_type MLP --use_label_correction --data_dir ./data/ad --lr 1e-4 --eta 0.1;
CUDA_VISIBLE_DEVICES=2 python scripts/train4.py --exp_name test_new_data_lc_new2 --mode teacher --teacher_type MLP --use_label_correction --data_dir ./data/af --lr 1e-3 --eta 0.1;
CUDA_VISIBLE_DEVICES=3 python scripts/train4.py --exp_name test_new_data_lc_new2 --mode teacher --teacher_type MLP --use_label_correction --data_dir ./data/ms --lr 1e-4 --eta 0.1;
CUDA_VISIBLE_DEVICES=0 python scripts/train4.py --exp_name test_new_data_lc_new2 --mode teacher --teacher_type MLP --use_label_correction --data_dir ./data/uc --lr 1e-4 --eta 0.1;

CUDA_VISIBLE_DEVICES=1 python scripts/train4.py --exp_name test_new_data_lc_new2 --mode teacher --teacher_type UGP_v3 --use_label_correction --data_dir ./data/ad --lr 1e-5 --eta 0.1;
CUDA_VISIBLE_DEVICES=2 python scripts/train4.py --exp_name test_new_data_lc_new2 --mode teacher --teacher_type UGP_v3 --use_label_correction --data_dir ./data/af --lr 1e-4 --eta 0.1;
CUDA_VISIBLE_DEVICES=3 python scripts/train4.py --exp_name test_new_data_lc_new2 --mode teacher --teacher_type UGP_v3 --use_label_correction --data_dir ./data/ms --lr 1e-5 --eta 0.1;
CUDA_VISIBLE_DEVICES=0 python scripts/train4.py --exp_name test_new_data_lc_new2 --mode teacher --teacher_type UGP_v3 --use_label_correction --data_dir ./data/uc --lr 1e-5 --eta 0.1;


# test_new_data_lc_new3 (eta↑)
CUDA_VISIBLE_DEVICES=1 python scripts/train4.py --exp_name test_new_data_lc_new3 --mode teacher --teacher_type MLP --use_label_correction --pseudo_label_interval 20 --data_dir ./data/ad --lr 1e-4 --eta 1.0;
CUDA_VISIBLE_DEVICES=2 python scripts/train4.py --exp_name test_new_data_lc_new3 --mode teacher --teacher_type MLP --use_label_correction --pseudo_label_interval 20 --data_dir ./data/af --lr 1e-3 --eta 1.0;
CUDA_VISIBLE_DEVICES=3 python scripts/train4.py --exp_name test_new_data_lc_new3 --mode teacher --teacher_type MLP --use_label_correction --pseudo_label_interval 20 --data_dir ./data/ms --lr 1e-4 --eta 1.0;
CUDA_VISIBLE_DEVICES=0 python scripts/train4.py --exp_name test_new_data_lc_new3 --mode teacher --teacher_type MLP --use_label_correction --pseudo_label_interval 20 --data_dir ./data/uc --lr 1e-4 --eta 1.0;

CUDA_VISIBLE_DEVICES=1 python scripts/train4.py --exp_name test_new_data_lc_new3 --mode teacher --teacher_type UGP_v3 --use_label_correction --pseudo_label_interval 20 --data_dir ./data/ad --lr 1e-5 --eta 1.0;
CUDA_VISIBLE_DEVICES=2 python scripts/train4.py --exp_name test_new_data_lc_new3 --mode teacher --teacher_type UGP_v3 --use_label_correction --pseudo_label_interval 20 --data_dir ./data/af --lr 1e-4 --eta 1.0;
CUDA_VISIBLE_DEVICES=3 python scripts/train4.py --exp_name test_new_data_lc_new3 --mode teacher --teacher_type UGP_v3 --use_label_correction --pseudo_label_interval 20 --data_dir ./data/ms --lr 1e-5 --eta 1.0;
CUDA_VISIBLE_DEVICES=0 python scripts/train4.py --exp_name test_new_data_lc_new3 --mode teacher --teacher_type UGP_v3 --use_label_correction --pseudo_label_interval 20 --data_dir ./data/uc --lr 1e-5 --eta 1.0;

# test_new_data_af_lr
CUDA_VISIBLE_DEVICES=1 python scripts/train4.py --exp_name test_new_data_af_lr --mode teacher --teacher_type MLP --use_label_correction --data_dir ./data/af --lr 1e-3;
CUDA_VISIBLE_DEVICES=2 python scripts/train4.py --exp_name test_new_data_af_lr --mode teacher --teacher_type UGP_v3 --use_label_correction --data_dir ./data/af --lr 1e-4;


# full (8 + 9 + 6 + + 4 + 5)
python scripts/train4.py --exp_name full --mode teacher --teacher_type MLP --age_threshold 65 --data_dir ./data/ad --lr 1e-4;
python scripts/train4.py --exp_name full --mode student --teacher_type MLP --student_type MLP --age_threshold 65 --data_dir ./data/ad --teacher_model_exp_name full --teacher_model_lr 1e-4 --lr 1e-4;
python scripts/train4.py --exp_name full --mode teacher --teacher_type AgeAwareMLP1 --age_threshold 65 --data_dir ./data/ad --lr 1e-4;
python scripts/train4.py --exp_name full --mode student --teacher_type AgeAwareMLP1 --student_type AgeAwareMLP1 --age_threshold 65 --data_dir ./data/ad --teacher_model_exp_name full --teacher_model_lr 1e-4 --lr 1e-4;
python scripts/train4.py --exp_name full --mode student --teacher_type AgeAwareMLP1 --student_type MLP --age_threshold 65 --data_dir ./data/ad --teacher_model_exp_name full --teacher_model_lr 1e-4 --lr 1e-4;
python scripts/train4.py --exp_name full --mode teacher --teacher_type AgeAwareMLP2 --age_threshold 65 --data_dir ./data/ad --lr 1e-4;
python scripts/train4.py --exp_name full --mode student --teacher_type AgeAwareMLP2 --student_type AgeAwareMLP2 --age_threshold 65 --data_dir ./data/ad --teacher_model_exp_name full --teacher_model_lr 1e-4 --lr 1e-4;
python scripts/train4.py --exp_name full --mode student --teacher_type AgeAwareMLP2 --student_type MLP --age_threshold 65 --data_dir ./data/ad --teacher_model_exp_name full --teacher_model_lr 1e-4 --lr 1e-4;

CUDA_VISIBLE_DEVICES=1 python scripts/train4.py --exp_name full --mode teacher --teacher_type UGP_v1 --data_dir ./data/ad --lr 1e-5;
CUDA_VISIBLE_DEVICES=1 python scripts/train4.py --exp_name full --mode student --teacher_type UGP_v1 --student_type UGP_v1 --data_dir ./data/ad --teacher_model_exp_name full --teacher_model_lr 1e-5 --lr 1e-5;
CUDA_VISIBLE_DEVICES=1 python scripts/train4.py --exp_name full --mode student --teacher_type UGP_v1 --student_type MLP --data_dir ./data/ad --teacher_model_exp_name full --teacher_model_lr 1e-5 --lr 1e-5;
CUDA_VISIBLE_DEVICES=1 python scripts/train4.py --exp_name full --mode teacher --teacher_type UGP_v2 --data_dir ./data/ad --lr 1e-5;
CUDA_VISIBLE_DEVICES=1 python scripts/train4.py --exp_name full --mode student --teacher_type UGP_v2 --student_type UGP_v2 --data_dir ./data/ad --teacher_model_exp_name full --teacher_model_lr 1e-5 --lr 1e-5;
CUDA_VISIBLE_DEVICES=1 python scripts/train4.py --exp_name full --mode student --teacher_type UGP_v2 --student_type MLP --data_dir ./data/ad --teacher_model_exp_name full --teacher_model_lr 1e-5 --lr 1e-5;
CUDA_VISIBLE_DEVICES=3 python scripts/train4.py --exp_name full --mode teacher --teacher_type UGP_v3 --data_dir ./data/ad --lr 1e-5;
CUDA_VISIBLE_DEVICES=3 python scripts/train4.py --exp_name full --mode student --teacher_type UGP_v3 --student_type UGP_v3 --data_dir ./data/ad --teacher_model_exp_name full --teacher_model_lr 1e-5 --lr 1e-5;
CUDA_VISIBLE_DEVICES=3 python scripts/train4.py --exp_name full --mode student --teacher_type UGP_v3 --student_type MLP --data_dir ./data/ad --teacher_model_exp_name full --teacher_model_lr 1e-5 --lr 1e-5;

CUDA_VISIBLE_DEVICES=2 python scripts/train4.py --exp_name full --mode teacher --teacher_type AgeUGP_v1 --data_dir ./data/ad --lr 1e-5;
CUDA_VISIBLE_DEVICES=2 python scripts/train4.py --exp_name full --mode student --teacher_type AgeUGP_v1 --student_type AgeUGP_v1 --data_dir ./data/ad --teacher_model_exp_name full --teacher_model_lr 1e-5 --lr 1e-5;
CUDA_VISIBLE_DEVICES=2 python scripts/train4.py --exp_name full --mode student --teacher_type AgeUGP_v1 --student_type MLP --data_dir ./data/ad --teacher_model_exp_name full --teacher_model_lr 1e-5 --lr 1e-5;
CUDA_VISIBLE_DEVICES=2 python scripts/train4.py --exp_name full --mode teacher --teacher_type AgeUGP_v2 --data_dir ./data/ad --lr 1e-5;
CUDA_VISIBLE_DEVICES=2 python scripts/train4.py --exp_name full --mode student --teacher_type AgeUGP_v2 --student_type AgeUGP_v2 --data_dir ./data/ad --teacher_model_exp_name full --teacher_model_lr 1e-5 --lr 1e-5;
CUDA_VISIBLE_DEVICES=2 python scripts/train4.py --exp_name full --mode student --teacher_type AgeUGP_v2 --student_type MLP --data_dir ./data/ad --teacher_model_exp_name full --teacher_model_lr 1e-5 --lr 1e-5;

CUDA_VISIBLE_DEVICES=1 python scripts/train4.py --exp_name full --mode student --teacher_type UGP_v3 --student_type AgeAwareMLP1 --data_dir ./data/ad --teacher_model_exp_name full --teacher_model_lr 1e-5 --lr 1e-4;
CUDA_VISIBLE_DEVICES=1 python scripts/train4.py --exp_name full --mode student --teacher_type UGP_v3 --student_type AgeAwareMLP2 --data_dir ./data/ad --teacher_model_exp_name full --teacher_model_lr 1e-5 --lr 1e-4;
CUDA_VISIBLE_DEVICES=2 python scripts/train4.py --exp_name full --mode student --teacher_type AgeAwareMLP1 --student_type UGP_v3 --data_dir ./data/ad --teacher_model_exp_name full --teacher_model_lr 1e-4 --lr 1e-5;
CUDA_VISIBLE_DEVICES=2 python scripts/train4.py --exp_name full --mode student --teacher_type AgeAwareMLP2 --student_type UGP_v3 --data_dir ./data/ad --teacher_model_exp_name full --teacher_model_lr 1e-4 --lr 1e-5;

python scripts/train_classical.py --exp_name full --model_type LinearRegression --data_dir ./data/ad;
python scripts/train_classical.py --exp_name full --model_type XGBoostClassifier --data_dir ./data/ad;
python scripts/train_classical.py --exp_name full --model_type XGBoostRegressor --data_dir ./data/ad;
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

# student_enhanced_lr3 (lr*10) (5 + 4 + 4)
python scripts/train4.py --exp_name student_enhanced_lr3 --mode student --teacher_type MLP --student_type MLP --age_threshold 65 --data_dir ./data/ad --teacher_model_exp_name full --lr 0.001;
python scripts/train4.py --exp_name student_enhanced_lr3 --mode student --teacher_type AgeAwareMLP1 --student_type AgeAwareMLP1 --age_threshold 65 --data_dir ./data/ad --teacher_model_exp_name full --lr 0.001;
python scripts/train4.py --exp_name student_enhanced_lr3 --mode student --teacher_type AgeAwareMLP1 --student_type MLP --age_threshold 65 --data_dir ./data/ad --teacher_model_exp_name full --lr 0.001;
python scripts/train4.py --exp_name student_enhanced_lr3 --mode student --teacher_type AgeAwareMLP2 --student_type AgeAwareMLP2 --age_threshold 65 --data_dir ./data/ad --teacher_model_exp_name full --lr 0.001;
python scripts/train4.py --exp_name student_enhanced_lr3 --mode student --teacher_type AgeAwareMLP2 --student_type MLP --age_threshold 65 --data_dir ./data/ad --teacher_model_exp_name full --lr 0.001;

python scripts/train4.py --exp_name student_enhanced_lr3 --mode student --teacher_type UGP_v1 --student_type UGP_v1 --data_dir ./data/ad --teacher_model_exp_name full --lr 0.001;
python scripts/train4.py --exp_name student_enhanced_lr3 --mode student --teacher_type UGP_v1 --student_type MLP --data_dir ./data/ad --teacher_model_exp_name full --lr 0.001;
python scripts/train4.py --exp_name student_enhanced_lr3 --mode student --teacher_type UGP_v2 --student_type UGP_v2 --data_dir ./data/ad --teacher_model_exp_name full --lr 0.001;
python scripts/train4.py --exp_name student_enhanced_lr3 --mode student --teacher_type UGP_v2 --student_type MLP --data_dir ./data/ad --teacher_model_exp_name full --lr 0.001;

python scripts/train4.py --exp_name student_enhanced_lr3 --mode student --teacher_type AgeUGP_v1 --student_type AgeUGP_v1 --data_dir ./data/ad --teacher_model_exp_name full --lr 0.0001;
python scripts/train4.py --exp_name student_enhanced_lr3 --mode student --teacher_type AgeUGP_v1 --student_type MLP --data_dir ./data/ad --teacher_model_exp_name full --lr 0.0001;
python scripts/train4.py --exp_name student_enhanced_lr3 --mode student --teacher_type AgeUGP_v2 --student_type AgeUGP_v2 --data_dir ./data/ad --teacher_model_exp_name full --lr 0.0001;
python scripts/train4.py --exp_name student_enhanced_lr3 --mode student --teacher_type AgeUGP_v2 --student_type MLP --data_dir ./data/ad --teacher_model_exp_name full --lr 0.0001;

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