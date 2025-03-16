source /data3/lihan/miniconda3/bin/activate ageaware

python src/demo.py

python src/table.py --disease ad
python src/table.py --disease ms
python src/table.py --disease uc
python src/table.py --disease all

python scripts/train_classical.py --mode teacher --data_dir ./data/ms --model_type XGBoost --exp_name test_classical;
python scripts/train_classical.py --mode student --data_dir ./data/ms --model_type XGBoost --teacher_type XGBoost --teacher_model_exp_name test_classical --exp_name test_classical;

python scripts/train_classical.py --mode teacher --data_dir ./data/ms --model_type LinearRegression --exp_name test_classical;
python scripts/train_classical.py --mode student --data_dir ./data/ms --model_type LinearRegression --teacher_type LinearRegression --teacher_model_exp_name test_classical --exp_name test_classical;

python scripts/train_classical.py --mode teacher --data_dir ./data/ms --model_type LightGBM --exp_name test_classical;
python scripts/train_classical.py --mode student --data_dir ./data/ms --model_type LightGBM --teacher_type LightGBM --teacher_model_exp_name test_classical --exp_name test_classical;

python scripts/train4.py --exp_name test --mode teacher --teacher_type MLP --age_threshold 65 --data_dir ./data/ad;python scripts/train4.py --exp_name test --mode student --teacher_type MLP --student_type MLP --age_threshold 65 --data_dir ./data/ad;python scripts/train4.py --exp_name test --mode teacher --teacher_type AgeAwareMLP1 --age_threshold 65 --data_dir ./data/ad;python scripts/train4.py --exp_name test --mode student --teacher_type AgeAwareMLP1 --student_type AgeAwareMLP1 --age_threshold 65 --data_dir ./data/ad;python scripts/train4.py --exp_name test --mode student --teacher_type AgeAwareMLP1 --student_type MLP --age_threshold 65 --data_dir ./data/ad;python scripts/train4.py --exp_name test --mode teacher --teacher_type AgeAwareMLP2 --age_threshold 65 --data_dir ./data/ad;python scripts/train4.py --exp_name test --mode student --teacher_type AgeAwareMLP2 --student_type AgeAwareMLP2 --age_threshold 65 --data_dir ./data/ad;python scripts/train4.py --exp_name test --mode student --teacher_type AgeAwareMLP2 --student_type MLP --age_threshold 65 --data_dir ./data/ad;

python scripts/train4.py --exp_name test --mode teacher --teacher_type MLP --age_threshold 65 --data_dir ./data/ms;python scripts/train4.py --exp_name test --mode student --teacher_type MLP --student_type MLP --age_threshold 65 --data_dir ./data/ms;
python scripts/train4.py --exp_name test --mode teacher --teacher_type AgeAwareMLP1 --age_threshold 65 --data_dir ./data/ms;python scripts/train4.py --exp_name test --mode student --teacher_type AgeAwareMLP1 --student_type AgeAwareMLP1 --age_threshold 65 --data_dir ./data/ms;python scripts/train4.py --exp_name test --mode student --teacher_type AgeAwareMLP1 --student_type MLP --age_threshold 65 --data_dir ./data/ms;
python scripts/train4.py --exp_name test --mode teacher --teacher_type AgeAwareMLP2 --age_threshold 65 --data_dir ./data/ms;python scripts/train4.py --exp_name test --mode student --teacher_type AgeAwareMLP2 --student_type AgeAwareMLP2 --age_threshold 65 --data_dir ./data/ms;python scripts/train4.py --exp_name test --mode student --teacher_type AgeAwareMLP2 --student_type MLP --age_threshold 65 --data_dir ./data/ms;

python scripts/train4.py --exp_name test --mode teacher --teacher_type MLP --age_threshold 65 --data_dir ./data/uc;python scripts/train4.py --exp_name test --mode student --teacher_type MLP --student_type MLP --age_threshold 65 --data_dir ./data/uc;
python scripts/train4.py --exp_name test --mode teacher --teacher_type AgeAwareMLP1 --age_threshold 65 --data_dir ./data/uc;python scripts/train4.py --exp_name test --mode student --teacher_type AgeAwareMLP1 --student_type AgeAwareMLP1 --age_threshold 65 --data_dir ./data/uc;python scripts/train4.py --exp_name test --mode student --teacher_type AgeAwareMLP1 --student_type MLP --age_threshold 65 --data_dir ./data/uc;
python scripts/train4.py --exp_name test --mode teacher --teacher_type AgeAwareMLP2 --age_threshold 65 --data_dir ./data/uc;python scripts/train4.py --exp_name test --mode student --teacher_type AgeAwareMLP2 --student_type AgeAwareMLP2 --age_threshold 65 --data_dir ./data/uc;python scripts/train4.py --exp_name test --mode student --teacher_type AgeAwareMLP2 --student_type MLP --age_threshold 65 --data_dir ./data/uc;python src/analyze_age_matrix_2.py


