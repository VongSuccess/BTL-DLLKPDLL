Bộ file gồm:
1. crop_yield_pipeline.py  -> code chính
2. requirements.txt       -> thư viện cần cài
3. DOWNLOAD_DATASET.txt   -> cách tải dataset từ Kaggle
4. run_crop_yield.bat     -> file chạy nhanh trên Windows

Cách chạy:
- Tải dataset Kaggle vào thư mục data/
- Cài thư viện: pip install -r requirements.txt
- Chạy: python crop_yield_pipeline.py

Kết quả sinh ra trong thư mục outputs/ gồm:
- regression_results.csv
- classification_results.csv
- cluster_profile_yield.csv
- top_rules_apriori.csv
- top_rules_fpgrowth.csv
- recommendations.txt
- actual_vs_predicted.png
- yield_trend_by_year.png
- clusters_pca.png
- summary.txt
