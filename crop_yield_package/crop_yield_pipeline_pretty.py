
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import silhouette_score
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

import matplotlib
matplotlib.use("Agg")  # Sử dụng backend không giao diện để lưu hình
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")  # Bỏ qua cảnh báo

# Kiểm tra xem có thư viện XGBoost không
try:
    from xgboost import XGBRegressor, XGBClassifier
    HAS_XGB = True
except Exception:
    HAS_XGB = False

# Kiểm tra xem có thư viện mlxtend không (cho luật kết hợp)
try:
    from mlxtend.frequent_patterns import apriori, association_rules, fpgrowth
    HAS_MLXTEND = True
except Exception:
    HAS_MLXTEND = False

# Danh sách tên file dataset có thể có
DATA_CANDIDATES = [
    "yield_df.csv",
    "yield.csv",
    "crop_yield.csv",
    "crop_yield_data.csv",
]


# ---------- Cài đặt kiểu dáng biểu đồ ----------
def set_plot_style():
    """Thiết lập kiểu dáng cho các biểu đồ matplotlib."""
    plt.rcParams.update({
        "figure.figsize": (9, 5.5),
        "figure.dpi": 140,
        "savefig.dpi": 220,
        "axes.titlesize": 15,
        "axes.labelsize": 12,
        "axes.titleweight": "bold",
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": "--",
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
    })


def save_fig(path: Path):
    """Lưu biểu đồ vào file với cài đặt chuẩn."""
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight", dpi=220)
    plt.close()


# ---------- Tiện ích xử lý dữ liệu ----------
def find_dataset_path(data_dir: str = "data") -> Path:
    """Tìm đường dẫn đến file dataset CSV trong thư mục data."""
    data_dir = Path(data_dir)
    for name in DATA_CANDIDATES:
        p = data_dir / name
        if p.exists():
            return p
    csvs = list(data_dir.glob("*.csv"))
    if csvs:
        return csvs[0]
    raise FileNotFoundError(
        f"Không tìm thấy file CSV trong thư mục {data_dir.resolve()}. "
        f"Hãy tải dataset Kaggle và đặt file vào đây."
    )


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Chuẩn hóa tên cột: lowercase, thay thế ký tự đặc biệt."""
    df = df.copy()
    df.columns = [
        c.strip().lower().replace("/", "_").replace(" ", "_").replace("-", "_")
        for c in df.columns
    ]
    return df


def find_first_existing(cols, candidates):
    """Tìm cột đầu tiên tồn tại trong danh sách ứng viên."""
    for c in candidates:
        if c in cols:
            return c
    return None


def clean_numeric_series(s: pd.Series) -> pd.Series:
    """Làm sạch chuỗi số: loại bỏ dấu phẩy, ký tự không phải số."""
    return pd.to_numeric(
        s.astype(str)
        .str.replace(",", "", regex=False)
        .str.replace(r"[^0-9.\-]", "", regex=True),
        errors="coerce",
    )


def load_and_clean(data_dir="data"):
    """Tải và làm sạch dữ liệu từ file CSV."""
    path = find_dataset_path(data_dir)
    df = pd.read_csv(path)
    df = normalize_columns(df)

    # Tìm cột mục tiêu (yield)
    target_col = find_first_existing(df.columns, ["yield", "hg_ha_yield", "hg/ha_yield"])
    if target_col is None:
        target_col = next((c for c in df.columns if "yield" in c), None)
    if target_col is None:
        raise ValueError("Không tìm thấy cột Yield trong dataset.")

    # Tìm các cột quan trọng khác
    year_col = find_first_existing(df.columns, ["year"])
    area_col = find_first_existing(df.columns, ["area", "country", "state", "region", "district"])
    crop_col = find_first_existing(df.columns, ["item", "crop", "crop_name", "item_name"])

    # Làm sạch cột số nếu cần
    for col in df.columns:
        if col in [area_col, crop_col]:
            continue
        if df[col].dtype == object:
            cleaned = clean_numeric_series(df[col])
            if cleaned.notna().mean() > 0.7:
                df[col] = cleaned

    if year_col:
        df[year_col] = clean_numeric_series(df[year_col]).astype("Int64")

    df[target_col] = clean_numeric_series(df[target_col])
    df = df[df[target_col].notna()].copy()
    df = df[df[target_col] > 0].copy()

    # Loại bỏ outlier bằng IQR
    q1 = df[target_col].quantile(0.25)
    q3 = df[target_col].quantile(0.75)
    iqr = q3 - q1
    lo = max(0, q1 - 3 * iqr)
    hi = q3 + 3 * iqr
    df = df[(df[target_col] >= lo) & (df[target_col] <= hi)].copy()

    df = df.drop_duplicates().reset_index(drop=True)

    meta = {
        "path": str(path),
        "target_col": target_col,
        "year_col": year_col,
        "area_col": area_col,
        "crop_col": crop_col,
    }
    return df, meta


def build_feature_lists(df, meta):
    """Xây dựng danh sách cột số và cột phân loại."""
    target_col = meta["target_col"]
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = [c for c in df.columns if c not in num_cols and c != target_col]
    num_cols = [c for c in num_cols if c != target_col]
    return num_cols, cat_cols


def make_preprocessor(num_cols, cat_cols, scale_numeric=True):
    """Tạo preprocessor cho pipeline: impute, scale cho số, encode cho phân loại."""
    num_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        num_steps.append(("scaler", StandardScaler()))
    num_pipe = Pipeline(num_steps)

    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    return ColumnTransformer([
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols)
    ])


def compute_rmse(y_true, y_pred):
    """Tính Root Mean Squared Error."""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def get_feature_names(preprocessor, num_cols, cat_cols):
    """Lấy tên các feature sau preprocessing."""
    try:
        return list(preprocessor.get_feature_names_out())
    except Exception:
        names = []
        names.extend([f"num__{c}" for c in num_cols])
        if cat_cols:
            try:
                oh = preprocessor.named_transformers_["cat"].named_steps["onehot"]
                cat_names = oh.get_feature_names_out(cat_cols)
                names.extend(list(cat_names))
            except Exception:
                names.extend([f"cat__{c}" for c in cat_cols])
        return names


# ---------- Biểu đồ EDA ----------
def plot_yield_distribution(df, meta, out_dir):
    """Vẽ phân phối năng suất."""
    target_col = meta["target_col"]
    vals = df[target_col].dropna()

    plt.figure(figsize=(9, 5.5))
    plt.hist(vals, bins=30, edgecolor="black", alpha=0.8)
    plt.axvline(vals.mean(), linestyle="--", linewidth=2, label=f"Mean = {vals.mean():.2f}")
    plt.axvline(vals.median(), linestyle=":", linewidth=2, label=f"Median = {vals.median():.2f}")
    plt.title("Distribution of Crop Yield")
    plt.xlabel("Yield")
    plt.ylabel("Frequency")
    plt.legend()
    save_fig(Path(out_dir) / "yield_distribution.png")


def plot_top_categories(df, meta, out_dir):
    """Vẽ top loại cây trồng và khu vực theo năng suất trung bình."""
    for key, title in [("crop_col", "Top Crops by Mean Yield"), ("area_col", "Top Areas by Mean Yield")]:
        col = meta.get(key)
        target_col = meta["target_col"]
        if col and col in df.columns:
            plot_df = (
                df.groupby(col)[target_col]
                .mean()
                .sort_values(ascending=False)
                .head(12)
                .sort_values()
            )
            plt.figure(figsize=(10, 6))
            plt.barh(plot_df.index.astype(str), plot_df.values)
            plt.title(title)
            plt.xlabel("Average Yield")
            plt.ylabel(col)
            save_fig(Path(out_dir) / f"top_{col}_mean_yield.png")


def plot_yearly_trend(df, meta, out_dir):
    """Vẽ xu hướng năng suất theo năm."""
    year_col = meta["year_col"]
    target_col = meta["target_col"]
    if not year_col or year_col not in df.columns:
        return

    yearly = df.groupby(year_col)[target_col].agg(["mean", "median"]).reset_index()
    plt.figure(figsize=(10, 5.5))
    plt.plot(yearly[year_col], yearly["mean"], marker="o", linewidth=2.2, label="Mean yield")
    plt.plot(yearly[year_col], yearly["median"], marker="s", linewidth=1.8, label="Median yield")
    plt.title("Yield Trend by Year")
    plt.xlabel("Year")
    plt.ylabel("Yield")
    plt.legend()
    save_fig(Path(out_dir) / "yield_trend_by_year.png")


# ---------- Mô hình hóa ----------
def regression_experiment(df, meta, out_dir):
    """Thực hiện thí nghiệm hồi quy với nhiều mô hình."""
    target_col = meta["target_col"]
    year_col = meta["year_col"]
    num_cols, cat_cols = build_feature_lists(df, meta)

    X = df[num_cols + cat_cols].copy()
    y = df[target_col].copy()

    # Chia train/test theo thời gian nếu có cột năm
    if year_col and df[year_col].notna().sum() > 0:
        unique_years = sorted(df[year_col].dropna().unique().tolist())
        split_idx = max(1, int(len(unique_years) * 0.8))
        cutoff_year = unique_years[split_idx - 1]
        train_mask = df[year_col] <= cutoff_year
        test_mask = df[year_col] > cutoff_year

        if test_mask.sum() == 0 or train_mask.sum() == 0:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            split_note = "Fallback random split do không đủ năm cho time split"
        else:
            X_train, X_test = X[train_mask], X[test_mask]
            y_train, y_test = y[train_mask], y[test_mask]
            split_note = f"Time split theo năm: train <= {cutoff_year}, test > {cutoff_year}"
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        split_note = "Random split do dataset không có cột năm"

    preprocess = make_preprocessor(num_cols, cat_cols, scale_numeric=True)

    # Các mô hình hồi quy
    models = {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "RandomForest": RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1),
    }
    if HAS_XGB:
        models["XGBRegressor"] = XGBRegressor(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=8,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            random_state=42,
            objective="reg:squarederror",
        )

    rows = []
    best_name, best_rmse, best_pipeline = None, float("inf"), None

    for name, model in models.items():
        pipe = Pipeline([("prep", preprocess), ("model", model)])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)

        mae = mean_absolute_error(y_test, pred)
        rmse = compute_rmse(y_test, pred)
        r2 = r2_score(y_test, pred)
        rows.append([name, mae, rmse, r2])

        if rmse < best_rmse:
            best_name, best_rmse, best_pipeline = name, rmse, pipe

    result = pd.DataFrame(rows, columns=["model", "MAE", "RMSE", "R2"]).sort_values("RMSE")
    result.to_csv(Path(out_dir) / "regression_results.csv", index=False)

    # Biểu đồ so sánh mô hình
    plot_df = result.copy().sort_values("RMSE", ascending=False)
    plt.figure(figsize=(9, 5.5))
    bars = plt.barh(plot_df["model"], plot_df["RMSE"], alpha=0.9)
    for bar, val in zip(bars, plot_df["RMSE"]):
        plt.text(val, bar.get_y() + bar.get_height()/2, f" {val:.2f}", va="center")
    plt.title("Regression Model Comparison by RMSE")
    plt.xlabel("RMSE (lower is better)")
    plt.ylabel("Model")
    save_fig(Path(out_dir) / "regression_model_comparison.png")

    # Actual vs Predicted
    best_pred = best_pipeline.predict(X_test)
    plt.figure(figsize=(7, 7))
    plt.scatter(y_test, best_pred, alpha=0.65)
    mn = float(min(y_test.min(), best_pred.min()))
    mx = float(max(y_test.max(), best_pred.max()))
    plt.plot([mn, mx], [mn, mx], linestyle="--", linewidth=2)
    plt.title(f"Actual vs Predicted Yield ({best_name})")
    plt.xlabel("Actual Yield")
    plt.ylabel("Predicted Yield")
    save_fig(Path(out_dir) / "actual_vs_predicted.png")

    # Residual plot
    residuals = y_test - best_pred
    plt.figure(figsize=(8.5, 5.5))
    plt.scatter(best_pred, residuals, alpha=0.6)
    plt.axhline(0, linestyle="--", linewidth=2)
    plt.title(f"Residual Plot ({best_name})")
    plt.xlabel("Predicted Yield")
    plt.ylabel("Residual (Actual - Predicted)")
    save_fig(Path(out_dir) / "residual_plot.png")

    # Feature importance cho mô hình cây
    try:
        model = best_pipeline.named_steps["model"]
        prep = best_pipeline.named_steps["prep"]
        if hasattr(model, "feature_importances_"):
            names = get_feature_names(prep, num_cols, cat_cols)
            imp = pd.DataFrame({
                "feature": names,
                "importance": model.feature_importances_,
            }).sort_values("importance", ascending=False).head(15)
            imp.to_csv(Path(out_dir) / "feature_importance.csv", index=False)

            plt.figure(figsize=(10, 6.5))
            plt.barh(imp["feature"][::-1], imp["importance"][::-1])
            plt.title(f"Top 15 Feature Importances ({best_name})")
            plt.xlabel("Importance")
            plt.ylabel("Feature")
            save_fig(Path(out_dir) / "feature_importance.png")
    except Exception:
        pass

    with open(Path(out_dir) / "regression_notes.txt", "w", encoding="utf-8") as f:
        f.write(split_note + "\n")
        f.write(f"Best model: {best_name}\n")

    return result, split_note


def classification_experiment(df, meta, out_dir):
    """Thực hiện thí nghiệm phân loại năng suất thành 3 lớp."""
    target_col = meta["target_col"]
    year_col = meta["year_col"]
    area_col = meta["area_col"]
    num_cols, cat_cols = build_feature_lists(df, meta)

    cls_df = df.copy()
    # Chia năng suất thành 3 lớp: low, medium, high
    cls_df["yield_class"] = pd.qcut(cls_df[target_col], q=3, labels=["low", "medium", "high"], duplicates="drop")
    cls_df = cls_df[cls_df["yield_class"].notna()].copy()

    X = cls_df[num_cols + cat_cols]
    y = cls_df["yield_class"].astype(str)

    # Chia train/test
    if year_col and cls_df[year_col].notna().sum() > 0:
        unique_years = sorted(cls_df[year_col].dropna().unique().tolist())
        split_idx = max(1, int(len(unique_years) * 0.8))
        cutoff_year = unique_years[split_idx - 1]
        train_mask = cls_df[year_col] <= cutoff_year
        test_mask = cls_df[year_col] > cutoff_year
        if test_mask.sum() == 0 or train_mask.sum() == 0:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
        else:
            X_train, X_test = X[train_mask], X[test_mask]
            y_train, y_test = y[train_mask], y[test_mask]
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    preprocess = make_preprocessor(num_cols, cat_cols, scale_numeric=True)

    rows = []
    best_pred = None
    best_name = None
    best_f1 = -1

    # Mô hình RandomForest
    rf_pipe = Pipeline([
        ("prep", preprocess),
        ("model", RandomForestClassifier(n_estimators=250, random_state=42, n_jobs=-1))
    ])
    rf_pipe.fit(X_train, y_train)
    rf_pred = rf_pipe.predict(X_test)
    rf_f1 = f1_score(y_test, rf_pred, average="macro")
    rows.append(["Baseline_RF_Classifier", rf_f1])
    with open(Path(out_dir) / "classification_report_rf.txt", "w", encoding="utf-8") as f:
        f.write(classification_report(y_test, rf_pred))
    best_pred, best_name, best_f1 = rf_pred, "Baseline_RF_Classifier", rf_f1

    # Mô hình XGBoost nếu có
    if HAS_XGB:
        le = LabelEncoder()
        y_train_enc = le.fit_transform(y_train)
        xgb_pipe = Pipeline([
            ("prep", preprocess),
            ("model", XGBClassifier(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=7,
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=42,
                eval_metric="mlogloss",
            ))
        ])
        xgb_pipe.fit(X_train, y_train_enc)
        pred_enc = xgb_pipe.predict(X_test)
        xgb_pred = le.inverse_transform(pred_enc)
        xgb_f1 = f1_score(y_test, xgb_pred, average="macro")
        rows.append(["XGBClassifier", xgb_f1])
        with open(Path(out_dir) / "classification_report_xgb.txt", "w", encoding="utf-8") as f:
            f.write(classification_report(y_test, xgb_pred))
        if xgb_f1 > best_f1:
            best_pred, best_name, best_f1 = xgb_pred, "XGBClassifier", xgb_f1

    result = pd.DataFrame(rows, columns=["model", "F1_macro"]).sort_values("F1_macro", ascending=False)
    result.to_csv(Path(out_dir) / "classification_results.csv", index=False)

    # Biểu đồ so sánh
    plot_df = result.copy().sort_values("F1_macro")
    plt.figure(figsize=(8, 5))
    bars = plt.barh(plot_df["model"], plot_df["F1_macro"], alpha=0.9)
    for bar, val in zip(bars, plot_df["F1_macro"]):
        plt.text(val, bar.get_y() + bar.get_height()/2, f" {val:.3f}", va="center")
    plt.title("Classification Model Comparison by F1 Macro")
    plt.xlabel("F1 Macro (higher is better)")
    plt.ylabel("Model")
    save_fig(Path(out_dir) / "classification_model_comparison.png")

    # Ma trận nhầm lẫn
    labels = sorted(pd.Series(y_test).unique().tolist())
    cm = confusion_matrix(y_test, best_pred, labels=labels)
    plt.figure(figsize=(6.5, 5.5))
    plt.imshow(cm, aspect="auto")
    plt.title(f"Confusion Matrix ({best_name})")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.xticks(range(len(labels)), labels)
    plt.yticks(range(len(labels)), labels)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")
    save_fig(Path(out_dir) / "confusion_matrix.png")

    # Phân tích vùng hiếm
    if area_col and area_col in cls_df.columns:
        test_view = X_test.copy()
        test_view["actual"] = list(y_test)
        test_view["pred"] = list(best_pred)
        freq = cls_df[area_col].value_counts()
        rare_regions = freq[freq <= max(3, int(freq.quantile(0.1)))].index
        rare_df = test_view[test_view[area_col].isin(rare_regions)].copy()
        if len(rare_df) > 0:
            rare_f1 = f1_score(rare_df["actual"], rare_df["pred"], average="macro")
            with open(Path(out_dir) / "rare_region_analysis.txt", "w", encoding="utf-8") as f:
                f.write(f"Rare region count: {len(rare_regions)}\n")
                f.write(f"Rare-region F1_macro: {rare_f1:.4f}\n")
                f.write("Một số vùng hiếm:\n")
                f.write("\n".join(map(str, list(rare_regions)[:20])))

    return result


def clustering_experiment(df, meta, out_dir):
    """Thực hiện phân cụm dữ liệu bằng KMeans."""
    target_col = meta["target_col"]
    num_cols, cat_cols = build_feature_lists(df, meta)
    feature_cols = num_cols + cat_cols

    X = df[feature_cols].copy()
    preprocess = make_preprocessor(num_cols, cat_cols, scale_numeric=True)
    Xt = preprocess.fit_transform(X)
    

    # Tìm số cụm tối ưu bằng silhouette score
    best_k, best_score = None, -1
    for k in range(2, 7):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(Xt)
        score = silhouette_score(Xt, labels)
        if score > best_score:
            best_k, best_score = k, score

    km = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    labels = km.fit_predict(Xt)

    cluster_df = df.copy()
    cluster_df["cluster"] = labels
    cluster_df.to_csv(Path(out_dir) / "clustered_data.csv", index=False)

    # Hồ sơ cụm theo năng suất
    profile = cluster_df.groupby("cluster")[target_col].agg(["count", "mean", "median", "min", "max"]).reset_index()
    profile.to_csv(Path(out_dir) / "cluster_profile_yield.csv", index=False)

    # Hồ sơ theo cột phân loại
    profile_text = []
    for c in cat_cols[:5]:
        mode_by_cluster = cluster_df.groupby("cluster")[c].agg(lambda s: s.mode().iloc[0] if not s.mode().empty else np.nan)
        profile_text.append(f"\nMode of {c}:\n{mode_by_cluster.to_string()}\n")

    with open(Path(out_dir) / "cluster_profile_notes.txt", "w", encoding="utf-8") as f:
        f.write(f"Best k: {best_k}, silhouette: {best_score:.4f}\n")
        f.write("\n".join(profile_text))

    # Biểu đồ năng suất trung bình theo cụm
    plt.figure(figsize=(8.5, 5.5))
    bars = plt.bar(profile["cluster"].astype(str), profile["mean"])
    for bar, val in zip(bars, profile["mean"]):
        plt.text(bar.get_x() + bar.get_width()/2, val, f"{val:.1f}", ha="center", va="bottom")
    plt.title("Average Yield by Cluster")
    plt.xlabel("Cluster")
    plt.ylabel("Mean Yield")
    save_fig(Path(out_dir) / "cluster_mean_yield.png")

    # Biểu đồ PCA của cụm
    Xt_dense = Xt.toarray() if hasattr(Xt, "toarray") else Xt
    pca = PCA(n_components=2, random_state=42)
    p2 = pca.fit_transform(Xt_dense)
    plt.figure(figsize=(8, 6))
    plt.scatter(p2[:, 0], p2[:, 1], c=labels, alpha=0.75)
    plt.title(f"KMeans Clusters on PCA Space (k={best_k})")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    save_fig(Path(out_dir) / "clusters_pca.png")

    return profile, best_k, best_score


def association_rules_experiment(df, meta, out_dir):
    """Khám phá luật kết hợp giữa các yếu tố và năng suất cao."""
    target_col = meta["target_col"]
    crop_col = meta["crop_col"]

    if not HAS_MLXTEND:
        with open(Path(out_dir) / "association_rules_notes.txt", "w", encoding="utf-8") as f:
            f.write("Chưa cài mlxtend. Chạy: pip install mlxtend\n")
        return None, None

    assoc_df = df.copy()
    # Tạo cờ năng suất cao (trên quantile 75%)
    assoc_df["high_yield"] = np.where(
        assoc_df[target_col] >= assoc_df[target_col].quantile(0.75),
        "high_yield",
        "not_high_yield"
    )

    # Chọn cột số và phân loại để phân tích
    candidate_num = [c for c in assoc_df.select_dtypes(include=np.number).columns if c != target_col]
    candidate_num = [c for c in candidate_num if assoc_df[c].nunique() > 5][:6]
    candidate_cat = [c for c in assoc_df.select_dtypes(exclude=np.number).columns][:4]

    basket = pd.DataFrame(index=assoc_df.index)

    # Phân loại cột số thành bins
    for c in candidate_num:
        try:
            binned = pd.qcut(assoc_df[c], q=3, duplicates="drop")
            basket[c] = c + "=" + binned.astype(str)
        except Exception:
            pass

    # Thêm cột phân loại
    for c in candidate_cat:
        basket[c] = c + "=" + assoc_df[c].astype(str)

    if crop_col and crop_col not in candidate_cat and crop_col in assoc_df.columns:
        basket[crop_col] = crop_col + "=" + assoc_df[crop_col].astype(str)

    basket["yield_flag"] = assoc_df["high_yield"]

    # Tạo danh sách items cho mỗi hàng
    items = basket.apply(lambda row: [v for v in row.values if pd.notna(v)], axis=1)
    all_items = sorted({item for row in items for item in row})

    # Chuyển thành one-hot encoding
    onehot = pd.DataFrame(False, index=basket.index, columns=all_items, dtype=bool)
    for i, row in items.items():
        onehot.loc[i, row] = True

    # Áp dụng Apriori
    freq_ap = apriori(onehot, min_support=0.05, use_colnames=True)
    rules_ap = association_rules(freq_ap, metric="confidence", min_threshold=0.6)
    rules_ap = rules_ap.sort_values(["lift", "confidence", "support"], ascending=False)
    rules_ap = rules_ap[rules_ap["consequents"].astype(str).str.contains("high_yield", regex=False)].copy()

    # Áp dụng FP-Growth
    freq_fp = fpgrowth(onehot, min_support=0.05, use_colnames=True)
    rules_fp = association_rules(freq_fp, metric="confidence", min_threshold=0.6)
    rules_fp = rules_fp.sort_values(["lift", "confidence", "support"], ascending=False)
    rules_fp = rules_fp[rules_fp["consequents"].astype(str).str.contains("high_yield", regex=False)].copy()

    def stringify_rules(rdf: pd.DataFrame) -> pd.DataFrame:
        """Chuyển luật thành chuỗi dễ đọc."""
        out = rdf.copy()
        if out.empty:
            return out
        out["antecedents"] = out["antecedents"].apply(lambda s: ", ".join(sorted(list(s))))
        out["consequents"] = out["consequents"].apply(lambda s: ", ".join(sorted(list(s))))
        return out[["antecedents", "consequents", "support", "confidence", "lift"]].head(15)

    top_ap = stringify_rules(rules_ap)
    top_fp = stringify_rules(rules_fp)
    top_ap.to_csv(Path(out_dir) / "top_rules_apriori.csv", index=False)
    top_fp.to_csv(Path(out_dir) / "top_rules_fpgrowth.csv", index=False)

    # Biểu đồ lift của top luật
    base = top_fp if len(top_fp) else top_ap
    if len(base) > 0:
        plot_df = base.head(10).copy().iloc[::-1]
        short_labels = [s[:65] + ("..." if len(s) > 65 else "") for s in plot_df["antecedents"]]
        plt.figure(figsize=(11, 6.5))
        plt.barh(short_labels, plot_df["lift"])
        plt.title("Top Association Rules Leading to High Yield")
        plt.xlabel("Lift")
        plt.ylabel("Antecedent Conditions")
        save_fig(Path(out_dir) / "top_rules_lift.png")

    # Tạo khuyến nghị
    with open(Path(out_dir) / "recommendations.txt", "w", encoding="utf-8") as f:
        f.write("Khuyến nghị canh tác ở mức mô tả, suy ra từ các luật gắn với high_yield:\n\n")
        base = top_fp if len(top_fp) else top_ap
        if len(base) == 0:
            f.write("Không tìm được luật đủ mạnh với ngưỡng hiện tại. Hạ min_support hoặc min_confidence.\n")
        else:
            for _, row in base.head(10).iterrows():
                f.write(
                    f"- Nếu điều kiện gồm [{row['antecedents']}] thì xác suất đi kèm năng suất cao tăng lên "
                    f"(confidence={row['confidence']:.2f}, lift={row['lift']:.2f}).\n"
                )

    return top_ap, top_fp


def create_requirements(out_dir):
    """Tạo file requirements.txt với các thư viện cần thiết."""
    req = """pandas>=2.2.0
numpy>=1.26.0
scikit-learn>=1.4.0
matplotlib>=3.8.0
xgboost>=2.0.0
mlxtend>=0.23.0
"""
    Path(out_dir, "requirements.txt").write_text(req, encoding="utf-8")


def create_download_guide(out_dir):
    """Tạo hướng dẫn tải dataset từ Kaggle."""
    text = """# Tải dataset Kaggle

## Cách 1: Kaggle API
1. Cài Kaggle API:
   pip install kaggle
2. Vào Kaggle > Account > Create New Token để lấy `kaggle.json`.
3. Đặt `kaggle.json` vào:
   - Windows: %USERPROFILE%\\.kaggle\\kaggle.json
   - Linux/macOS: ~/.kaggle/kaggle.json
4. Chạy lệnh:
   kaggle datasets download -d patelris/crop-yield-prediction-dataset -p data --unzip

## Cách 2: Tải thủ công
- Mở trang dataset:
  https://www.kaggle.com/datasets/patelris/crop-yield-prediction-dataset/data
- Tải file CSV rồi bỏ vào thư mục `data/` cạnh file Python.

## File thường gặp
- yield_df.csv
- yield.csv
"""
    Path(out_dir, "DOWNLOAD_DATASET.txt").write_text(text, encoding="utf-8")


def main():
    """Hàm chính chạy toàn bộ pipeline."""
    set_plot_style()  # Thiết lập kiểu dáng biểu đồ
    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)  # Tạo thư mục outputs nếu chưa có

    # Tải và làm sạch dữ liệu
    df, meta = load_and_clean("data")

    # Tạo tổng quan dữ liệu
    overview = pd.DataFrame({
        "column": df.columns,
        "dtype": [str(df[c].dtype) for c in df.columns],
        "missing": [int(df[c].isna().sum()) for c in df.columns],
        "nunique": [int(df[c].nunique(dropna=True)) for c in df.columns],
    })
    overview.to_csv(out_dir / "data_overview.csv", index=False)

    # Vẽ biểu đồ EDA
    plot_yield_distribution(df, meta, out_dir)
    plot_top_categories(df, meta, out_dir)
    plot_yearly_trend(df, meta, out_dir)

    # Chạy các thí nghiệm mô hình
    reg_res, split_note = regression_experiment(df, meta, out_dir)
    cls_res = classification_experiment(df, meta, out_dir)
    cluster_profile, best_k, best_score = clustering_experiment(df, meta, out_dir)
    top_ap, top_fp = association_rules_experiment(df, meta, out_dir)

    # Tạo tóm tắt kết quả
    summary_lines = [
        f"Dataset: {meta['path']}",
        f"Số dòng sau làm sạch: {len(df)}",
        f"Target column: {meta['target_col']}",
        f"Year column: {meta['year_col']}",
        f"Area column: {meta['area_col']}",
        f"Crop column: {meta['crop_col']}",
        "",
        "== Regression ==",
        reg_res.to_string(index=False),
        "",
        f"Split note: {split_note}",
        "",
        "== Classification ==",
        cls_res.to_string(index=False),
        "",
        "== Clustering ==",
        f"Best k: {best_k}, silhouette: {best_score:.4f}",
        cluster_profile.to_string(index=False),
    ]

    if top_fp is not None:
        summary_lines += ["", "== Top FP-Growth Rules ==", top_fp.head(10).to_string(index=False)]

    Path(out_dir / "summary.txt").write_text("\n".join(summary_lines), encoding="utf-8")
    create_requirements(".")  # Tạo requirements.txt
    create_download_guide(".")  # Tạo hướng dẫn tải dataset

    print("Done. Xem thư mục outputs/ để lấy biểu đồ đẹp hơn và file tổng hợp.")


if __name__ == "__main__":
    main()
