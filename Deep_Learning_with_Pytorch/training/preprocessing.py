from typing import List, Optional, Tuple, Union
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def preprocessing(
    df: pd.DataFrame,
    label_col: str = 'label_column',
    feature_cols: Optional[List[str]] = None,
    le: Optional[LabelEncoder] = None,
    inference: bool = False
) -> Union[Tuple[pd.DataFrame, LabelEncoder], pd.DataFrame]:
    """
    データ前処理関数
    
    Args:
        df: 入力DataFrame
        label_col: ラベルカラム名
        feature_cols: 使う特徴量カラム名リスト（Noneならlabel_col以外全部）
        le: 学習時にfitしたLabelEncoder（学習時はNone、推論時は渡す）
        inference: Trueならラベル無しで特徴量だけ返す
        
    Returns:
        inference=False and le=None  -> (df_encoded, le)
        inference=False and le!=None -> df_encoded
        inference=True               -> df_features_only
    """
    if feature_cols is None:
        feature_cols = [col for col in df.columns if col != label_col]

    df_features = df[feature_cols].copy()

    if inference:
        return df_features

    if le is None:
        le = LabelEncoder()
        df_features["label"] = le.fit_transform(df[label_col])
        return df_features, le
    else:
        df_features["label"] = le.transform(df[label_col])
        return df_features


def main():
    # ===== Dummy Data =====
    train_df = pd.DataFrame({
        'feature1': [1.0, 2.0, 3.0],
        'feature2': [1.5, 2.5, 3.5],
        'label': ['dog', 'cat', 'bird']
    })

    test_df = pd.DataFrame({
        'feature1': [4.0, 5.0],
        'feature2': [4.5, 5.5],
        'label': ['cat', 'dog']
    })

    infer_df = pd.DataFrame({
        'feature1': [6.0],
        'feature2': [6.5]
    })

    feature_cols = ['feature1', 'feature2']

    # ===== Train =====
    train_df_encoded, le = preprocessing(train_df, label_col='label', feature_cols=feature_cols)

    # Test: trainのshape確認
    assert train_df_encoded.shape == (3, 3), "Train shape mismatch"
    # Test: labelカラムが数値になっているか
    assert train_df_encoded['label'].tolist() == [2, 1, 0], f"Unexpected train labels: {train_df_encoded['label'].tolist()}"
    # Test: leが保持しているクラス名
    assert list(le.classes_) == ['bird', 'cat', 'dog'], f"Unexpected label classes: {le.classes_}"

    print("[OK] Train preprocessing test passed.")

    # ===== Test =====
    test_df_encoded = preprocessing(test_df, label_col='label', feature_cols=feature_cols, le=le)

    # Test: testのshape確認
    assert test_df_encoded.shape == (2, 3), "Test shape mismatch"
    # Test: labelエンコード値確認
    assert test_df_encoded['label'].tolist() == [1, 2], f"Unexpected test labels: {test_df_encoded['label'].tolist()}"

    print("[OK] Test preprocessing test passed.")

    # ===== Inference =====
    infer_df_encoded = preprocessing(infer_df, label_col='label', feature_cols=feature_cols, le=le, inference=True)

    # Test: inferはlabel無しのshape
    assert infer_df_encoded.shape == (1, 2), "Inference shape mismatch"
    assert 'label' not in infer_df_encoded.columns, "Inference dataframe should not have 'label' column"

    print("[OK] Inference preprocessing test passed.")

if __name__ == "__main__":
    main()