"""Extract features from a dataframe

Note
----
These features are intendended to be a demonstration of feature extraction.
The model is intended to learn it's own features. If we were training a model
solely on these features, then joint time-frequency features would likely
perform better.
"""
import pandas as pd
import pywt


def extract_basic_features_from_series(s: pd.Series):
    """Extracts basic statistical features"""

    # Exclude the target column
    s = s[s.index.drop("target")]
    s = s.astype(float)

    features = pywt.dwt(s, "morl")
    print(features)
    return features


def extract_features_from_dataframe(df: pd.DataFrame):
    """Extract statistical features for the model"""

    return pd.concat(
        [
            df.apply(
                extract_basic_features_from_series,
                axis=1,
                result_type="expand",
            ),
            df.target,
        ],
        axis=1,
    )
