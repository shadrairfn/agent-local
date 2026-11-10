# 📚 LSA (Latent Semantic Analysis) Dense Topics

## Overview
Decomposes large term-document matrices into $K$ latent thematic topics.

## Rationale
GBDT models (LightGBM, XGBoost, CatBoost) struggle with high-dimensional ultra-sparse matrices (>20k columns). LSA compresses this into 16-64 dense numeric columns that GBDTs can split on easily alongside metadata features.
