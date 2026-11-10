# 😀 Emoji Sentiment Tagging

## Overview
Default regex cleaning scripts frequently strip emoji characters as punctuation/noise. In sentiment and subjective classification, emojis are high-value discriminant features.

## Rationale
Tagging emojis as explicit tokens (`[EMOJI_POS]`, `[EMOJI_NEG]`) allows TF-IDF and Transformer backbones to treat them as distinct, learnable tokens without character-encoding corruption.
