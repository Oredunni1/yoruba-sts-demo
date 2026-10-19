---
title: Yoruba-English Speech Translation
emoji: 🗣️
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
---

# Bilingual Yoruba–English Speech-to-Speech Translation

Fine-tuned Whisper for Yoruba ASR + Whisper-medium for English ASR → NLLB-1.3B → MMS-TTS-Yoruba.

The Yoruba ASR model was fine-tuned on 484 utterances from the project corpus, reducing held-out WER from 0.815 (zero-shot baseline) to 0.591.