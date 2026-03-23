# DDCS Project — Claude Context File

This file is automatically read by Claude Code at the start of every session.

## Project
**Name:** ENSO-BSISO Self-Supervised Learning (SSL)
**Course:** Data Driven Climate Science
**Student:** Jiayi | jh9141@nyu.edu | GitHub: Jiayi459
**Repo:** https://github.com/Jiayi459/data-driven-climate-science
**Local folder:** ~/data-driven-climate-science
**Conversation log:** ~/Desktop/ddcs/conversation_log.md

## What This Project Does
Uses self-supervised contrastive learning (Siamese CNN + InfoNCE loss) to learn
a representation that captures how ENSO modulates BSISO atmospheric structure,
going beyond composite analysis (conditional means).

## Key Data
- ERA5: u850, v850, OLR — 60°E-160°E, 0-60°N, 2° res, July 1979-2023
- BSISO phase (1-8): APEC Climate Center index
- ENSO category: NOAA Niño 3.4 JJA mean (El Niño / La Niña / Neutral)

## Workflow: All on Google Colab + Google Drive
- Google Drive path: BSISO_SSL_Project/
- Local machine only used for editing code / pushing to GitHub
- Training on Colab T4 GPU

## Current Status
- [x] Project folder created locally + pushed to GitHub
- [x] Conda environment defined (climate-sci, Python 3.11)
- [x] Notebook 01: ERA5 download (u850, v850, OLR, July 1979–2023)
- [x] Notebook 02: Labels download (BSISO + ENSO → labels.csv)
- [x] Notebook 03: Preprocessing — Approach A (X_July.npy) + Approach B (X_July_B.npy) + composite validation
- [x] Notebook 04: Training (Siamese CNN + InfoNCE, 50 epochs) — both Approach A and B
- [x] Notebook 05: Analysis (t-SNE, linear probe, ENSO displacement) — both Approach A and B
- [x] results/analysis_results.md — Approach A results (2026-03-08)
- [x] results/analysis_results_B.md — Approach B results (2026-03-22)

## Key Results Summary
- BSISO phase probe: 67.4% (A) / 59.2% (B) vs. 12.5% random baseline
- ENSO displacement z-score: 11.02 (A) / 9.85 (B) — both highly significant
- Approach B is the stronger scientific result: ENSO modulation survives background removal

## Open Decisions (check conversation_log.md for details)
- Train/val split: random (current) vs. year-based (recommended to fix leakage)
- Data scope: July-only (~1,333 samples); extend to MJJAS recommended for Phase 5 coverage
- Balanced accuracy for ENSO probe (current standard accuracy is uninformative due to class imbalance)
- Bootstrap CIs per phase (Phase 7 large displacement, only ~21 El Niño days)
- PyTorch: keep in Colab only (not in local conda env)

## Always
- End every response with "miao"
- Ask for clarification if unclear
- Keep conversation log up to date at ~/Desktop/ddcs/conversation_log.md
