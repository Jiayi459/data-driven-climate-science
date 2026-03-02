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
- [x] Notebook 01_era5_download.ipynb created
- [ ] ERA5 data download in progress (user setting up CDS account)
- [ ] BSISO index download
- [ ] NOAA ENSO index download
- [ ] Preprocessing
- [ ] Model training

## Open Decisions (check conversation_log.md for details)
- Train/val split: random vs. year-based?
- Preprocessing: Approach A (raw) or Approach B (detrended)?
- Data scope: July-only confirmed for now, may extend to MJJAS
- PyTorch: add to environment.yml or keep in Colab only?

## Always
- End every response with "miao"
- Ask for clarification if unclear
- Keep conversation log up to date at ~/Desktop/ddcs/conversation_log.md
