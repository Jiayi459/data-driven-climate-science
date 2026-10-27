# Data Driven Climate Science

Course project for Data Driven Climate Science.

## Project Structure

```
data-driven-climate-science/
├── data/           # Raw and processed datasets
├── notebooks/      # Jupyter notebooks for analysis
├── src/            # Source code / scripts
├── results/        # Figures, outputs, and results
├── environment.yml # Conda environment specification
└── README.md
```

## Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/Jiayi459/data-driven-climate-science.git
   cd data-driven-climate-science
   ```

2. Create and activate the conda environment:
   ```bash
   conda env create -f environment.yml
   conda activate climate-sci
   ```

3. Launch Jupyter:
   ```bash
   jupyter lab
   ```

## Toy oscillator validation

The [L1 pendulum experiments](notebooks/toy_pendulum/README.md) use self-contained
Colab notebooks **39–42** for simulation, NSV, InfoNCE/Barlow Twins and held-out
evaluation. Start with `39_pendulum_data.ipynb`; all outputs use a separate
`BSISO_SSL_Project/toy/pendulum/` folder on Google Drive.

## Author

- **Jiayi** — jh9141@nyu.edu
