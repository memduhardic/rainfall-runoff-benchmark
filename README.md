# Rainfall-Runoff Benchmark

Compares two conceptual models (GR4J, HBV) and three machine-learning models (Random Forest, ANN, LSTM) on the same catchment, with the same inputs and the same test period.

![Results](docs/comparison.png)

All models use only precipitation and PET (plus temperature for HBV's snow routine). The data are split chronologically into train / validation / test, and the last period is never seen by any model. GR4J and HBV are calibrated with SCE-UA.

## Usage

```bash
pip install -r requirements.txt
python examples/fetch_example_data.py
python run.py
```

Options: `--models gr4j hbv`, `--objective KGE`, `--input my_basin.csv --area 692`. See `python run.py --help`.

Input is a daily CSV with `Date, P, PET, T, Q` (mm/day, °C, m³/s). Results are saved in `output/`: metrics, simulated series and figures. `output/simulations.csv` can be opened in [Calibration Studio](https://github.com/memduhardic/calibration-studio).

## Credits

- GR4J code adapted from [amacd31/gr4j](https://github.com/amacd31/gr4j) by Andrew MacDonald (BSD 3-Clause).
- HBV structure based on [mxgiuliani00/hbv](https://github.com/mxgiuliani00/hbv) by Giuliani, Kollat, Herman et al.
- Machine learning with [scikit-learn](https://scikit-learn.org) and [PyTorch](https://pytorch.org); calibration with [SPOTPY](https://github.com/thouska/spotpy).
- Example data from the [airGR](https://CRAN.R-project.org/package=airGR) R package (INRAE), downloaded by the fetch script.

## References

- Bergström, S. (1995). The HBV model. In: Singh, V.P. (ed.), *Computer Models of Watershed Hydrology*, 443–476.
- Herman, J.D., Reed, P.M., Wagener, T. (2013). Time-varying sensitivity analysis clarifies the effects of watershed model formulation on model behavior. *Water Resources Research* 49, 1400–1414.
- Perrin, C., Michel, C., Andréassian, V. (2003). Improvement of a parsimonious model for streamflow simulation. *Journal of Hydrology* 279, 275–289.

## License

MIT
