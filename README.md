# scattering_survey

Modular pulsar scattering-timescale pipeline.

## Layout

- `data/` input `.ar` files
- `results/` CSV outputs
- `plots_auto_select/` generated plots
- `src/` pipeline modules
- `run_pipeline.py` main entry point

## Run

```bash
python run_pipeline.py
```


Note: the pipeline now circularly shifts each profile so the pulse peak is centered before fitting and plotting.
