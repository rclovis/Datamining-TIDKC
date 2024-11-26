# Implementation of TIDKC

This is a group project implementation of _Distribution-Based Trajectory Clustering_ paper for CSIT5210 Data Mining and Knowledge Discovery course.

**Paper citation**

Z. J. Wang, Y. Zhu and K. M. Ting, "Distribution-Based Trajectory Clustering," 2023 IEEE International Conference on Data Mining (ICDM), Shanghai, China, 2023, pp. 1379-1384, doi: 10.1109/ICDM58522.2023.00178.

## How to run

Create a python virtual environment:

```bash
python -m venv <venv>
```

### Activation

And activate it depending of your platform

**Using Bash/Zsh**

```bash
source <venv>/bin/activate
```

**Using Windows CMD**

```cmd
<venv>\Scripts\activate.bat
```

**Using Powershell**

```powershell
<venv>\Scripts\Activate.ps1
```

### Dependencies installation

```bash
pip install -r requirements.txt
```

### Run

While in the activated virtual environment

```bash
python main.py
```

## How to use

The project's results are available using the `TrajClustering` class. It allows running the different trajectory clustering algorithms and distance measures on the provided datasets, and plotting the results.

**Available datasets**

_string identifiers used by the program and their ground truth_
|String identifer| # of clusters| # of trajectories |
| -------------- | -----------: | ----------------: |
| `"CASIA"` | 15 | 1500 |
| `"cross"` | 19 | 1900|
| `"cyclists"` | 3 | 494 |
| `"geolife"` | 12 | 9192 |
| `"pedes3"` | 3 | 610 |
| `"pedes4"` | 4 | 710 |
| `"TRAFFIC"` | 11 | 300 |

```python
# Example
tc = TrajClustering()
# ...
tc.load_dataset("TRAFFIC")
```

**Available distance measures**

_string identifiers used by the program_

- `"IDK2`
- `"IDK`
- `"Hausdorff`
- `"DTW`
- `"EMD`
- `"GDK`

```python
# Example
tc = TrajClustering()
# ...
tc.run_distance("IDK")
```

**Available trajectory clustering algorithms**

_string identifiers used by the program_

- `"KMeans"`
- `"Spectral"`
- `"TIDKC"`

> [!important] Important
> "TIDKC" implementation is independant of the set distance measure,
> it will always result in using first and second level IDK.

```python
# Example
tc = TrajClustering()
# ... set a distance measure
tc.run_clustering("Spectral", 10)
```

```python
# Example
tc = TrajClustering()
# ... no need for setting a distance measure
tc.run_clustering("TIDKC", 7)
```

> [!note] Note
> `run_clustering` method takes 2 parameters:
>
> - the string identifier,
> - and the number of clusters to find.

**Plot MDS representation**

After setting a metric, you can plot its MDS.

```python
# Example
tc = TrajClustering()
# ... run distance measure
tc.plot_mds()
```

**Plot trajectory clustering**

After running a clustering algorithm you can plot its results.

```python
# Example
tc = TrajClustering()
# ... run trajectory clustering
tc.plot_clusters()
```

### Examples

Those are example demonstrating full usage of the `TrajClustering` class.

```python
"""
Create a class instance
Load the "TRAFFIC" dataset
Uses the "IDK" distance measure
Plot the "IDK" results using MDS
Run the "Spectral" clustering algorithm for 10 clusters
Plot the clustering results
"""
tc = TrajClustering()
tc.load_dataset("TRAFFIC")
tc.run_distance("IDK")
tc.plot_mds()
tc.run_clustering("Spectral", 11)
tc.plot_clusters()
```

```python
"""
Create a class instance
Load the "pedes3" dataset
Run the "TIDKC" clustering algorithm for 3 clusters
Plot the clustering results
"""
tc = TrajClustering()
tc.load_dataset("pedes3")
tc.run_clustering("TIDKC", 3)
tc.plot_clusters()
```

## Project structure

The following hierarchy hint the purpose of each core file of the project.

```
Datamining-TIDKC
├── datasets/                   # Folder containing the used datasets
├── t2vec/                      # t2vec implementation
├── utils/                      ## Utilities for:
│   ├── dataloader.py           #  -  loading datasets
│   ├── distance_measure.py     #  -  using Hausdorff, DTW, EMD and GDK
│   ├── eval_clusters.py        #  -  calculating ARI and NMI metrics
│   └── visualizer.py           #  -  ploting trajectories
├── cyclistData.py              # Code preparing the Cyclist
│                                 dataset for consumption
├── find_mode.py                # FindMode step implementation
├── IDK.py                      # IDK implementation
├── local_contrast.py           # Local-Constrast implementation
├── tidkc.py                    # TIDKC implementation
├── TrajClustering.py           # Class handling trajectory clustering
└── main.py                     # Main file
```

## Members

Group #3

- RABOT Clovis
- GONZALES Erwan
- LIU Runrong
- SMITH Caroline
- ZHANG Zexuan
- ARSHAD Muhammad Hassan

Project URL: https://github.com/rclovis/Datamining-TIDKC
