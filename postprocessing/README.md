# Post Processing Scripts 

All scripts are intended to be ran in this directory. 

Requires `yt project`, `numpy`, and `python 3.0`. 

To post processes a run do: \
`$ python3 [post processing script] [simulation relative path] [starting snapshot] [ending snapshot] [step]`

For example, to extract Pop II data from the `CC-Fiducial` simulation run: \
`$ python3 bsc_pipeline.py ../../sim_data/cluster_evolution/CC-Fiducial 405 432 1`

Note, the path to the simulation needs to be `relative`. 

By default, the post processed data is placed in a master directory called `container_tiramisu`, in the same directory as the cloned __tiramisu__ container. If this contaniner is not present, the script will make one and place the post processed data there. 

Below is an outline of the directory structure, with relevant (sub)directories highlighted with `*`
```
├── * tiramisu
│   ├── garcia+23
│   ├── gas
│   ├── movie_wzrd
│   ├── *postprocessing
│   ├── src
|   ├── starburst
|   ├── tools
|   ├── __init__.py
|   ├── sim_scraper.py
|   ├── README.md  
└── * container_tiramisu
    ├── * post_processed
    │   ├── bsc_catalogues
    │   ├── dm_hop
    │   └── * pop2
    │       ├── CC-Fiducial
    │       │   ├── pop2-00304-493_69-myr-z-9_952.txt
    │       │   ├── ...                
    │       └── ...   
    └── ...
```