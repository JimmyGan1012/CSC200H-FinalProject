# CSC200H - k>1 Query Model For Data Tailoring

## Authors
- Jiamin Gan (jgan4@u.rochester.edu)
- Enting Zhou (ezhou12@u.rochester.edu)
- Shengyi Jia (sjia6@u.rochester.edu)


## Project Architecture
- Simulator.py: Implements the simulator and corresponding data structures to hold information and conduct query.
- Algorithms.py: Implements baseline and proposed algorithm.
- Experiments_known.py: Runs experiments for known distribution. Results saved to ./Result directory in json format.
- Experiments_ucb.py: Runs experiments for unknown distribution. Results saved to ./Result directory in json format.
- Creat_plot.py: Generates the plot used in report.

## Dependencies
```
python3 -m pip install numpy pandas seaborn matplotlib
```

## To Run
```
python3 Experiments_known.py
python3 Experiments_ucb.py
python3 ECreat_plot.py
```