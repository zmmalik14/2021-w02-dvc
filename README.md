# 2021-w01-dvc
A starter repository for AI Club at NC State's first workshop of 2021 which covers Data Version Control.

Adapted from:
https://www.analyticsvidhya.com/blog/2021/06/mlops-tracking-ml-experiments-with-data-version-control/

Dataset:
- [Source](https://www.kaggle.com/rashikrahmanpritom/heart-attack-analysis-prediction-dataset)
- [Download](https://drive.google.com/file/d/1eM8eCMP1JszNX24Kewx7xG8pAjEj4fV7/view?usp=sharing)

### Workshop Steps
1. Install prerequisites
   1. Python
   2. Git/GitHub Desktop
   3. IDE/Text Editor
2. Click "Use this template" to create a personal copy of the workshop repository
3. Clone your personal workshop repository onto your local machine
   1. Open terminal/command prompt in the directory where you want to clone the repository
   2. `git clone [repository url]`
4. Activate virtual environment from terminal/command prompt
   ```
   # Move into repository directory
   cd [repository directory]
   
   # Create a new virtual environment
   python3 -m venv .venv
   ```
   1. Windows: `.venv\Scripts\activate.bat`
   2. Unix/MacOS: `source .venv/bin/activate`
5. Install required Python packages
6. Place data file in `./data/external/` directory
7. Setup DVC
   ```
   # Initialize DVC for project and add "external" data
   dvc init
   dvc add data/external/heart.csv
   ```
8. View directed acyclic graph of operations with `dvc dag`
9. View generated plots in `plots.html`
10. View model metrics with `dvc metrics show`
11. Show metric plots with `dvc plots show cm.csv --template confusion -x output -y Predicted`
12. Commit changes with Git
    ```
    git add .
    git commit -m "Initialize DVC and run pipeline"
    ```
13. Change parameters in `params.yaml`
15. Run pipeline again with `dvc repro`
16. Run `dvc metrics diff`
