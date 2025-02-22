# Disaster Response Pipeline

This project classifies disaster-related messages into various categories, enabling quicker and more targeted responses from relevant agencies.

## Project Components
1. **ETL Pipeline (process_data.py)**
   - Loads the messages and categories datasets.
   - Cleans the data (splits categories, converts to binary, drops duplicates).
   - Stores the clean data into an SQLite database.

2. **Machine Learning Pipeline (train_classifier.py)**
   - Loads data from the SQLite database.
   - Splits data into training and test sets.
   - Builds and tunes a multi-output classifier using GridSearchCV.
   - Outputs model performance metrics (F1-score, Precision, Recall).
   - Exports the final model as a pickle file.

3. **Flask Web App (run.py)**
   - Serves as an interface for entering messages to classify.
   - Displays classification results for each category.
   - Visualizes dataset distributions.

## File Structure
```
Disaster-Response-Pipeline/
├── app
│   ├── templates
│   │   ├── go.html          # Results page
│   │   └── master.html      # Main page
│   └── run.py               # Flask app
├── data
│   ├── disaster_messages.csv
│   ├── disaster_categories.csv
│   ├── DisasterResponse.db   # SQLite DB (created by ETL)
│   └── process_data.py       # ETL script
├── models
│   ├── train_classifier.py   # ML pipeline
├── requirements.txt          # Dependencies
└── README.md                 # Project Documentation
```

## Getting Started

### Prerequisites
- Python 3.6+
- [pip](https://pip.pypa.io/en/stable/)
- Install dependencies:
  ```bash
  pip install -r requirements.txt
  ```

### NLTK Setup
Some NLTK data might need to be downloaded:
```python
import nltk
nltk.download('punkt')
nltk.download('wordnet')
```

## Instructions

1. **Run the ETL pipeline** to clean and store data in the database:
   ```bash
   python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
   ```
2. **Run the ML pipeline** to train and save the model:
   ```bash
   python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl
   ```
3. **Start the Flask app**:
   ```bash
   python app/run.py
   ```
4. **Open your browser** at [http://127.0.0.1:3001/](http://127.0.0.1:3001/) to:
   - Input a message for classification.
   - View classification results.
   - Explore dataset visualizations.

## Project Highlights
- **Multi-output classification** for classifying messages into multiple categories.
- **GridSearchCV** for hyperparameter tuning.
- **Tokenization & Lemmatization** via NLTK.
- **Interactive Flask web app** with Bootstrap and Plotly visualizations.

## Handling Class Imbalance
In many real-world disaster datasets, certain categories (e.g., "food", "medical_help", "missing_people") are much rarer than others (e.g., "related"). This leads to a **class imbalance** situation, where some labels are extremely frequent and others appear only a handful of times.

1. **Impact on Training**: Models can become biased towards predicting frequent classes. If a category rarely appears, it may be harder for the classifier to learn patterns for it.
2. **Precision vs. Recall**: For rare but critical categories (like "missing_people"), we might prioritize **recall** over precision, ensuring we don't miss vital messages even at the risk of some false positives. On the other hand, for more common categories, we might prioritize **precision** to avoid overwhelming the system with false alarms.
3. **Possible Mitigations**:
   - **Class Weighting**: Adjusting weights in the classifier to pay more attention to rare classes.
   - **Oversampling or Undersampling**: Balancing data distribution with methods like SMOTE or random undersampling.
   - **Threshold Tuning**: Adjusting decision thresholds to shift emphasis between precision and recall.

By carefully examining the confusion matrix and classification reports, you can decide where to emphasize precision or recall for each category. In scenarios where the cost of missing a critical category is very high, a recall-oriented approach is often favored. In other contexts, you may want fewer false positives, thus emphasizing precision.

## Licensing, Authors, and Acknowledgements
- **Credits** to [Udacity](https://www.udacity.com) for project inspiration.
- Data provided by [Figure Eight](https://appen.com/) (formerly CrowdFlower).
- This project is open source under the MIT License (or your chosen license).

## Contact
If you have any questions, please [reach out](mailto:youremail@domain.com) or open an issue on the repository.

