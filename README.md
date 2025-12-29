#  Consumer Complaint Prediction

##  Overview
This project is a Machine Learning classification application that predicts whether a consumer will dispute a company's response to their complaint.

By analyzing historical complaint data, the model identifies patterns in customer issues, products, and narratives to forecast the likelihood of a dispute. This helps companies prioritize high-risk complaints and improve customer satisfaction.

* CSV FILES : https://drive.google.com/drive/folders/1Xatzc8y38dgDt2PS1qJgFf0YHaaAgN_W?usp=sharing
##  Dataset
The project uses two main datasets:
1.  **`consumer_complaints_train.csv`**: Used to train and validate the machine learning models.
2.  **`consumer_complaints_test.csv`**: Used to evaluate the final performance of the model on unseen data.

*Note: The dataset contains features such as Product, Sub-product, Issue, Company response to consumer, and Consumer complaint narrative.*

## 🛠️ Tech Stack
* **Language:** Python
* **Data Manipulation:** Pandas, NumPy
* **Visualization:** Matplotlib, Seaborn
* **Machine Learning:** Scikit-learn (e.g., Random Forest, Logistic Regression, XGBoost)
* **Natural Language Processing (NLP):** NLTK / TF-IDF (if text analysis was used on the narratives)

##  Project Workflow

### 1. Data Preprocessing
* **Loaded Data:** Imported the Train and Test CSV files.
* **Handling Missing Values:** Imputed or removed missing entries in critical columns.
* **Feature Engineering:**
    * Encoded categorical variables (like `Product`, `Issue`, `Company`) using Label Encoding/One-Hot Encoding.
    * (Optional) Processed text data from the "Complaint Narrative" using TF-IDF vectorization.
* **Target Variable:** Mapped the target column (`Consumer disputed?`) to binary values (0 and 1).

### 2. Exploratory Data Analysis (EDA)
* Analyzed the distribution of complaints across different products and companies.
* Visualized the ratio of disputed vs. non-disputed complaints to check for class imbalance.

### 3. Model Building
Tried various classification algorithms to find the best fit:
* **Logistic Regression** (Baseline model)
* **Random Forest Classifier**
* **XGBoost / Gradient Boosting**

**Result:** The best performing model was selected based on Accuracy and F1-Score.

##  Results
* **Best Model:** [Enter Model Name, e.g., Random Forest]
* **Accuracy:** [Enter Accuracy, e.g., 82%]
* **F1 Score:** [Enter Score, e.g., 0.78]

##  How to Run

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/consumer-complaint-prediction.git](https://github.com/your-username/consumer-complaint-prediction.git)
    cd consumer-complaint-prediction
    ```

2.  **Install dependencies:**
    ```bash
    pip install pandas numpy scikit-learn matplotlib seaborn
    ```

3.  **Download the Data:**
    * Since the CSV files are large, download them from [Link to Kaggle or Drive] and place them in the project folder.
    * *Or, if you zipped them:* Unzip the `data.zip` file.

4.  **Run the Notebook/Script:**
    Open the Jupyter Notebook to see the training process:
    ```bash
    jupyter notebook "Consumer Complaint Analysis.ipynb"
    ```

##  Future Improvements
* Implement Deep Learning (LSTM/BERT) for better text analysis of complaint narratives.
* Build a Flask dashboard to allow users to input complaint details and get real-time predictions.
* Handle class imbalance using SMOTE techniques to improve dispute detection.

