# 🧠 DataMindAI

### AI-Powered Data Analysis & AutoML Platform

DataMindAI is an AI-powered data analysis platform that allows users to upload a dataset, interact with it using natural language, automatically analyze the data, train multiple machine-learning models, compare their performance, and make predictions through an interactive web interface.

The goal of DataMindAI is to make data science workflows more accessible by combining **natural-language dataset analysis, automated machine learning, preprocessing intelligence, model persistence, and prediction** into a single application.

---

## ✨ Features

### 💬 Natural-Language Data Analysis

Interact with your dataset through a conversational interface instead of writing pandas or SQL code manually.

Examples:

```text
Show me the top 5 companies by employees.

What is the average salary?

Which columns contain missing values?

How many unique companies are there?

Show me the relationship between revenue and employees.
```

DataMindAI analyzes the question and uses the appropriate dataset-analysis functionality to generate the answer.

---

### 📊 Dataset Intelligence

After uploading a dataset, DataMindAI analyzes its structure and provides information about:

* Dataset dimensions
* Column types
* Missing values
* Duplicate records
* Numerical and categorical features
* Dataset quality
* Potential target variables
* Data characteristics

The system also generates preprocessing recommendations based on the dataset.

---

### 🤖 AutoML

DataMindAI can automatically train and compare multiple machine-learning models.

Supported models include:

* Logistic Regression
* Random Forest
* XGBoost
* LightGBM
* Linear Regression

The system automatically detects whether the problem is:

* Binary Classification
* Multiclass Classification
* Regression

Models are evaluated and compared using task-appropriate metrics, and the best-performing model is selected.

---

### ⚙️ Intelligent Preprocessing

The preprocessing pipeline automatically handles common dataset requirements such as:

* Numerical features
* Categorical features
* Missing values
* Feature transformations
* Encoding
* Feature scaling when required
* Unknown categorical values

The preprocessing pipeline is integrated with model training so that the same transformations can be reused during prediction.

---

### 🔮 Model Prediction

After training a model, users can move to the separate **Prediction** workspace.

The application:

1. Loads the saved model artifact.
2. Displays the required input features.
3. Collects prediction values from the user.
4. Applies the saved preprocessing pipeline.
5. Generates the prediction.
6. Displays prediction results and classification probabilities when available.

This prevents the preprocessing used during prediction from becoming inconsistent with the preprocessing used during training.

---

### 💾 Model Persistence

The selected model and its associated preprocessing information can be saved as a reusable model artifact.

This allows a trained model to be loaded later for prediction without retraining it.

---

### 🧭 Separate Workspaces

The application separates its major workflows into dedicated sections:

```text
┌──────────────────────────────┐
│          DataMindAI          │
├──────────────────────────────┤
│                              │
│  💬 Chat                     │
│  🤖 Model / AutoML           │
│  🔮 Prediction               │
│                              │
└──────────────────────────────┘
```

This keeps the conversational data-analysis workflow separate from model training and prediction.

---

### 📁 Dataset Support

DataMindAI supports common tabular dataset formats including:

* CSV
* Excel / XLSX

The uploaded dataset becomes the central source for the analysis, modeling, and prediction workflows.

---

## 🏗️ Architecture

The application follows a modular architecture:

```text
                         ┌─────────────────┐
                         │   Streamlit UI  │
                         └────────┬────────┘
                                  │
              ┌───────────────────┼───────────────────┐
              │                   │                   │
              ▼                   ▼                   ▼
        ┌───────────┐       ┌───────────┐       ┌────────────┐
        │   Chat    │       │   AutoML  │       │ Prediction │
        └─────┬─────┘       └─────┬─────┘       └──────┬─────┘
              │                   │                    │
              ▼                   ▼                    ▼
        ┌───────────┐       ┌──────────────┐     ┌─────────────┐
        │   Agent   │       │ ML Pipeline  │     │ Prediction  │
        └─────┬─────┘       └──────┬───────┘     │   Engine    │
              │                    │             └──────┬──────┘
              ▼                    ▼                    │
        ┌──────────────┐    ┌──────────────┐            │
        │ Dataset Q&A │    │ Preprocessing │            │
        └──────┬───────┘    └──────┬───────┘            │
               │                   │                    │
               └───────────────────┼────────────────────┘
                                   ▼
                           ┌─────────────────┐
                           │  Model Artifact │
                           └─────────────────┘
```

---

## 🛠️ Technology Stack

### Frontend

* Streamlit
* Python
* Pandas
* Plotly

### Machine Learning

* Scikit-learn
* XGBoost
* LightGBM
* Pandas
* NumPy

### AI / Agent Layer

* Qwen / local LLM integration
* Agent-based dataset interaction
* Natural-language dataset Q&A

### Data Processing

* Pandas
* Scikit-learn preprocessing pipelines

### Testing

* Python `unittest`
* Automated unit tests for:

  * Dataset analysis
  * Task detection
  * Preprocessing
  * Model training
  * Prediction
  * Dataset Q&A
  * Agent behavior

---

## 📂 Project Structure

```text
DataMindAI/
│
├── run.py
├── requirements.txt
├── README.md
│
├── src/
│   │
│   ├── agent/
│   │   └── agent.py
│   │
│   ├── backend/
│   │   ├── data_loader.py
│   │   ├── dataset_qa.py
│   │   ├── dataset_intelligence.py
│   │   └── preprocessing_intelligence.py
│   │
│   ├── core/
│   │   ├── app_state.py
│   │   ├── config.py
│   │   ├── exceptions.py
│   │   └── logger.py
│   │
│   ├── frontend/
│   │   ├── chat_page.py
│   │   ├── home.py
│   │   └── sidebar.py
│   │
│   ├── main/
│   │   └── app.py
│   │
│   └── ml/
│       ├── task_detector.py
│       ├── preprocessing_pipeline.py
│       ├── trainer.py
│       ├── pipeline.py
│       ├── model_persistence.py
│       └── prediction_engine.py
│
└── tests/
    ├── test_dataset_qa.py
    ├── test_task_detector.py
    ├── test_preprocessing_pipeline.py
    ├── test_trainer.py
    └── ...
```

---

## 🚀 Installation

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/DataMindAI.git
cd DataMindAI
```

### 2. Create a virtual environment

#### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

#### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the application

```bash
streamlit run run.py
```

The application will open in your browser.

---

## 🧪 Running Tests

Run the complete test suite:

```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

The project contains tests covering the major components of the application, including dataset analysis, preprocessing, task detection, model training, prediction, and agent behavior.

---

## 🔄 Typical Workflow

```text
1. Launch DataMindAI
        ↓
2. Upload CSV / XLSX dataset
        ↓
3. Dataset Intelligence analyzes the data
        ↓
4. Chat with the dataset
        ↓
5. Open Model / AutoML
        ↓
6. Select target variable
        ↓
7. Select candidate models
        ↓
8. Train and evaluate models
        ↓
9. Select the best model
        ↓
10. Save model artifact
        ↓
11. Open Prediction
        ↓
12. Enter feature values
        ↓
13. Generate prediction
```

---

## 🎯 Why DataMindAI?

Traditional data-science workflows often require users to manually:

* Inspect datasets
* Identify data types
* Handle missing values
* Encode categorical variables
* Scale features
* Select machine-learning algorithms
* Train multiple models
* Compare metrics
* Save models
* Prepare prediction inputs

DataMindAI brings many of these steps together into a single workflow.

Instead of starting with:

```python
df = pd.read_csv(...)
```

and manually writing the complete analysis and ML pipeline, users can start with:

```text
Upload dataset → Ask questions → Train models → Predict
```

---

## 🔐 Design Principles

### Modular Architecture

Dataset analysis, preprocessing, ML training, prediction, agent logic, and frontend components are separated into independent modules.

### Reusable ML Pipelines

The preprocessing applied during training is preserved with the model so that prediction uses the same transformations.

### Deterministic Dataset Analysis

Dataset questions that require exact numerical answers are handled through programmatic data-analysis operations rather than relying solely on an LLM to calculate values.

### Testable Components

Core functionality is implemented in modules that can be independently tested.

---

## 🧠 Future Improvements

Potential future improvements include:

* More advanced natural-language data analysis
* Better conversational memory
* More visualization types
* Automated feature engineering
* Hyperparameter optimization
* Experiment tracking
* Model explainability
* Feature importance visualization
* SHAP-based explanations
* Automated ML reports
* Model monitoring
* Deployment-ready prediction APIs
* Support for additional file formats

---

## 👨‍💻 Author

**Atyant Srivastava**

B.Tech Computer Science & Engineering

Interested in:

* Machine Learning
* Data Science
* Artificial Intelligence
* Full-Stack Development
* Applied AI Systems

---

## ⭐ If you find this project useful

Give the repository a ⭐ on GitHub and feel free to explore, contribute, or suggest improvements.

---

## 📄 License

This project is intended for educational and development purposes. Add an appropriate open-source license such as MIT if you plan to distribute the project publicly.
