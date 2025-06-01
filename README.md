# Text Mining Project

### Table of Contents

* [About the Project](#about-the-project)
    * [Built With](#built-with)
* [Getting Started](#getting-started)
    * [Prerequisites](#prerequisites)
    * [Installation](#installation)
* [Usage](#usage)
* [Datasets](#datasets)
* [Contributing](#contributing)
* [License](#license)
* [Contact](#contact)
* [Acknowledgements](#acknowledgements)

---


### About The Project

This project explores fundamental Natural Language Processing (NLP) tasks: **Sentiment Analysis**, **Named Entity Recognition and Classification (NERC)**, and **Topic Classification**. It provides practical examples and implementations of various NLP techniques, from data preprocessing and feature extraction to the application of machine learning and deep learning models.

The goal is to demonstrate how to:
* Analyze textual data to classify opinions (positive, negative, neutral).
* Identify and categorize named entities (e.g., persons, organizations, locations) within text.
* Assign thematic categories or topics to documents.

The project includes runnable Jupyter Notebooks that guide through each task, showcasing different approaches and model implementations, including Support Vector Machines (SVM) for sentiment and a spaCy-based baseline for NERC, along with a separate notebook for Topic Classification.

#### Built With

* Python
* Jupyter Notebook
* Pandas (for data manipulation)
* Scikit-learn (for machine learning models like SVM and evaluation metrics)
* NLTK (Natural Language Toolkit for text preprocessing)
* spaCy (for efficient NLP, particularly for NERC)
* Hugging Face Transformers (potentially for more advanced NERC or Topic Classification, as indicated in `NERC CODE.ipynb` snippet)

### Getting Started

To get a local copy up and running, follow these simple steps.

#### Prerequisites

Ensure you have Python 3.x installed on your system. You'll also need `pip` for package installation.

* Python 3.x
* pip

#### Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your_username/your_nlp_project.git](https://github.com/your_username/your_nlp_project.git)
    cd your_nlp_project
    ```
2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: `venv\Scripts\activate`
    ```
3.  **Install the required Python packages:**
    *If you have a `requirements.txt` file, use:*
    ```bash
    pip install -r requirements.txt
    ```
    *Otherwise, install manually:*
    ```bash
    pip install pandas scikit-learn numpy jupyter nltk spacy transformers torch
    ```
4.  **Download NLTK data:**
    Open a Python interpreter or a Jupyter Notebook cell and run:
    ```python
    import nltk
    nltk.download('punkt')
    nltk.download('stopwords')
    # You might need other NLTK datasets as specified in the notebooks
    ```
5.  **Download spaCy models:**
    For the NERC part, you'll need a spaCy model:
    ```bash
    python -m spacy download en_core_web_sm
    ```

### Usage

The core of this project is demonstrated within the Jupyter Notebooks.

1.  **Start Jupyter Notebook:**
    ```bash
    jupyter notebook
    ```
2.  **Open the notebooks:**
    * `SentimentCodeHasan.ipynb`: Contains the main sentiment analysis workflow, covering data loading, preprocessing, model training, and evaluation.
    * `Setimental analysis-SVM.ipynb`: Specifically focuses on implementing and evaluating a Support Vector Machine (SVM) model for sentiment classification.
    * `NERC CODE.ipynb`: Demonstrates Named Entity Recognition and Classification, possibly using a spaCy baseline and suggesting a Hugging Face model for comparison.
    * `Topic_Classification_for_Report.ipynb`: Implements and evaluates a Topic Classification model.

Follow the instructions and code cells within the notebooks to run the analyses and experiments for each NLP task. You can modify the code to experiment with different models, features, or datasets.

### Datasets

The project utilizes the following datasets (included in the repository):

 The dataset(`sentiment_data.csv`) used in this study is the "Financial Sentiment Analysis" dataset, containing over 5,800 entries. 
 Each entry consists of two features: text expressing opinions related to financial topics and the corresponding sentiment label (positive, negative, or neutral). 
 The training data is highly imbalanced: neutral ~53%, positive ~31%, and negative ~14%. 
 The test dataset contains 18 text samples labeled with the same three sentiment classes, where all classes are evenly balanced.

### Contributing

Contributions are what make the open-source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also open an issue with the tag "enhancement".

1.  Fork the Project
2.  Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3.  Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4.  Push to the Branch (`git push origin feature/AmazingFeature`)
5.  Open a Pull Request

### License

Distributed under the MIT License. See `LICENSE` for more information.

### Contact

**Vrije University**
* Hyunwoo Song - h.song@student.vu.nl
* Hassan - ..
* D.G.J.K. Linger - ..
* F. Moser - ..

Project Link: [https://github.com/AIVU2026/TextMining-project]

### Acknowledgements

* Scikit-learn documentation
* NLTK library
* spaCy documentation
* Hugging Face Transformers library
* Jupyter Project
