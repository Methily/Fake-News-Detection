# Fake News Detection Project

A simple machine learning project that can detect fake news articles using text analysis.

## What This Project Does

This project trains a computer to read news articles and tell you whether they're real or fake news. It's like having a smart assistant that can spot fake news for you!

## The Data

- **Dataset**: 44,898 news articles
- **What it learns from**: The text content of news articles
- **What it predicts**: Whether an article is REAL (0) or FAKE (1)

## How to Run the Project

### Step 1: Setup (One-time only)
```bash
# Clone the repository
git clone <your-repo-url>
cd fraud-mini

# Create a virtual environment
python -m venv fraud_env

# Activate the virtual environment
# On Windows:
fraud_env\Scripts\activate
# On macOS/Linux:
source fraud_env/bin/activate

# Install required packages
pip install -r requirements.txt
```

### Step 2: Run the Project
```bash
# Make sure your virtual environment is activated
fraud_env\Scripts\activate

# Run the project
python main.py
```

## What Happens When You Run It

1. **Loads** the fake news dataset (44,898 articles)
2. **Prepares** the text data for analysis
3. **Trains** 3 different AI models:
   - Logistic Regression
   - Linear SVM
   - Random Forest
4. **Tests** how well each model performs
5. **Saves** the best model for future use

## Results

The project achieved excellent results:
- **Best Model**: Random Forest
- **Accuracy**: 99.63% (F1-Score)
- **Success Rate**: 99.97% (ROC-AUC)

## What Gets Created

After running, you'll find these files in the `models/` folder:
- `best_model.pkl` - The best performing model
- `random_forest_model.pkl` - Random Forest model
- `linear_svm_(sgd)_model.pkl` - Linear SVM model
- `logistic_regression_model.pkl` - Logistic Regression model
- `tfidf_vectorizer.pkl` - Text processor

## Requirements

- Python 3.13+
- Virtual environment (recommended)
- 175MB of disk space for the dataset

## How It Works

1. **Text Processing**: Converts news articles into numbers that computers can understand
2. **Feature Extraction**: Identifies important words and patterns in the text
3. **Model Training**: Teaches the computer to recognize patterns that indicate fake news
4. **Prediction**: Uses the trained model to classify new articles as real or fake

## Use Cases

- **Fact-checking**: Quickly identify potentially fake news articles
- **Research**: Study patterns in fake news writing
- **Education**: Learn about machine learning and text analysis
- **Development**: Build your own fake news detection tools

## Troubleshooting

**If you get errors:**
1. Make sure your virtual environment is activated
2. Check that all packages are installed: `pip install -r requirements.txt`
3. Ensure the dataset file exists in `data/fake.csv`

**Common Issues:**
- **NumPy warnings**: These are normal and won't affect the project
- **Slow loading**: The dataset is large (175MB), so it may take a moment to load

## 📝 Next Steps

Once the model is trained, you can:
- Use the saved models to predict on new articles
- Improve the model with more data
- Build a web interface for easy use
- Add more features like source credibility

## 🤝 Contributing

Feel free to contribute to this project by:
- Reporting bugs
- Suggesting new features
- Improving the documentation
- Adding new models or preprocessing techniques

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

**Happy Fake News Detection! 🕵️‍♂️**
