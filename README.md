# Customer Churn Prediction Flask App

A modern, responsive Flask web application for predicting customer churn using machine learning. Features a beautiful glassmorphism UI design and real-time predictions.

![Flask App](https://img.shields.io/badge/Flask-2.3.3-blue)
![Python](https://img.shields.io/badge/Python-3.8+-green)
![Machine Learning](https://img.shields.io/badge/ML-Random%20Forest-orange)

## ✨ Features

- 🎨 **Modern UI Design** - Beautiful glassmorphism interface with gradient backgrounds
- 📱 **Responsive Design** - Works perfectly on desktop and mobile devices
- 🤖 **Machine Learning** - Random Forest model for accurate churn predictions
- ⚡ **Real-time Predictions** - Instant results with color-coded risk assessment
- 🔧 **Easy Setup** - Simple installation and configuration
- 📊 **Multiple Features** - Contract type, tenure, charges analysis

## 🚀 Live Demo

[Try the live demo here](https://your-app-url.com) _(Add your deployed URL)_

## 📋 Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

## 🛠️ Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/yourusername/customer-churn-prediction.git
   cd customer-churn-prediction
   ```

2. **Create a virtual environment**

   ```bash
   python -m venv venv

   # On Windows
   venv\Scripts\activate

   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Train the model** (Optional - if you have the dataset)

   ```bash
   python train_model.py
   ```

5. **Run the application**

   ```bash
   python main.py
   ```

6. **Open your browser**
   Navigate to `http://localhost:5000`

## 📁 Project Structure

```
customer-churn-prediction/
│
├── main.py                 # Main Flask application
├── train_model.py          # Model training script
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── .gitignore             # Git ignore rules
├── churn_model.pkl        # Trained model (generated)
├── scaler.pkl             # Feature scaler (generated)
└── templates/
    └── index.html         # Web interface template
```

## 🎯 Usage

1. **Open the application** in your web browser
2. **Fill in customer information:**
   - Contract Type (Month-to-month, One year, Two year)
   - Tenure (number of months)
   - Monthly Charges (dollar amount)
   - Total Charges (dollar amount)
3. **Click "Predict Churn Risk"**
4. **View the prediction result** with color-coded risk assessment

## 🔬 Model Information

The application uses a **Random Forest Classifier** trained on the Telco Customer Churn dataset with the following features:

- **Contract Information**: Contract type and tenure
- **Financial Data**: Monthly and total charges
- **Risk Factors**: Contract duration, payment patterns, and service costs

### Prediction Logic

When the trained model is available:

- Uses machine learning model for predictions
- Provides probability scores for churn risk

When model is not available:

- Uses rule-based prediction logic
- Considers factors like contract type, tenure, and charges

## 🎨 UI Features

- **Glassmorphism Design**: Modern transparent glass effect
- **Gradient Backgrounds**: Beautiful purple-blue gradients
- **Responsive Layout**: Adapts to all screen sizes
- **Interactive Elements**: Hover effects and smooth animations
- **Color-coded Results**: Red for high risk, blue for low risk
- **Form Validation**: Input validation and error handling

## 🔧 Configuration

### Environment Variables

Create a `.env` file for custom configuration:

```env
FLASK_ENV=development
FLASK_DEBUG=True
PORT=5000
HOST=0.0.0.0
```

### Model Training

To train your own model:

1. Place your dataset in the `data/` directory
2. Update the `train_model.py` script with your data path
3. Run the training script
4. The model will be saved as `churn_model.pkl`

## 🚀 Deployment

### Heroku Deployment

1. **Create a Procfile**

   ```
   web: gunicorn main:app
   ```

2. **Add gunicorn to requirements.txt**

   ```
   gunicorn==20.1.0
   ```

3. **Deploy to Heroku**
   ```bash
   heroku create your-app-name
   git push heroku main
   ```

### Docker Deployment

1. **Create Dockerfile**

   ```dockerfile
   FROM python:3.9-slim
   WORKDIR /app
   COPY requirements.txt .
   RUN pip install -r requirements.txt
   COPY . .
   EXPOSE 5000
   CMD ["python", "main.py"]
   ```

2. **Build and run**
   ```bash
   docker build -t churn-predictor .
   docker run -p 5000:5000 churn-predictor
   ```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Flask framework for the web framework
- Scikit-learn for machine learning capabilities
- Telco Customer Churn dataset for training data
- Modern CSS techniques for the beautiful UI

## 📞 Support

If you have any questions or need help:

- Create an issue in this repository
- Contact: your-email@example.com

---

⭐ **Star this repository if you found it helpful!**
