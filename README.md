# **🌆 Urban Heat Island Index Prediction**
**Using Satellite & Weather Data with Machine Learning**  
🚀 **Streamlit Web App** | 🌍 **NYC Data** | 🤖 **Random Forest Model**  

![Streamlit](https://img.shields.io/badge/Streamlit-1.32.2-red?logo=streamlit)  
![Azure](https://img.shields.io/badge/Deployed%20on-Azure-blue?logo=microsoftazure)  
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Random%20Forest-green?logo=sklearn)  

---

## **🌍 Overview**
This project predicts the **Urban Heat Island (UHI) Index** for **New York City (NYC)** using a **Random Forest Model** trained on:  
✅ **Satellite Imagery** (Sentinel-2 & Landsat-8)  
✅ **Weather Data** (Temperature, Humidity, Wind Speed)  
✅ **Building Footprints** (Urban structure & land use)  

🔹 Users can **click on a map** to get a **UHI prediction**.  
🔹 Predictions are categorized as **High / Moderate / Low** based on training data.  

📌 **[Live Demo](https://uhi-app.azurewebsites.net)**  

---

## **🖥️ How to Use**
You have two options to run this application:

### **1️⃣ Use the Hosted App on Azure** (Recommended)
Simply visit **[https://uhi-app.azurewebsites.net](https://uhi-app.azurewebsites.net)** to start predicting the UHI index!

### **2️⃣ Build & Run the App Using Docker**
If you prefer to run the app locally using Docker, follow these steps:

#### **Step 1: Clone the Repository**
```sh
git clone https://github.com/margotgeerts/uhi-app.git
cd uhi-app
```

#### **Step 2: Build the Docker Image**
```sh
docker build -t uhi-app .
```

#### **Step 3: Run the Container**
```sh
docker run -p 8501:8501 uhi-app
```

The app will be available at **http://localhost:8501** 🎉.

---

## **📊 Model Details**
- **Algorithm**: Random Forest Regressor  
- **Training Data**:  
  🔹 Sentinel-2 & Landsat-8 (NDVI, LST, Albedo)  
  🔹 NYC Weather Data (Temp, Humidity, Wind, Precipitation)  
  🔹 NYC Building Footprints (Density, Height, Coverage)  
- **Prediction Output**:  
  🔹 **UHI Index** (Numerical Value)  
  🔹 **Risk Category** (High / Moderate / Low)  

---

## **🚀 Deployment & CI/CD**
This project is deployed on **Azure App Service** using **Docker** and **GitHub Actions** for automated deployment.


- **Dockerized Streamlit app** pushed to **GitHub Container Registry (GHCR)**
- **GitHub Actions** automates the build & deployment process
- **Azure App Service** pulls the latest image and serves the app


---


## **📬 Questions or Issues?**
- Open an **issue** on GitHub  
- Reach out via **email** or **LinkedIn**  

---

### **📌 Summary**
✅ **Streamlit App in Docker**  
✅ **Predicts UHI Index using ML & Satellite Data**  
✅ **Docker Image pushed to GHCR Registry**  
✅ **Deployed on Azure**  
✅ **CI/CD with GitHub Actions**  


