# CourseCraft: A Graph Convolutional Network Approach for Personalized Course Recommendation

CourseCraft is a personalized course recommendation system built using a **Graph Convolutional Network (GCN)**.  
The model predicts user–course ratings by learning from the structure of a large bipartite interaction graph.  
The system is deployed as a full-stack **Django web application** with GCN-powered predictions.

---

## 🚀 Overview

Online course catalogs continue to grow, making it difficult for students to identify the most relevant options.  
Traditional recommenders (content-based, collaborative filtering) struggle with **sparsity**, **limited personalization**,  
and **lack of relational understanding** between users and courses.

CourseCraft addresses these limitations using a **GCN encoder–decoder architecture** inspired by  
Graph Convolutional Matrix Completion (GC-MC).  
The model achieves:

- **1.5M user–course interactions** modeled as a graph  
- **GCN embeddings for users & courses**  
- **RMSE: 0.8576** on held-out test data  
- Deployment inside a Django application with real-time predictions  

---

## 📊 Dataset

The dataset consists of:

- **61,322 users**
- **3,000 courses**
- **1,532,556 user–course interactions**
- Ratings: **1–5 (balanced)**  
- Metadata: userID, courseID, description, ratings
Since MOOCs do not provide ratings, interaction data was **simulated and balanced** across 1–5 ranges to mimic realistic user behavior.

---

## 🧠 Model

CourseCraft uses a bipartite graph where:

- User nodes ↔ Course nodes  
- Edges represent rating values (treated as edge types)

### **GCN Encoder**
Aggregates neighbor information through message passing and learns dense embeddings for users and courses.

### **Bilinear Decoder**
Uses learned embeddings to predict a continuous rating score for unseen user–course pairs.

### **Training**
- Optimizer: **Adam**  
- Loss: **Negative Log-Likelihood**  
- Regularization: Node dropout + hidden-layer dropout  
- Mini-batch training for efficiency  
- Validation/test RMSE tracked each epoch

Final best model checkpoint: **gcmc_full_epoch1.pt**

---

## 🧩 Project Structure

```
CourseRecApp/
├── course_recommender/
│   ├── recsys/
│   │   ├── model/                 
│   │   ├── gcmc_model.py          
│   │   ├── recommender.py         
│   │   ├── models.py              
│   │   ├── views.py               
│   │   ├── forms.py               
│   │   └── templates/recsys/      
│   │
│   ├── train_gcmc.py              
│   ├── gcmc_full_epoch1.pt        
│   ├── course_ratings_dataset.csv 
│   └── db.sqlite3                 
│
├── manage.py
└── README.md
```

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the repository  
```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
```

### 2️⃣ Create virtual environment  
```bash
python -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows
```

### 3️⃣ Install dependencies  
```bash
pip install -r requirements.txt
```

### 4️⃣ Run migrations  
```bash
python manage.py migrate
```

### 5️⃣ Start the Django server  
```bash
python manage.py runserver
```

---

## 🔮 Inference (How Recommendations Work)

During inference:

1. User ID → mapped to embedding index  
2. Course candidates → embeddings fetched  
3. Bilinear decoder computes predicted rating  
4. Top-ranked courses returned to the UI  

The inference module is implemented in:

```
course_recommender/recsys/recommender.py
```

---

## 📈 Results

| Model | RMSE |
|-------|------|
| Matrix Factorization | 1.10 |
| **GCN (CourseCraft)** | **0.8576** |


## 🤝 Contributions

Pull requests and improvements are welcome!

---

