# Task 2 - Customer Segmentation using KMeans Clustering

## Overview
This project performs **customer segmentation** on a mall dataset using the **KMeans clustering algorithm**. The goal is to group customers based on their **Annual Income** and **Spending Score**, helping businesses target marketing strategies more effectively.

---

## Dataset
- **Mall_Customers.csv**: Contains customer information including:
  - CustomerID
  - Gender
  - Age
  - Annual Income (k$)
  - Spending Score (1-100)

- **Clustered_Customers.csv**: Output dataset with an additional `Cluster` column indicating the assigned cluster for each customer.

## Project Structure

SCT_ML_2/
├─ data/
│ └─ Mall_Customers.csv
├─ results/
│ ├─ Clustered_Customers.csv
│ ├─ Customer_data_distribution.png
│ ├─ Elbow_Method_Optimal_K.png
│ └─ Customers_Segments_KMeans.png
├─ src/
│ └─ task2.py
├─ README.md
└─ requirements.txt


- **data/**: Original datasets  
- **results/**: Output plots and clustered data  
- **src/**: Python script for analysis  
- **requirements.txt**: Required Python libraries  

---

## How to Run

1. Clone the repository:
```bash
git clone https://github.com/PAVANSAIVVIT/SCT_ML_2.git
