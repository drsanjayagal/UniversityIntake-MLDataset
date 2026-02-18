# UniversityIntake-MLDataset
# 🎓📊 **Synthetic Student Admission Dataset**  
### *For Machine Learning Research & Educational Use*  

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Dataset Size](https://img.shields.io/badge/Records-100%2C000+-brightgreen)](https://github.com/yourusername/UniversityIntake-MLDataset)
[![GitHub Repo](https://img.shields.io/badge/GitHub-Repository-blue?logo=github)](https://github.com/yourusername/UniversityIntake-MLDataset)
[![Made with Python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg?logo=python)](https://www.python.org/)
[![Designed & Developed By](https://img.shields.io/badge/Designed%20%26%20Developed%20By-Dr.%20Sanjay%20Agal-blueviolet)](#)

---

## 🌟 **Overview**

This repository provides a **fully synthetic**, large-scale dataset of **100,000+ student records** designed to simulate realistic university admission scenarios. The data is generated using probabilistic models and real-world correlation patterns, making it ideal for:

- 🧠 **Predicting student academic levels** (Beginner / Intermediate / Advanced / Exceptional)
- 🏫 **Automatically allotting academic divisions** (Remedial / Regular / Advanced / Honors)
- 📈 **Benchmarking educational AI models** (classification, regression, fairness analysis)
- 🔍 **Exploring socio-economic and academic interactions** in higher education

All data is **anonymized and synthetic**, ensuring no resemblance to real individuals — perfect for open‑source research and teaching.

---

## ✨ **Key Features**

| Category | Attributes | Emoji |
|----------|------------|-------|
| **Demographics** | `student_id`, `gender`, `age`, `category`, `state`, `urban_rural` | 👤🌍 |
| **Academic Background** | `board_type`, `class_10_percentage`, `class_12_percentage`, `stream`, `math_score`, `science_score`, `english_score`, `medium_of_instruction`, `gap_year` | 📚🎓 |
| **Entrance & Aptitude** | `entrance_exam_score`, `aptitude_score`, `logical_reasoning_score`, `language_proficiency_score` | 🧠📝 |
| **Socio‑Economic Indicators** | `family_income_range`, `parent_education_level`, `first_generation_learner`, `internet_access`, `study_resources_index` | 💰🏠 |
| **Behavioral / Soft Indicators** | `self_learning_index`, `motivation_score`, `attendance_likelihood` | ❤️📊 |
| **Target Variables** | `predicted_student_level`, `division_allotted` | 🎯🏷️ |

### 🧩 **Realism & Complexity**
- **Non‑linear relationships** and injected noise
- **Missing values** (MCAR / MAR patterns)
- **Correlations** between academic scores, aptitude, and socio‑economic factors
- **Regional & board‑level variations**
- **Class imbalance** in target variables (optional)

---

## 📁 **Repository Structure**

---

## 📊 **Data Dictionary (Preview)**

| Column | Type | Description |
|--------|------|-------------|
| `student_id` | object | Unique anonymized identifier |
| `gender` | object | Male / Female / Other |
| `age` | int64 | Age at admission (16–25) |
| `category` | object | General / OBC / SC / ST / EWS |
| `state` | object | Indian state of residence |
| `urban_rural` | object | Urban / Rural |
| `board_type` | object | CBSE / ICSE / State / International |
| `class_10_percentage` | float64 | Percentage in Class 10 (30–100) |
| `class_12_percentage` | float64 | Percentage in Class 12 (30–100) |
| `stream` | object | Science / Commerce / Arts |
| `math_score` | float64 | Mathematics score (0–100) |
| `science_score` | float64 | Science score (0–100) |
| `english_score` | float64 | English score (0–100) |
| `medium_of_instruction` | object | English / Regional |
| `gap_year` | int64 | Number of gap years after Class 12 (0–3) |
| `entrance_exam_score` | float64 | Entrance exam score (0–100) |
| `aptitude_score` | float64 | General aptitude score (0–100) |
| `logical_reasoning_score` | float64 | Logical reasoning score (0–100) |
| `language_proficiency_score` | float64 | Language proficiency score (0–100) |
| `family_income_range` | object | Low / Lower‑Middle / Upper‑Middle / High |
| `parent_education_level` | object | No formal / Primary / Secondary / Graduate / Postgraduate |
| `first_generation_learner` | object | Yes / No |
| `internet_access` | object | Yes / No |
| `study_resources_index` | float64 | Index of study resources availability (0–1) |
| `self_learning_index` | float64 | Self‑learning capability index (0–1) |
| `motivation_score` | int64 | Self‑reported motivation level (1–10) |
| `attendance_likelihood` | float64 | Predicted attendance likelihood (0–1) |
| `predicted_student_level` | object | **Target:** Beginner / Intermediate / Advanced / Exceptional |
| `division_allotted` | object | **Target:** Remedial / Regular / Advanced / Honors |

👉 **Full data dictionary**: [`outputs/data_dictionary.md`](outputs/data_dictionary.md)

---
