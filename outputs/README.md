# Synthetic Student Admission Dataset

## Overview
This dataset contains 100000 synthetic student records generated for machine learning research on university admission and academic division allocation. It simulates realistic patterns and correlations among demographic, academic, socio-economic, and behavioral factors.

## Purpose
- Predict student's academic level at admission
- Automatically allot academic division (Remedial/Regular/Advanced/Honors)
- Benchmark educational AI models

## Features
- **Demographics**: gender, age, category, state, urban/rural
- **Academic Background**: board, percentages, stream, subject scores, medium, gap years
- **Entrance & Aptitude**: entrance exam, aptitude, logical reasoning, language proficiency
- **Socio-Economic Indicators**: family income, parental education, first-generation learner, internet access, study resources
- **Behavioral**: self-learning index, motivation, attendance likelihood
- **Targets**:
  - `predicted_student_level`: Beginner, Intermediate, Advanced, Exceptional
  - `division_allotted`: Remedial, Regular, Advanced, Honors

## Data Generation Process
- Realistic distributions and correlations (e.g., academic scores correlate with entrance scores)
- Non-linear relationships and injected noise
- Missing values (MCAR/MAR patterns)
- Fully synthetic, no real individuals

## File Formats
- CSV: `outputs/student_admission.csv`
- Parquet: `outputs/student_admission.parquet`

## Usage Examples
- Classification tasks (multiclass, ordinal)
- Regression on composite score
- Fairness analysis across demographic groups
- Missing value imputation

## Ethical Note
This dataset is entirely synthetic and generated using probabilistic models. It does not contain any real personal information and is intended for research and educational purposes only.

## License
[Choose appropriate license, e.g., CC BY 4.0]

## Contact
[Your information]
