# generate_dataset.py
import yaml
import pandas as pd
import os
from utils import set_seed
from data_generator import DataGenerator
from label_generator import LabelGenerator

def main():
    # Load config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Set seed for reproducibility
    set_seed(config['dataset']['random_seed'])

    # Generate data
    print("Generating student profiles...")
    data_gen = DataGenerator(config)
    df = data_gen.generate()
    print(f"Generated {len(df)} raw records.")

    # Add labels
    print("Generating target variables...")
    label_gen = LabelGenerator(config)
    df = label_gen.add_labels(df)
    print("Labels added.")

    # Drop intermediate composite if not desired
    df = df.drop(columns=['composite_score'], errors='ignore')

    # Save outputs
    os.makedirs(os.path.dirname(config['output']['csv_path']), exist_ok=True)
    df.to_csv(config['output']['csv_path'], index=False)
    df.to_parquet(config['output']['parquet_path'], index=False)
    print(f"Data saved to {config['output']['csv_path']} and {config['output']['parquet_path']}")

    # Generate documentation
    generate_data_dictionary(df, config)
    generate_readme(config)

def generate_data_dictionary(df, config):
    lines = ["# Data Dictionary\n"]
    lines.append("| Column | Type | Description |")
    lines.append("|--------|------|-------------|")
    for col in df.columns:
        dtype = df[col].dtype
        desc = get_description(col, config)
        lines.append(f"| {col} | {dtype} | {desc} |")
    with open(config['output']['data_dict_path'], 'w') as f:
        f.write("\n".join(lines))

def get_description(col, config):
    descriptions = {
        'student_id': 'Unique anonymized identifier',
        'gender': 'Gender of the student',
        'age': 'Age at admission (16-25)',
        'category': 'Social category (General/OBC/SC/ST/EWS)',
        'state': 'Indian state of residence',
        'urban_rural': 'Urban or Rural background',
        'board_type': 'Education board (CBSE/ICSE/State/International)',
        'class_10_percentage': 'Percentage in Class 10 (30-100)',
        'class_12_percentage': 'Percentage in Class 12 (30-100)',
        'stream': 'Stream in Class 11-12 (Science/Commerce/Arts)',
        'math_score': 'Mathematics score (0-100)',
        'science_score': 'Science score (0-100)',
        'english_score': 'English score (0-100)',
        'medium_of_instruction': 'Primary medium (English/Regional)',
        'gap_year': 'Number of gap years after Class 12',
        'entrance_exam_score': 'Entrance exam score (0-100)',
        'aptitude_score': 'General aptitude score (0-100)',
        'logical_reasoning_score': 'Logical reasoning score (0-100)',
        'language_proficiency_score': 'Language proficiency score (0-100)',
        'family_income_range': 'Family income bracket',
        'parent_education_level': 'Highest parental education',
        'first_generation_learner': 'Whether first in family to attend higher education',
        'internet_access': 'Access to internet at home',
        'study_resources_index': 'Index of study resources availability (0-1)',
        'self_learning_index': 'Self-learning capability index (0-1)',
        'motivation_score': 'Self-reported motivation level (1-10)',
        'attendance_likelihood': 'Predicted attendance likelihood (0-1)',
        'predicted_student_level': 'Target: predicted academic level (Beginner/Intermediate/Advanced/Exceptional)',
        'division_allotted': 'Target: allotted academic division (Remedial/Regular/Advanced/Honors)'
    }
    return descriptions.get(col, 'Synthetic feature')

def generate_readme(config):
    content = f"""# Synthetic Student Admission Dataset

## Overview
This dataset contains {config['dataset']['n_students']} synthetic student records generated for machine learning research on university admission and academic division allocation. It simulates realistic patterns and correlations among demographic, academic, socio-economic, and behavioral factors.

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
- CSV: `{config['output']['csv_path']}`
- Parquet: `{config['output']['parquet_path']}`

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
"""
    with open(config['output']['readme_path'], 'w') as f:
        f.write(content)

if __name__ == "__main__":
    main()