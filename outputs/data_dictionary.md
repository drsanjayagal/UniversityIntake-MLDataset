# Data Dictionary

| Column | Type | Description |
|--------|------|-------------|
| student_id | object | Unique anonymized identifier |
| gender | object | Gender of the student |
| age | float64 | Age at admission (16-25) |
| category | object | Social category (General/OBC/SC/ST/EWS) |
| state | object | Indian state of residence |
| urban_rural | object | Urban or Rural background |
| board_type | object | Education board (CBSE/ICSE/State/International) |
| class_10_percentage | float64 | Percentage in Class 10 (30-100) |
| class_12_percentage | float64 | Percentage in Class 12 (30-100) |
| stream | object | Stream in Class 11-12 (Science/Commerce/Arts) |
| math_score | float64 | Mathematics score (0-100) |
| science_score | float64 | Science score (0-100) |
| english_score | float64 | English score (0-100) |
| medium_of_instruction | object | Primary medium (English/Regional) |
| gap_year | float64 | Number of gap years after Class 12 |
| logical_reasoning_score | float64 | Logical reasoning score (0-100) |
| language_proficiency_score | float64 | Language proficiency score (0-100) |
| entrance_exam_score | float64 | Entrance exam score (0-100) |
| aptitude_score | float64 | General aptitude score (0-100) |
| family_income_range | object | Family income bracket |
| parent_education_level | object | Highest parental education |
| first_generation_learner | object | Whether first in family to attend higher education |
| internet_access | object | Access to internet at home |
| study_resources_index | float64 | Index of study resources availability (0-1) |
| self_learning_index | float64 | Self-learning capability index (0-1) |
| motivation_score | float64 | Self-reported motivation level (1-10) |
| attendance_likelihood | float64 | Predicted attendance likelihood (0-1) |
| family_income | float64 | Synthetic feature |
| parent_education | float64 | Synthetic feature |
| predicted_student_level | object | Target: predicted academic level (Beginner/Intermediate/Advanced/Exceptional) |
| division_allotted | object | Target: allotted academic division (Remedial/Regular/Advanced/Honors) |