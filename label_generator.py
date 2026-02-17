# label_generator.py
import numpy as np
import pandas as pd

class LabelGenerator:
    def __init__(self, config):
        self.config = config['target']

    def compute_composite_score(self, df):
        weights = self.config['composite_weights']
        score = 0
        for col, w in weights.items():
            if col in df.columns:
                # normalize each to 0-1
                min_val = df[col].min()
                max_val = df[col].max()
                if max_val > min_val:
                    normalized = (df[col] - min_val) / (max_val - min_val)
                else:
                    normalized = 0
                score += w * normalized
        # add some non-linear transformation (e.g., polynomial)
        score = score + 0.1 * score**2  # non-linear
        # noise
        noise = np.random.normal(0, 0.05, len(df))
        score = np.clip(score + noise, 0, 1)
        return score

    def assign_level(self, df, composite):
        thresholds = self.config['level_thresholds']  # percentiles
        # convert to quantiles
        t1, t2, t3 = np.percentile(composite, thresholds)
        conditions = [
            composite < t1,
            (composite >= t1) & (composite < t2),
            (composite >= t2) & (composite < t3),
            composite >= t3
        ]
        choices = ['Beginner', 'Intermediate', 'Advanced', 'Exceptional']
        return np.select(conditions, choices, default='Intermediate')

    def assign_division(self, df, level_col):
        rules = self.config['division_rules']
        noise_prob = self.config['noise_level']

        def get_division(row):
            level = row[level_col]
            base_rule = rules[level]
            base_div = base_rule['base']
            # apply probabilistic rule
            if base_div == 'Remedial':
                probs = [base_rule['prob_remedial'], base_rule['prob_regular'], 0, 0]
                choices = ['Remedial', 'Regular', 'Advanced', 'Honors']
            elif base_div == 'Regular':
                probs = [0, base_rule['prob_regular'], base_rule['prob_advanced'], 0]
                choices = ['Remedial', 'Regular', 'Advanced', 'Honors']
            elif base_div == 'Advanced':
                probs = [0, 0, base_rule['prob_advanced'], base_rule['prob_honors']]
                choices = ['Remedial', 'Regular', 'Advanced', 'Honors']
            elif base_div == 'Honors':
                probs = [0, 0, base_rule['prob_advanced'], base_rule['prob_honors']]
                choices = ['Remedial', 'Regular', 'Advanced', 'Honors']
            # add socio-economic adjustments: first_generation or low income increase chance of remedial
            if row.get('first_generation_learner') == 'Yes' or row.get('family_income_range') in ['Low', 'Lower-Middle']:
                # shift probability towards remedial for lower levels
                if level in ['Intermediate', 'Advanced']:
                    # increase remedial chance slightly
                    probs[0] += 0.1
                    probs[1] -= 0.05
                    probs[2] -= 0.05
            probs = np.clip(probs, 0, 1)
            probs = probs / probs.sum()
            # noise: with prob noise_prob, choose uniformly at random
            if np.random.random() < noise_prob:
                return np.random.choice(choices)
            else:
                return np.random.choice(choices, p=probs)

        return df.apply(get_division, axis=1)

    def add_labels(self, df):
        composite = self.compute_composite_score(df)
        df['composite_score'] = composite  # optional intermediate
        df['predicted_student_level'] = self.assign_level(df, composite)
        df['division_allotted'] = self.assign_division(df, 'predicted_student_level')
        return df