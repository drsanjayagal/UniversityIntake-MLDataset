# data_generator.py
import numpy as np
import pandas as pd
from faker import Faker
from utils import truncated_normal, beta_dist, multivariate_normal_sample, add_noise
import yaml

fake = Faker('en_IN')

class DataGenerator:
    def __init__(self, config):
        self.config = config
        self.n = config['dataset']['n_students']
        self.df = pd.DataFrame()
        self.df['student_id'] = [f"STU{10000000 + i:08d}" for i in range(self.n)]

    def generate_demographics(self):
        cfg = self.config['demographics']
        # gender
        genders = list(cfg['gender'].keys())
        probs = list(cfg['gender'].values())
        self.df['gender'] = np.random.choice(genders, size=self.n, p=probs)

        # age (truncated normal)
        age = truncated_normal(cfg['age']['mean'], cfg['age']['std'],
                               cfg['age']['min'], cfg['age']['max'], self.n)
        self.df['age'] = np.round(age).astype(int)

        # category
        cats = list(cfg['category'].keys())
        cat_probs = list(cfg['category'].values())
        self.df['category'] = np.random.choice(cats, size=self.n, p=cat_probs)

        # state
        states = [s['name'] for s in cfg['state']]
        state_weights = [s['weight'] for s in cfg['state']]
        state_weights = np.array(state_weights) / sum(state_weights)
        self.df['state'] = np.random.choice(states, size=self.n, p=state_weights)

        # urban_rural
        ur = list(cfg['urban_rural'].keys())
        ur_probs = list(cfg['urban_rural'].values())
        self.df['urban_rural'] = np.random.choice(ur, size=self.n, p=ur_probs)

    def generate_academic(self):
        cfg = self.config['academic']
        # board_type
        boards = list(cfg['board_type'].keys())
        board_probs = list(cfg['board_type'].values())
        self.df['board_type'] = np.random.choice(boards, size=self.n, p=board_probs)

        # class_10 percentage with board effect
        base_mean = cfg['class_10']['mean']
        base_std = cfg['class_10']['std']
        # generate raw scores from truncated normal (will add board effect after)
        raw_10 = truncated_normal(base_mean, base_std,
                                  cfg['class_10']['min'], cfg['class_10']['max'], self.n)
        # add board effect
        board_effect = self.df['board_type'].map(cfg['class_10']['board_effects']).fillna(0)
        class_10 = raw_10 + board_effect.values
        class_10 = np.clip(class_10, cfg['class_10']['min'], cfg['class_10']['max'])
        self.df['class_10_percentage'] = np.round(class_10, 1)

        # class_12 correlated with class_10
        corr = cfg['class_12']['corr_with_10']
        mean_10 = self.df['class_10_percentage'].mean()
        std_10 = self.df['class_10_percentage'].std()
        mean_12 = mean_10 + cfg['class_12']['mean_offset']
        std_12 = cfg['class_12']['std']

        # generate using multivariate normal
        cov = [[std_10**2, corr * std_10 * std_12],
               [corr * std_10 * std_12, std_12**2]]
        means = [mean_10, mean_12]
        # generate per row? Better: use conditional distribution: class_12 = mean_12 + corr*(std_12/std_10)*(class_10 - mean_10) + noise
        noise_std = std_12 * np.sqrt(1 - corr**2)
        class_12 = mean_12 + corr * (std_12 / std_10) * (self.df['class_10_percentage'] - mean_10) + np.random.normal(0, noise_std, self.n)
        # add board effect
        class_12 += self.df['board_type'].map(cfg['class_12']['board_effects']).fillna(0)
        class_12 = np.clip(class_12, cfg['class_12']['min'], cfg['class_12']['max'])
        self.df['class_12_percentage'] = np.round(class_12, 1)

        # stream based on class_10
        def assign_stream(row):
            p = np.random.random()
            base = cfg['stream']['base_prob']
            low_thresh = cfg['stream']['class_10_thresholds']['low']
            high_thresh = cfg['stream']['class_10_thresholds']['high']
            if row['class_10_percentage'] >= high_thresh:
                # higher chance of Science
                prob_science = min(0.8, base['Science'] + 0.2)
                prob_commerce = base['Commerce'] - 0.05
                prob_arts = base['Arts'] - 0.05
            elif row['class_10_percentage'] <= low_thresh:
                prob_science = base['Science'] - 0.1
                prob_commerce = base['Commerce']
                prob_arts = base['Arts'] + 0.1
            else:
                prob_science = base['Science']
                prob_commerce = base['Commerce']
                prob_arts = base['Arts']
            # normalize
            total = prob_science + prob_commerce + prob_arts
            probs = [prob_science/total, prob_commerce/total, prob_arts/total]
            return np.random.choice(['Science', 'Commerce', 'Arts'], p=probs)

        self.df['stream'] = self.df.apply(assign_stream, axis=1)

        # subject scores: math, science, english (0-100)
        # They are correlated with class_12 and each other, and depend on stream.
        # We'll generate multivariate normal for each stream.
        subject_corr = 0.6  # between math, science, english
        std_vals = cfg['stream']['subject_scores']['std']
        # For each stream, generate mean vector and covariance
        means_stream = cfg['stream']['subject_scores']
        math_science = []
        math_english = []
        science_english = []
        for i in range(self.n):
            stream = self.df.loc[i, 'stream']
            mean_math, mean_sci, mean_eng = means_stream[stream]
            # Adjust based on class_12 (higher class_12 => higher subject scores)
            # Use linear shift: subject_score = mean + 0.5*(class_12 - overall_mean_12) + noise
            class_12_i = self.df.loc[i, 'class_12_percentage']
            overall_mean_12 = self.df['class_12_percentage'].mean()
            # covariance matrix
            cov = [[std_vals[0]**2, subject_corr*std_vals[0]*std_vals[1], subject_corr*std_vals[0]*std_vals[2]],
                   [subject_corr*std_vals[0]*std_vals[1], std_vals[1]**2, subject_corr*std_vals[1]*std_vals[2]],
                   [subject_corr*std_vals[0]*std_vals[2], subject_corr*std_vals[1]*std_vals[2], std_vals[2]**2]]
            # generate sample
            sample = np.random.multivariate_normal([mean_math, mean_sci, mean_eng], cov)
            # shift by class_12 effect
            sample += 0.3 * (class_12_i - overall_mean_12)
            sample = np.clip(sample, 0, 100)
            math_science.append(sample[0])
            math_english.append(sample[1])  # careful: order [math, science, english]
            science_english.append(sample[2])
        self.df['math_score'] = np.round(math_science, 1)
        self.df['science_score'] = np.round(math_english, 1)
        self.df['english_score'] = np.round(science_english, 1)

        # medium_of_instruction
        # base probability of English, then adjust by factors
        base_eng_prob = cfg['medium_of_instruction']['English']['base_prob']
        factors = cfg['medium_of_instruction']['English']['factors']
        probs_eng = []
        for i in range(self.n):
            prob = base_eng_prob
            if self.df.loc[i, 'urban_rural'] == 'Urban':
                prob *= factors.get('urban_rural_Urban', 1)
            if self.df.loc[i, 'board_type'] in factors:
                prob *= factors[self.df.loc[i, 'board_type']]
            probs_eng.append(min(prob, 1.0))
        self.df['medium_of_instruction'] = [np.random.choice(['English', 'Regional'], p=[p, 1-p]) for p in probs_eng]

        # gap_year
        gap_probs = cfg['gap_year']['probs']
        self.df['gap_year'] = np.random.choice([0,1,2,3], size=self.n, p=gap_probs)

    def generate_entrance_aptitude(self):
        cfg = self.config['entrance_aptitude']
        # logical_reasoning correlated with math
        corr = cfg['logical_reasoning']['corr_with_math']
        mean_lr = cfg['logical_reasoning']['mean']
        std_lr = cfg['logical_reasoning']['std']
        # conditional on math_score
        math = self.df['math_score'].values
        mean_math = math.mean()
        std_math = math.std()
        lr = mean_lr + corr * (std_lr / std_math) * (math - mean_math) + np.random.normal(0, std_lr * np.sqrt(1-corr**2), self.n)
        lr = np.clip(lr, cfg['logical_reasoning']['min'], cfg['logical_reasoning']['max'])
        self.df['logical_reasoning_score'] = np.round(lr, 1)

        # language_proficiency correlated with english_score
        corr_lang = cfg['language_proficiency']['corr_with_english']
        mean_lang = cfg['language_proficiency']['mean']
        std_lang = cfg['language_proficiency']['std']
        english = self.df['english_score'].values
        mean_eng = english.mean()
        std_eng = english.std()
        lang = mean_lang + corr_lang * (std_lang / std_eng) * (english - mean_eng) + np.random.normal(0, std_lang * np.sqrt(1-corr_lang**2), self.n)
        lang = np.clip(lang, cfg['language_proficiency']['min'], cfg['language_proficiency']['max'])
        self.df['language_proficiency_score'] = np.round(lang, 1)

        # entrance_exam_score = w1*class_12 + w2*socio_effect + noise
        w_class = cfg['entrance_exam']['weight_class12']
        w_socio = cfg['entrance_exam']['weight_socio']
        class12 = self.df['class_12_percentage'].values
        # socio effect placeholder (will be updated later with actual socio features)
        # for now, generate random socio effect based on urban and income proxies (but we don't have them yet)
        # We'll approximate using urban_rural and maybe later adjust
        socio_proxy = (self.df['urban_rural'] == 'Urban').astype(int) * 5  # just an example
        entrance = w_class * class12 + w_socio * socio_proxy + np.random.normal(0, cfg['entrance_exam']['noise_std'], self.n)
        entrance = np.clip(entrance, cfg['entrance_exam']['min'], cfg['entrance_exam']['max'])
        self.df['entrance_exam_score'] = np.round(entrance, 1)

        # aptitude_score = w1*class12 + w2*logical + w3*language + noise
        w_c12 = cfg['aptitude']['weight_class12']
        w_log = cfg['aptitude']['weight_logical']
        w_lang = cfg['aptitude']['weight_language']
        aptitude = w_c12 * class12 + w_log * self.df['logical_reasoning_score'].values + w_lang * self.df['language_proficiency_score'].values + np.random.normal(0, cfg['aptitude']['noise_std'], self.n)
        aptitude = np.clip(aptitude, cfg['aptitude']['min'], cfg['aptitude']['max'])
        self.df['aptitude_score'] = np.round(aptitude, 1)

    def generate_socio_economic(self):
        cfg = self.config['socio_economic']
        # family_income: base probs, adjusted by urban_rural
        income_cats = cfg['family_income']['categories']
        base_probs = cfg['family_income']['probs']
        urban_effect = cfg['family_income']['urban_effect']
        income = []
        for i in range(self.n):
            if self.df.loc[i, 'urban_rural'] == 'Urban':
                probs = urban_effect
            else:
                probs = base_probs
            income.append(np.random.choice(income_cats, p=probs))
        self.df['family_income_range'] = income

        # parent_education: correlated with income
        edu_levels = cfg['parent_education']['levels']
        edu_probs = cfg['parent_education']['probs']
        # create mapping from income to shift
        # higher income -> higher education probability
        income_order = {cat: i for i, cat in enumerate(income_cats)}
        income_num = self.df['family_income_range'].map(income_order).values
        # shift probability towards higher education for higher income
        edu = []
        for i in range(self.n):
            probs = np.array(edu_probs)
            shift = income_num[i] - 1.5  # range -1.5 to 1.5 approx
            # shift mass to the right for positive shift
            if shift > 0:
                for j in range(len(probs)-1, 0, -1):
                    probs[j] += shift * 0.1 * probs[j-1]  # simplistic
                    probs[j-1] -= shift * 0.1 * probs[j-1]
            elif shift < 0:
                for j in range(len(probs)-1):
                    probs[j] += abs(shift) * 0.1 * probs[j+1]
                    probs[j+1] -= abs(shift) * 0.1 * probs[j+1]
            probs = np.clip(probs, 0, 1)
            probs /= probs.sum()
            edu.append(np.random.choice(edu_levels, p=probs))
        self.df['parent_education_level'] = edu

        # first_generation_learner based on parent_education
        def is_first_gen(row):
            edu = row['parent_education_level']
            base = cfg['first_generation']['base_prob']
            factor = cfg['first_generation']['edu_factor'].get(edu, 0.2)
            prob = base * factor
            return np.random.choice(['Yes', 'No'], p=[prob, 1-prob])
        self.df['first_generation_learner'] = self.df.apply(is_first_gen, axis=1)

        # internet_access
        def has_internet(row):
            base = cfg['internet_access']['base_prob']
            if row['urban_rural'] == 'Urban':
                base *= cfg['internet_access']['urban_factor']
            inc_factor = cfg['internet_access']['income_factor'].get(row['family_income_range'], 1)
            prob = base * inc_factor
            prob = min(prob, 1.0)
            return np.random.choice(['Yes', 'No'], p=[prob, 1-prob])
        self.df['internet_access'] = self.df.apply(has_internet, axis=1)

        # study_resources_index: beta distribution, mean shifted by income
        alpha = cfg['study_resources_index']['alpha']
        beta = cfg['study_resources_index']['beta']
        income_effect = cfg['study_resources_index']['income_effect']
        # base mean of beta: alpha/(alpha+beta)
        base_mean = alpha / (alpha + beta)
        # adjust per row based on income numeric
        income_num_norm = income_num / 3  # map 0-3 to 0-1
        shift = income_effect * (income_num_norm - base_mean)
        # new mean for each row, then sample from beta with appropriate parameters? Hard. We'll sample from beta with fixed alpha,beta and then shift linearly.
        raw = np.random.beta(alpha, beta, self.n)
        # shift and clip
        study_res = raw + shift
        study_res = np.clip(study_res, 0, 1)
        self.df['study_resources_index'] = np.round(study_res, 3)

    def generate_behavioral(self):
        cfg = self.config['behavioral']
        # self_learning_index
        alpha = cfg['self_learning_index']['alpha']
        beta = cfg['self_learning_index']['beta']
        self.df['self_learning_index'] = np.round(np.random.beta(alpha, beta, self.n), 3)

        # motivation_score
        mean = cfg['motivation_score']['mean']
        std = cfg['motivation_score']['std']
        low = cfg['motivation_score']['min']
        high = cfg['motivation_score']['max']
        mot = truncated_normal(mean, std, low, high, self.n)
        if cfg['motivation_score']['integer']:
            mot = np.round(mot).astype(int)
        self.df['motivation_score'] = mot

        # attendance_likelihood
        alpha = cfg['attendance_likelihood']['alpha']
        beta = cfg['attendance_likelihood']['beta']
        self.df['attendance_likelihood'] = np.round(np.random.beta(alpha, beta, self.n), 3)

    def inject_missingness(self):
        cfg = self.config['missingness']
        # MCAR
        mcar_rate = cfg['mcar_rate']
        mask = np.random.random(self.n) < mcar_rate
        cols = self.df.select_dtypes(include=[np.number]).columns
        for col in cols:
            self.df.loc[mask, col] = np.nan

        # MAR patterns
        for pattern in cfg['mar_patterns']:
            col = pattern['column']
            condition = pattern['condition']
            missing_rate = pattern['missing_rate']
            # evaluate condition
            subset = self.df.query(condition).index
            mask = np.random.random(len(subset)) < missing_rate
            self.df.loc[subset[mask], col] = np.nan

    def generate(self):
        self.generate_demographics()
        self.generate_academic()
        self.generate_entrance_aptitude()
        self.generate_socio_economic()
        self.generate_behavioral()
        self.inject_missingness()
        return self.df