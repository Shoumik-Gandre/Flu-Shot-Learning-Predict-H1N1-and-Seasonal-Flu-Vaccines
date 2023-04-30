from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from category_encoders import OrdinalEncoder

# Column Types
categorical_columns = [
    'race',
    'sex',
    'marital_status',
    'employment_status',
    'hhs_geo_region',
    'census_msa',
    'employment_industry',
    'employment_occupation',
    'rent_or_own'
]

ordinal_numeric_columns = [
    'h1n1_concern',
    'h1n1_knowledge',
    'opinion_h1n1_vacc_effective',
    'opinion_h1n1_risk',
    'opinion_h1n1_sick_from_vacc',
    'opinion_seas_vacc_effective',
    'opinion_seas_risk',
    'opinion_seas_sick_from_vacc',
    'household_adults',
    'household_children',
]

ordinal_object_columns = [
    'age_group',
    'education',
    'income_poverty',
]

boolean_columns = [
    'behavioral_antiviral_meds',
    'behavioral_avoidance', 
    'behavioral_face_mask', 
    'behavioral_wash_hands',
    'behavioral_large_gatherings', 
    'behavioral_outside_home',
    'behavioral_touch_face', 
    'doctor_recc_h1n1', 
    'doctor_recc_seasonal',
    'chronic_med_condition', 
    'child_under_6_months', 
    'health_worker',
    'health_insurance',
]

# Ordinal Mapping
ordinal_mapping = [
    {
        'col': 'age_group',
        'mapping': {
            '18 - 34 Years': 0, 
            '35 - 44 Years': 1, 
            '45 - 54 Years': 2,
            '55 - 64 Years': 3, 
            '65+ Years': 4,
        }
    },
    {
        'col': 'education',
        'mapping': {
            '< 12 Years': 0, 
            '12 Years': 1, 
            'College Graduate': 2, 
            'Some College': 3
        }
    },
    {
        'col': 'income_poverty',
        'mapping': {
            'Below Poverty': 0, 
            '<= $75,000, Above Poverty': 1,
            '> $75,000': 2
        }
    }
]

category_preprocessor = make_pipeline(
    OneHotEncoder(drop='first'),
)

ordinal_object_preprocessor = make_pipeline(
    OrdinalEncoder(mapping=ordinal_mapping),
    SimpleImputer(strategy='constant', fill_value=-1),
)

preprocessor = ColumnTransformer(
    transformers=[
        ('boolean_imputer', SimpleImputer(strategy='mean'), boolean_columns),
        ('ordinal_numeric_imputer', SimpleImputer(strategy='mean'), ordinal_numeric_columns),
        ('ordinal_object_preprocessor', ordinal_object_preprocessor, ordinal_object_columns),
        ('category_preprocessor', category_preprocessor, categorical_columns)
    ],
    remainder='passthrough'
)