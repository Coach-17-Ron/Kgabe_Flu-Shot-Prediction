# Load the data
train_features = pd.read_csv('training_set_features.csv')
train_labels = pd.read_csv('training_set_labels.csv')
test_features = pd.read_csv('test_set_features.csv')

# Merge features and labels for exploration
full_train = pd.merge(train_features, train_labels, on='respondent_id')

# Basic exploration
print("Training features shape:", train_features.shape)
print("Training labels shape:", train_labels.shape)
print("Test features shape:", test_features.shape)

# Check for missing values
print("\nMissing values in training set:")
print(train_features.isnull().sum().sort_values(ascending=False))

# Examine target variable distribution
print("\nTarget variable distribution:")
print(train_labels[['h1n1_vaccine', 'seasonal_vaccine']].mean())


# Data Preprocessing 
# We have to handle:
  #Missing values
  #Categorical encoding
  #Feature scaling

# Define numeric and categorical features
numeric_features = ['h1n1_concern', 'h1n1_knowledge', 
                   'behavioral_antiviral_meds', 'behavioral_avoidance',
                   'behavioral_face_mask', 'behavioral_wash_hands',
                   'behavioral_large_gatherings', 'behavioral_outside_home',
                   'behavioral_touch_face', 'doctor_recc_h1n1',
                   'doctor_recc_seasonal', 'chronic_med_condition',
                   'child_under_6_months', 'health_worker',
                   'health_insurance', 'opinion_h1n1_vacc_effective',
                   'opinion_h1n1_risk', 'opinion_h1n1_sick_from_vacc',
                   'opinion_seas_vacc_effective', 'opinion_seas_risk',
                   'opinion_seas_sick_from_vacc', 'household_adults',
                   'household_children']

categorical_features = ['age_group', 'education', 'race', 'sex',
                      'income_poverty', 'marital_status', 'rent_or_own',
                      'employment_status', 'hhs_geo_region', 'census_msa',
                      'employment_industry', 'employment_occupation']

# Create preprocessing pipelines
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)])