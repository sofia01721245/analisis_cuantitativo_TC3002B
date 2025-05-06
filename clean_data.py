import pandas as pd

df = pd.read_csv('data.tsv', sep='\t', engine='python')


# Define a function to get the graduation year or -1 for each student
def compute_graduation_year(group):
    if group['student.isGraduated'].any():
        # Get the term.desc of the first row where the student graduated
        return group.loc[group['student.isGraduated'] == 1, 'term.desc'].iloc[0]
    else:
        return -1

# Create a Series with graduation year per student.id
graduation_years = df.groupby('student.id').apply(compute_graduation_year).rename('graduation_year')

# Merge this Series back into the original DataFrame on student.id
df = df.merge(graduation_years, on='student.id')

# Drop the columns you don't need (e.g., 'subject_LiFE.portfolioCategory')
df = df.drop(columns=['subject_LiFE.portfolioCategory'])
# Drop the specified column
df = df.drop(columns=['subject_LiFE.portfolioClassification'])

# Remove the student because she still has not graduated and can bias results
df = df[df['student.id'] != 3621]

# Save the cleaned DataFrame to a new CSV file
df.to_csv('cleaned_data.csv', index=False)

# Remove duplicates based on student.id and term.desc, keeping the first row per term
df = df.drop_duplicates(subset=['student.id', 'term.desc'])

# Done: df now has a new column 'graduation_year' for each row
print(df[['student.id', 'term.desc', 'student.isGraduated', 'graduation_year']])

from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
zone_mapping = {'Urban': 0, 'Semiurban': 1, 'Rural': 2, 'No information': -1}
df['student_permAddress.zone_type'] = df['student_permAddress.zone_type'].map(zone_mapping)

# Define a mapping for the age ranges
age_mapping = {'18 and below': 0, '19 to 21': 1,'22 and above': 2}
# Apply the mapping
df['student.age'] = df['student.age'].map(age_mapping)

#Remove no information from student.isForeign
df['student.isForeign'] = df['student.isForeign'].astype(str).str.strip().str.title()
foreign = {'0': 0, '1': 1, 'No Information': -1}
df['student.isForeign'] = df['student.isForeign'].map(foreign)

#Remove no information from student.isForeign
df['student.isFirstGeneration'] = df['student.isFirstGeneration'].astype(str).str.strip().str.title()
firstGen = {'No': 0, 'Yes': 1, 'No Information': -1}
df['student.isFirstGeneration'] = df['student.isFirstGeneration'].map(firstGen)

#Drop test type because its the same for all students
df = df.drop(columns=['student_admission_test.type_desc'])

df['student_admission_test_disc.dominance_score'] = df['student_admission_test_disc.dominance_score'].replace('Does not apply', -1)
df['student_admission_test_disc.influence_score'] = df['student_admission_test_disc.influence_score'].replace('Does not apply', -1)
df['student_admission_test_disc.conscientiousness_score'] = df['student_admission_test_disc.conscientiousness_score'].replace('Does not apply', -1)
df['student_admission_test_disc.steadiness_score'] = df['student_admission_test_disc.steadiness_score'].replace('Does not apply', -1)
df['student_admission_test_valuesIndex.aesthetic_score'] = df['student_admission_test_valuesIndex.aesthetic_score'].replace('Does not apply', -1)
df['student_admission_test_valuesIndex.economic_score'] = df['student_admission_test_valuesIndex.economic_score'].replace('Does not apply', -1)
df['student_admission_test_valuesIndex.individualistic_score'] = df['student_admission_test_valuesIndex.individualistic_score'].replace('Does not apply', -1)
df['student_admission_test_valuesIndex.political_score'] = df['student_admission_test_valuesIndex.political_score'].replace('Does not apply', -1)
df['student_admission_test_valuesIndex.altruistic_score'] = df['student_admission_test_valuesIndex.altruistic_score'].replace('Does not apply', -1)
df['student_admission_test_valuesIndex.regulatory_score'] = df['student_admission_test_valuesIndex.regulatory_score'].replace('Does not apply', -1)
df['student_admission_test_valuesIndex.theoretical_score'] = df['student_admission_test_valuesIndex.theoretical_score'].replace('Does not apply', -1)




#Empty colums probably mean the student did not participate in that term
df = df[df['student.fte'].notna() & (df['student.fte'] != '')]
df = df[df['student.term_gpa'].notna() & (df['student.term_gpa'] != '')]

#Plot status in acending order
df['student.status_academic_desc'] = df['student.status_academic_desc'].astype(str).str.strip().str.title()
academicDesc = {
    'Academic Support, Failed >=10 Courses Before 50% Of The Academic Program': 0,
    'Academic Support, Failed >=2 Courses In Each Of Last 3 Semesters': 1,
    'Academic Support, Failed >=3 Courses In Each Of Last 2 Semesters': 2,
    'Conditioned Student, Failed >=6  Courses Before 50% Of Total Units Of The Academic Program': 3,
    'Conditioned Student, Failed >=3 Courses In The Last Completed Semester': 4,
    'Conditioned Student, Failed 2 Courses In Each Of The Last 2 Completed Semesters': 5,
    'Conditioned Student, Failed 1 Or 2 Courses After Previously Being Conditioned Student': 6,
    'Regular Student': 7,
    'No Status Information': -1     
}
df['student.status_academic_desc'] = df['student.status_academic_desc'].map(academicDesc)

#Drop columns where data is insignificant
df.drop([
    'student_admission_socialProject.type',
    'student_admission_socialProject.scope',
    'student_admission_cv.sports_level',
    'student_admission_cv.cultural_level',
    'student_admission_cv.student_level',
    'student_admission_cv.community_level',
    'student_admission_cv.leadership_level',
    'student_admission_cv.work_level',
    'student_admission_cv.academic_level',
    'student_admission_cv.international_level'
], axis=1, inplace=True)

# Save the cleaned DataFrame to a new CSV files
df.to_csv('cleaned_data.csv', index=False)

# Sort by student ID and term
df_sorted = df.sort_values(['student.id', 'term.desc'])

# Get latest semester per student
df_latest = df_sorted.groupby('student.id').tail(1)

# Aggregate semester-based features
agg_df = df.groupby('student.id').agg({
    'student.fte': 'mean',
    'student.term_gpa': ['mean', 'max', 'min', 'last'],
    'student.term_gpa_program': ['mean', 'max', 'min', 'last'],
    'student.status_academic_desc': 'last',
    'student.isGraduated': 'max',
    'graduation_year': 'max'
})
agg_df.columns = ['_'.join(col).strip() for col in agg_df.columns.values]
agg_df.reset_index(inplace=True)

# Define static columns
static_cols = [
    'student.id', 'student.gender_desc', 'student.age', 'student_permAddress.zone_type',
    'student_originSchool.gpa', 'student_originSchool.isITESM', 'student.isForeign',
    'student.isFirstGeneration', 'student_admission_test.score', 
    'student_admission_test_disc.dominance_score', 'student_admission_test_disc.influence_score',
    'student_admission_test_disc.steadiness_score', 'student_admission_test_disc.conscientiousness_score',
    'student_admission_test_valuesIndex.aesthetic_score', 'student_admission_test_valuesIndex.economic_score',
    'student_admission_test_valuesIndex.individualistic_score', 'student_admission_test_valuesIndex.political_score',
    'student_admission_test_valuesIndex.altruistic_score', 'student_admission_test_valuesIndex.regulatory_score',
    'student_admission_test_valuesIndex.theoretical_score',
    'student.cohort_id', 'student.isTec21',
    'mainCampus.region_code', 'program.school_desc', 'program.major_id'
]

# Drop duplicate static records
df_static = df[static_cols].drop_duplicates(subset='student.id')

# Merge aggregated data with static features
df_final = pd.merge(agg_df, df_static, on='student.id', how='left')

# Save the cleaned DataFrame to a new CSV files
df_final.to_csv('cleaned_data_per_student.csv', index=False)