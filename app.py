import os
import pickle
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from streamlit_option_menu import option_menu
import streamlit.components.v1 as components

# Set page configuration
st.set_page_config(page_title="Diabetes and Heart Diseases Prediction",
                   layout="wide",
                   page_icon="🧑‍⚕️")

# Load the saved models
working_dir = os.path.dirname(os.path.abspath(__file__))
diabetes_model = pickle.load(open(f'{working_dir}/saved_models/diabetes_model.sav', 'rb'))
heart_disease_model = pickle.load(open(f'{working_dir}/saved_models/heart_disease_model.sav', 'rb'))

# Check if CSV files exist, if not, create them
diabetes_csv_path = f'{working_dir}/Saved_data from user/diabetes_data.csv'
if not os.path.exists(diabetes_csv_path):
    df_diabetes = pd.DataFrame(columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
                                        'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome'])
    df_diabetes.to_csv(diabetes_csv_path, index=False)

heart_csv_path = f'{working_dir}/Saved_data from user/heart_data.csv'
if not os.path.exists(heart_csv_path):
    df_heart = pd.DataFrame(columns=['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach',
                                      'exang', 'oldpeak', 'slope', 'ca', 'thal', 'Outcome'])
    df_heart.to_csv(heart_csv_path, index=False)

# Initialize session state for login status
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

# Sidebar for navigation
with st.sidebar:
    selected = option_menu('Diabetes & Heart Prediction System',
                           ['Diabetes Prediction', 'Heart Disease Prediction','Statistics'],
                           menu_icon='hospital-fill',
                           icons=['activity', 'heart', 'bar-chart'],
                           default_index=0)
                           
  # Embed the chatbot in the sidebar
    components.html("""
    <script>
    window.embeddedChatbotConfig = {
    chatbotId: "ovklNOH-QGI6hN3O3jucQ",
    domain: "www.chatbase.co"
    }
    </script>
    <script
    src="https://www.chatbase.co/embed.min.js"
    chatbotId="ovklNOH-QGI6hN3O3jucQ"
    domain="www.chatbase.co"
    defer>
    </script>
    """, height=400)                           

# Automatically log out when switching to a different page
if selected != 'Statistics':
    st.session_state.logged_in = False

# Define the valid ranges for each parameter
valid_ranges = {
    'Pregnancies': (0, 20),
    'Glucose': (70, 200),
    'BloodPressure': (50, 250),
    'SkinThickness': (0.5, 99),
    'Insulin': (0, 300),
    'BMI': (10, 70),
    'DiabetesPedigreeFunction': (0.08, 2.5),
    'Age': (1, 110),
    'age': (1, 110),
    'sex': (0, 1),
    'cp': (0, 3),
    'trestbps': (50, 250),
    'chol': (100, 410),
    'fbs': (0, 1),
    'restecg': (0, 2),
    'thalach': (50, 300),
    'exang': (0, 1),
    'oldpeak': (0, 10),
    'slope': (0, 2),
    'ca': (0, 4),
    'thal': (0, 2)
}

# Function to validate user inputs
def validate_input(param, value):
    min_val, max_val = valid_ranges[param]
    if min_val <= value <= max_val:
        return True
    else:
        return False



# Diabetes Prediction Page
if selected == 'Diabetes Prediction':
    st.title('Diabetes Prediction using ML')

    col1, col2, col3 = st.columns(3)
    with col1:
        Pregnancies = st.text_input('Number of Pregnancies (0 For Men)')
    with col2:
        Glucose = st.text_input('Glucose Level (mg/dL, 70-100)')
    with col3:
        BloodPressure = st.text_input('Blood Pressure (mmHg, 80-120)')
    with col1:
        SkinThickness = st.text_input('Skin Thickness (mm, 0.5-15)')
    with col2:
        Insulin = st.text_input('Insulin Level (mg/dL, <90)')
    with col3:
        BMI = st.text_input('Body Mass Index (BMI, 18.5-27.5)')
    with col1:
        DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function (0.08-0.5)')
    with col2:
        Age = st.text_input('Age of the Person')

    # Prediction code
    diab_diagnosis = ''
    if st.button('Diabetes Test Result'):
        try:
            user_input = [float(Pregnancies), float(Glucose), float(BloodPressure), float(SkinThickness), float(Insulin),
                          float(BMI), float(DiabetesPedigreeFunction), float(Age)]
            if all(validate_input(param, value) for param, value in zip(['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'], user_input)):
                diab_prediction = diabetes_model.predict([user_input])

                if diab_prediction[0] == 1:
                    diab_diagnosis = 'The person is diabetic'
                    outcome = 1
                    st.info("**Health Tips for Managing Diabetes:**")
                    st.write("""
                    - Exercise regularly to help manage blood sugar levels.
                    - Eat a balanced diet rich in fiber, fruits, and vegetables.
                    - Monitor your blood sugar levels regularly.
                    - Stay hydrated by drinking plenty of water.
                    - Avoid sugary foods and beverages.
                    """)
                else:
                    diab_diagnosis = 'The person is not diabetic'
                    outcome = 0

                # Save the user input and diagnosis to the CSV file
                df_diabetes = pd.read_csv(diabetes_csv_path)
                new_entry = pd.DataFrame([user_input + [outcome]], columns=df_diabetes.columns)
                df_diabetes = pd.concat([df_diabetes, new_entry], ignore_index=True)
                df_diabetes.to_csv(diabetes_csv_path, index=False)

                st.success(diab_diagnosis)
            else:
                st.error('Please enter values within the valid range for each parameter.')
        except ValueError:
            st.error('Please enter valid numeric values for all inputs.')

# Heart Disease Prediction Page
if selected == 'Heart Disease Prediction':
    st.title('Heart Disease Prediction using ML')

    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.text_input('Age')
    with col2:
        sex = st.text_input('Sex (0 for male, 1 for female)')
    with col3:
        cp = st.text_input('Chest Pain Types (0: Typical Angina, 1: Atypical Angina, 2: Non-anginal Pain, 3: Asymptomatic)')
    with col1:
        trestbps = st.text_input('Resting Blood Pressure (mmHg, 90-120)')
    with col2:
        chol = st.text_input('Serum Cholesterol (mg/dL, 190-270)')
    with col3:
        fbs = st.text_input('Fasting Blood Sugar (1 if >120 mg/dL, else 0)')
    with col1:
        restecg = st.text_input('Resting Electrocardiographic Results (0: Normal, 1: Having ST-T wave abnormality, 2: Showing probable or definite left ventricular hypertrophy)')
    with col2:
        thalach = st.text_input('Maximum Heart Rate Achieved')
    with col3:
        exang = st.text_input('Exercise Induced Angina (1: Yes, 0: No)')
    with col1:
        oldpeak = st.text_input('ST Depression Induced by Exercise (0-6)')
    with col2:
        slope = st.text_input('Slope of the Peak Exercise ST Segment (0: Upsloping, 1: Flat, 2: Downsloping)')
    with col3:
        ca = st.text_input('Number of Major Vessels Colored by Fluoroscopy (0-4)')
    with col1:
        thal = st.text_input('Thalassemia (0: Normal, 1: Fixed Defect, 2: Reversible Defect)')

    # Prediction code
    heart_diagnosis = ''
    if st.button('Heart Disease Test Result'):
        try:
            user_input = [float(age), float(sex), float(cp), float(trestbps), float(chol), float(fbs), float(restecg),
                          float(thalach), float(exang), float(oldpeak), float(slope), float(ca), float(thal)]
            if all(validate_input(param, value) for param, value in zip(['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'], user_input)):
                heart_prediction = heart_disease_model.predict([user_input])

                if heart_prediction[0] == 1:
                    heart_diagnosis = 'The person is having heart disease'
                    outcome = 1
                    st.info("**Health Tips for Managing Heart Disease:**")
                    st.write("""
                    - Exercise regularly to strengthen your heart and improve circulation.
                    - Eat a heart-healthy diet that includes plenty of fruits, vegetables, and whole grains.
                    - Avoid smoking and limit alcohol consumption.
                    - Manage stress through relaxation techniques such as meditation or yoga.
                    - Maintain a healthy weight and monitor your blood pressure regularly.
                    """)
                else:
                    heart_diagnosis = 'The person does not have any heart disease'
                    outcome = 0

                # Save the user input and diagnosis to the CSV file
                df_heart = pd.read_csv(heart_csv_path)
                new_entry = pd.DataFrame([user_input + [outcome]], columns=df_heart.columns)
                df_heart = pd.concat([df_heart, new_entry], ignore_index=True)
                df_heart.to_csv(heart_csv_path, index=False)

                st.success(heart_diagnosis)
            else:
                st.error('Please enter values within the valid range for each parameter.')
        except ValueError:
            st.error('Please enter valid numeric values for all inputs.')



# Statistics Page with login requirement
if selected == 'Statistics':
    if not st.session_state.logged_in:
        st.title('Login')

        username = st.text_input('Username')
        password = st.text_input('Password', type='password')
        login_button = st.button('Login')

        if login_button:
            if username == 'admin' and password == '12345':
                st.session_state.logged_in = True
                st.rerun()  # Reload the page to display statistics after login
            else:
                st.error('Incorrect username or password')
    else:
        st.title('Statistics and Visualizations')



        # Load the datasets
        df_diabetes = pd.read_csv('E:/BCA/PROJECT(SIXTH)/MDP/dataset/diabetes.csv')
        df_heart = pd.read_csv('E:/BCA/PROJECT(SIXTH)/MDP/dataset/heart.csv')

        # Check if the DataFrames are empty
        if df_diabetes.empty or df_heart.empty:
            st.warning("No data available to display statistics. Please enter predictions in the other sections first.")
        else:
            # Filter for age > 40 and BP > 70 from diabetes data
            df_diabetes_filtered = df_diabetes[(df_diabetes['Age'] > 40) & (df_diabetes['BloodPressure'] > 70)]
            df_heart_filtered = df_heart[(df_heart['age'] > 40) & (df_heart['trestbps'] > 140)]

            # Check if the filtered DataFrames are empty
            if df_diabetes_filtered.empty or df_heart_filtered.empty:
                st.warning("No data found for people over age 40 with blood pressure above 140.")
            else:
                st.write("## Diabetes Data Statistics")
                st.write(df_diabetes_filtered.describe())

                st.write("## Heart Disease Data Statistics")
                st.write(df_heart_filtered.describe())

                # Bar chart for diabetes outcomes
                fig, ax = plt.subplots()
                sns.countplot(data=df_diabetes_filtered, x='Outcome', ax=ax)
                ax.set_title('Diabetes Outcomes for Age > 40 and BP > 140')
                st.pyplot(fig)

                # Bar chart for heart disease outcomes
                fig, ax = plt.subplots()
                sns.countplot(data=df_heart_filtered, x='target', ax=ax)
                ax.set_title('Heart Disease Outcomes for Age > 40 and BP > 140')
                st.pyplot(fig)

                
                # Filter for people with heart disease from diabetes filtered data
                df_diabetes_heart_disease = df_diabetes_filtered[df_diabetes_filtered['Outcome'] == 1]

                # Calculate the ratio of people with heart disease from diabetes
                heart_disease_ratio = (len(df_diabetes_heart_disease) / len(df_diabetes_filtered)) * 100

                # Display the heart disease ratio as a percentage
                st.write("Percentage of people with heart disease from diabetes:")
                st.write(f"{heart_disease_ratio:.2f}%")

                st.write("### Common Heart Diseases")
                st.write("""
                1. Coronary Artery Disease (CAD)
                2. Hypertensive Heart Disease
                3. Heart Failure
                4. Arrhythmias
                5. Cardiomyopathies
                6. Valvular Heart Disease
                """)


