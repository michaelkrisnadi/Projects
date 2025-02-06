# To run:
# `streamlit run app_new.py`

import matplotlib.pyplot as plt
import plotly.express as px
import keras_tuner as kt
import streamlit as st
import altair as alt
import pandas as pd
import numpy as np
import tempfile
import warnings
import joblib
import random
import shutil
import time
import os
import re

warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Bidirectional, Dropout
from keras.regularizers import l2
from keras.callbacks import EarlyStopping, TensorBoard
from keras.models import load_model

from scikeras.wrappers import KerasRegressor
from collections import defaultdict
from tqdm import tqdm, tqdm_gui

from join_excel_files import join_files
from individual_display_new import extract_student_row, individual_display_row

# UI
from zipfile import ZipFile
import zipfile
from streamlit_option_menu import option_menu

# Initialize session state
if 'course_code' not in st.session_state:
    st.session_state['course_code'] = ''
if 'academic_session' not in st.session_state:
    st.session_state['academic_session'] = ''
if 'results' not in st.session_state:
    st.session_state['results'] = None
if 'show_tabs' not in st.session_state:
    st.session_state['show_tabs'] = False
if 'course_code_input' not in st.session_state:
    st.session_state['course_code_input'] = ''
if 'results__newdata' not in st.session_state:
    st.session_state['results__newdata'] = None
if 'model_results' not in st.session_state:
    st.session_state['model_results'] = None
if 'intakes_plus_intakes__newdata' not in st.session_state:
    st.session_state['intakes_plus_intakes__newdata'] = None
if 'intakes__newdata' not in st.session_state:
    st.session_state['intakes__newdata'] = None
if 'yearly_individualdisplay_combined' not in st.session_state:
    st.session_state['yearly_individualdisplay_combined'] = None
years = [1, 2, 3, 4]

# Application title
st.title("Program Outcome Enhancement")

# Sidebar navigation
with st.sidebar:
    selected = option_menu(
        menu_title="Navigation", 
        options=[
            "Upload OBE Files",
            "Access OBE Information",
            "Display PO Attainment",
            "Display Individual Student Data",
            "Plot PLO Averages and Categories",
            "Model Training Outputs",
        ],
        icons=[
            "cloud-upload", 
            "info-circle",
            "graph-up",
            "person",
            "bar-chart",
            "gear",
        ],
        menu_icon="cast",     
        default_index=0,  
        styles={
            "nav-link-selected": {"background-color": "#1C83E1"}
        }
    )


# Function to color cell
def highlight_percentages(val):
    try:
        if isinstance(val, str):
            if val.endswith('%'):
                num_val = float(val[:-1])
            elif val == "Strong":
                return "background-color: #E8F9EE;"
            elif val == "Moderate":
                return "background-color: #FFFCE7;" 
            elif val == "Weak":
                return "background-color: #FFECEC;"
            else:
                return ""
        else:
            return ""
    except:
        return ""

    if 80 <= num_val <= 100:
        return "background-color: #E8F9EE;" 
    elif 50 <= num_val < 80:
        return "background-color: #FFFCE7;"
    elif 0 < num_val < 50:
        return "background-color: #FFECEC;" 
    return ""
    
def show_colored_dataframe(df: pd.DataFrame):
    styled_df = df.style.applymap(highlight_percentages)
    st.dataframe(styled_df)

# -----------------------------------------------------
# 1. Process OBE forms

def process_uploaded_file(uploaded_file):
    """
    Processes the uploaded ZIP file by extracting its contents.

    Args:
        uploaded_file (UploadedFile): The uploaded ZIP file from Streamlit.

    Returns:
        extract_folder (str): Path to the extracted folder.
    """
    progress_bar = st.progress(0)
    status_text = st.empty()
    status_text.text("Processing .zip file...")
    # Create a temporary directory to store the ZIP file and its contents
    temp_dir = tempfile.mkdtemp(prefix="extracted_zip_")
    progress_bar.progress(5)

    zip_path = os.path.join(temp_dir, uploaded_file.name)
    
    # Save the uploaded ZIP file to the temporary directory
    with open(zip_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Define the extraction folder path
    extract_folder = os.path.join(temp_dir, uploaded_file.name.replace('.zip', ''))
    progress_bar.progress(10)
    
    # Extract the ZIP file into the extraction folder
    try:
        with ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_folder)
        st.write(f"ZIP file extracted to: {extract_folder}")
    except zipfile.BadZipFile:
        print("Error: The uploaded file is not a valid ZIP archive.")
        shutil.rmtree(temp_dir)  # Clean up temporary directory on failure
        return None
    progress_bar.progress(70)
    
    # Optionally, delete the ZIP file after extraction to save space
    os.remove(zip_path)
    progress_bar.progress(100)

    progress_bar.empty()
    status_text.empty()

    # Return the path to the extracted folder
    return extract_folder


 
def delete_temp_directory(directory_path, retries=3, delay=1): 
    if not os.path.exists(directory_path):
        print(f'Temporary directory "{directory_path}" does not exist.')
        return
    for i in range(retries):
        try:
            shutil.rmtree(directory_path)
            print(f'Temporary directory "directory_path" ({directory_path}) has been deleted.') 
            return
        except PermissionError as e:
            print(f"Attempt {i + 1} failed: {e}")
            time.sleep(delay)
    print(f'Failed to delete temporary directory "directory_path" ({directory_path}) after {retries} attempts.') 

def process_obe_forms(directory_path): 
    file_list = os.listdir(directory_path)
    unsupported_files = []
    results = {}

    # For progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()

    for file_name in file_list:

        # Progress bar
        progress = int((file_list.index(file_name) / len(file_list)) * 100)
        progress_bar.progress(progress)
        status_text.text(f"Processing {file_name}...")

        try:
            results[file_name] = {}
            xls = pd.ExcelFile(os.path.join(directory_path, file_name))

            # Sheet title
            df_info = pd.read_excel(xls, "Setup", nrows=20)
            sheet_title = df_info.at[4, 'Unnamed: 0']
            if sheet_title == 'OBE Measurement Tools v3.1':
                sheet_title = 'OBE Measurement Tools v3.1 - Setup Sheet'
            results[file_name]['sheet_title'] = sheet_title
            
            # Course info
            df_info = pd.read_excel(xls, "Setup", skiprows=range(7), nrows=11)
            course_info = {}
            for index, row in df_info.iterrows():
                indicator = None
                value = None
                for cell in row:
                    if pd.notna(cell):
                        if indicator is None:
                            indicator = str(cell)
                        elif value is None:
                            value = str(cell)
                            course_info[indicator] = value
                            break
            results[file_name]['course_info'] = course_info
            
            # Components
            df_weights = pd.read_excel(xls, "Setup", header=0, usecols="H, J, M", skiprows=5, nrows=14)
            df_weights = df_weights.dropna(how='all')
            assessment_info = {}
            current_assessment = None
            for index, row in df_weights.iterrows():
                if pd.notna(row['Unnamed: 7']):
                    current_assessment = row['Unnamed: 7']
                    assessment_info[current_assessment] = []
                if current_assessment and pd.notna(row['Component']):
                    if isinstance(assessment_info[current_assessment], list):
                        assessment_info[current_assessment].append((row['Component'], row['%'] / 100))
                elif current_assessment and pd.isna(row['Component']) and pd.notna(row['%']):
                    if isinstance(assessment_info[current_assessment], list):
                        assessment_info[current_assessment] = row['%'] / 100
            results[file_name]['assessment_info'] = assessment_info
            
            # Extract CLO/PLO Mappings table
            clos_amount = 5
            if 'CLO9' in str(pd.read_excel(xls, sheet_name="Setup").iloc[31, 0]):
                clos_amount = 9
            elif 'CLO1' in str(pd.read_excel(xls, sheet_name="Setup").iloc[35, 0]):
                clos_amount = 8
            elif 'CLO1' in str(pd.read_excel(xls, sheet_name="Setup").iloc[32, 0]):
                clos_amount = 5
            if clos_amount == 9:
                df_clo_plo_mappings = pd.read_excel(xls, "Setup", skiprows = 23, header = [0], nrows = 9)
            elif clos_amount == 8:
                df_clo_plo_mappings = pd.read_excel(xls, "Setup", skiprows = 23, header = [0], nrows = 8)
            elif clos_amount == 5:
                df_clo_plo_mappings = pd.read_excel(xls, "Setup", skiprows = 23, header = [0], nrows = 5)
            df_clo_plo_mappings = df_clo_plo_mappings.dropna(how = 'all')
            results[file_name]['df_clo_plo_mappings'] = df_clo_plo_mappings
            
            # CLO/PLO Mappings
            clo_plo_mappings = []
            for col in df_clo_plo_mappings.columns[2:]:
                for index, row in df_clo_plo_mappings.iterrows():
                    if row[col] == 'Y':
                        clo = row['Unnamed: 0']
                        plo = col
                        clo_plo_mappings.append((clo, plo))
            results[file_name]['clo_plo_mappings'] = clo_plo_mappings
            
            # CLO/Assessment Mapping for CA
            if clos_amount == 9:
                df_clo_ass_ca = pd.read_excel(xls, "Setup", skiprows=35, header=[0, 1], nrows=11)
            elif clos_amount == 8:
                df_clo_ass_ca = pd.read_excel(xls, "Setup", skiprows=34, header=[0, 1], nrows=10)
            elif clos_amount == 5:
                df_clo_ass_ca = pd.read_excel(xls, "Setup", skiprows = 31, header = [0, 1], nrows = 7)
            df_clo_ass_ca = df_clo_ass_ca.dropna(how = 'all')
            results[file_name]['df_clo_ass_ca'] = df_clo_ass_ca
            
            # CLO mapping to Assessment CA
            clo_ass_ca = []
            if clos_amount == 9:
                section_marks_row = 9
            elif clos_amount == 8:
                section_marks_row = 8
            elif clos_amount == 5:
                section_marks_row =5
            for col in df_clo_ass_ca.columns[2:]:
                for index, row in df_clo_ass_ca.iterrows():
                    if row[col] == 'Y':
                        clo = row[('Course Learning Outcome (CLO)', 'Unnamed: 0_level_1')]
                        question = col[1]
                        assessment = col[0]
                        section_marks = df_clo_ass_ca.at[section_marks_row, col]
                        clo_ass_ca.append((clo, question, assessment, section_marks))
            results[file_name]['clo_ass_ca'] = clo_ass_ca

            total_marks_ca = {}
            for _, _, assessment, section_marks in clo_ass_ca:
                if assessment in total_marks_ca:
                    total_marks_ca[assessment] += section_marks
                else:
                    total_marks_ca[assessment] = section_marks
            results[file_name]['total_marks_ca'] = total_marks_ca
            
            # CLO/Assessment Mapping for FA
            if clos_amount == 9:
                df_clo_ass_fa = pd.read_excel(xls, "Setup", skiprows = 50, header = [0, 1], nrows = 11)
            elif clos_amount == 8:
                df_clo_ass_fa = pd.read_excel(xls, "Setup", skiprows=48, header = [0, 1], nrows=10)
            elif clos_amount == 5:
                df_clo_ass_fa = pd.read_excel(xls, "Setup", skiprows = 42, header = [0, 1], nrows = 7)
            df_clo_ass_fa = df_clo_ass_fa.dropna(how = 'all')
            results[file_name]['df_clo_ass_fa'] = df_clo_ass_fa
            
            # CLO mapping to Assessment FA
            clo_ass_fa = []
            for col in df_clo_ass_fa.columns[2:]:
                for index, row in df_clo_ass_fa.iterrows():
                    if row[col] == 'Y':
                        clo = row[('Course Learning Outcome (CLO)', 'Unnamed: 0_level_1')]
                        question = col[1]
                        assessment = col[0]
                        section_marks = df_clo_ass_fa.at[section_marks_row, col]
                        clo_ass_fa.append((clo, question, assessment, section_marks))
            results[file_name]['clo_ass_fa'] = clo_ass_fa

            total_marks_fa = {}
            total_marks_fa = {}
            for _, _, assessment, section_marks in clo_ass_fa:
                if assessment in total_marks_fa:
                    total_marks_fa[assessment] += section_marks
                else:
                    total_marks_fa[assessment] = section_marks
            results[file_name]['total_marks_fa'] = total_marks_fa
            
            # Unique CLO list
            unique_clos = set()
            for clo, _, _, _ in clo_ass_ca:
                unique_clos.add(clo)
            for clo, _, _, _ in clo_ass_fa:
                unique_clos.add(clo)
            unique_clos_list = sorted(unique_clos, key=lambda x: int(x[3:]))
            results[file_name]['unique_clos_list'] = unique_clos_list
            
            # CLO - Assessments
            clo_assessments = {clo: [] for clo in unique_clos_list}
            for clo, questions, assessment, score in clo_ass_ca:
                clo_assessments[clo].append((questions, assessment, score))
            for clo, questions, assessment, score in clo_ass_fa:
                clo_assessments[clo].append((questions, assessment, score))
            results[file_name]['clo_assessments'] = clo_assessments

            # CLO weightages and percentages
            total_marks = {**total_marks_ca, **total_marks_fa}
            clo_assessment_marks = defaultdict(float)
            for clo, _, assessment, marks in clo_ass_ca:
                clo_assessment_marks[(clo, assessment)] += marks
            for clo, _, assessment, marks in clo_ass_fa:
                clo_assessment_marks[(clo, assessment)] += marks
            assessment_weightage = {}
            for key, value in assessment_info.items():
                if isinstance(value, list):
                    for assessment, weight in value:
                        assessment_weightage[assessment] = weight
                else:
                    assessment_weightage[key] = value
            clo_weightages = defaultdict(float)
            for (clo, assessment), marks_allocated in clo_assessment_marks.items():
                total_marks_assessment = total_marks[assessment]
                weightage_assessment = assessment_weightage[assessment]
                weightage = (marks_allocated / total_marks_assessment) * weightage_assessment
                clo_weightages[(clo, assessment)] = weightage
            clo_total_weightages = defaultdict(float)
            for (clo, assessment), weightage in clo_weightages.items():
                clo_total_weightages[clo] += weightage
            clo_percentages = {}
            for (clo, assessment) in clo_weightages.keys():
                weightage = clo_weightages[(clo, assessment)]
                total_weightage = clo_total_weightages[clo]
                percentage = (weightage / total_weightage)
                clo_percentages[(clo, assessment)] = percentage
            results[file_name]['clo_percentages'] = clo_percentages
            results[file_name]['clo_weightages'] = clo_weightages
            
            # Extract student marks CA and FA
            df_student_marks_ca = pd.read_excel(xls, "Student Marks CA", skiprows = 21, header = [0, 1])
            df_student_marks_ca = df_student_marks_ca.drop(columns = [('Unnamed: 0_level_0', 'No.')])
            df_student_marks_ca = df_student_marks_ca.dropna(how = 'all')
            df_student_marks_ca = df_student_marks_ca.dropna(axis=1, how='all')
            
            df_student_marks_fa = pd.read_excel(xls, "Student Marks FA", skiprows = 21, header = [0, 1])
            df_student_marks_fa = df_student_marks_fa.drop(columns = [('Unnamed: 0_level_0', 'No.')])
            df_student_marks_fa = df_student_marks_fa.dropna(how = 'all')
            df_student_marks_fa = df_student_marks_fa.dropna(axis=1, how='all')
            
            # Merge CA and FA data
            final_assessment_columns = [col for col in df_student_marks_fa.columns if col[0] == 'Final Assessment']
            df_final_assessment = df_student_marks_fa[final_assessment_columns]
            df_student_marks_combined = pd.concat([df_student_marks_ca, df_final_assessment], axis=1)
            df_student_marks_combined = df_student_marks_combined.dropna(subset=[("Unnamed: 1_level_0", 'Student ID')])
            results[file_name]['df_student_marks_combined'] = df_student_marks_combined
            
            # Obtain CLO from CA and FA
            adjusted_clo_percentages = defaultdict(lambda: defaultdict(float))
            clo_assessment_total_marks = defaultdict(lambda: defaultdict(float))
            for clo, assessments in clo_assessments.items():
                for _, assessment, marks in assessments:
                    clo_assessment_total_marks[clo][assessment] += marks
            for clo, assessments in clo_assessments.items():
                for _, assessment, marks in assessments:
                    total_marks = clo_assessment_total_marks[clo][assessment]
                    adjusted_percentage = marks / total_marks
                    adjusted_clo_percentages[clo][assessment] += adjusted_percentage
            clo_scores = defaultdict(lambda: defaultdict(list))
            clo_max_scores = defaultdict(lambda: defaultdict(list))
            clo_assessment_percentages = defaultdict(lambda: defaultdict(list))
            clo_assessment_details = defaultdict(lambda: defaultdict(list))
            for clo, assessments in clo_assessments.items():
                for question, assessment, max_score in assessments:
                    col_pair = (assessment, question)
                    if col_pair not in df_student_marks_combined.columns:
                        continue
                    for _, row in df_student_marks_combined.iterrows():
                        student_id = row[('Unnamed: 1_level_0', 'Student ID')]
                        student_name = row[('Unnamed: 2_level_0', 'Student Name')]
                        student_score = row[col_pair]
                        clo_scores[(student_id, student_name)][clo].append(student_score)
                        clo_max_scores[(student_id, student_name)][clo].append(max_score)
                        clo_assessment_percentages[(student_id, student_name)][clo].append(clo_percentages.get((clo, assessment), 0))
                        clo_assessment_details[(student_id, student_name)][clo].append((assessment, question))
            student_clo_data = defaultdict(lambda: defaultdict(list))
            for (student_id, student_name), scores in clo_scores.items():
                for clo, score_list in sorted(scores.items()):
                    max_score_list = clo_max_scores[(student_id, student_name)][clo]
                    total_max_score = sum(max_score_list)
                    assessment_details_list = clo_assessment_details[(student_id, student_name)][clo]
                    for score, max_score, (assessment, question) in zip(score_list, max_score_list, assessment_details_list):
                        percentage = clo_percentages.get((clo, assessment), 0)
                        student_clo_data[(student_id, student_name)][clo].append({
                            'score': score,
                            'max_score': max_score,
                            'percentage': percentage,
                            'assessment': assessment,
                            'question': question
                        })
            results[file_name]['student_clo_data'] = student_clo_data

            # Student CLO values (from marks to values of CLO parts)
            max_scores_per_clo_assessment = {}
            total_max_scores = {}
            for (student_id, student_name), clo_data in student_clo_data.items():
                for clo, data_list in clo_data.items():
                    for data in data_list:
                        assessment = data['assessment']
                        question = data['question']
                        max_score = data['max_score']
                        key = (clo, assessment)
                        if key not in max_scores_per_clo_assessment:
                            max_scores_per_clo_assessment[key] = {}
                        if question not in max_scores_per_clo_assessment[key]:
                            max_scores_per_clo_assessment[key][question] = max_score
                break
            for (clo, assessment), question_max_scores in max_scores_per_clo_assessment.items():
                total_max_score = sum(question_max_scores.values())
                total_max_scores[(clo, assessment)] = total_max_score
            student_clo_values = defaultdict(dict)
            for (student_id, student_name), clo_data in student_clo_data.items():
                if pd.isna(student_id) or pd.isna(student_name):
                    continue
                for clo, data_list in clo_data.items():
                    clo_value = 0
                    for data in data_list:
                        score = data['score']
                        max_score = data['max_score']
                        assessment = data['assessment']
                        question = data['question']
                        key = (clo, assessment)
                        total_max_score = total_max_scores[key]
                        clo_assessment_percentage = clo_percentages.get((clo, assessment), 0)
                        adjusted_percentage = (max_score / total_max_score) * clo_assessment_percentage
                        value = (score / max_score) * adjusted_percentage
                        clo_value += value
                    student_clo_values[(student_id, student_name)][clo] = clo_value
            results[file_name]['student_clo_values'] = student_clo_values
            
            # Overall CLOs DataFrame
            students_info = df_student_marks_combined[[
                ('Unnamed: 1_level_0', 'Student ID'), 
                ('Unnamed: 2_level_0', 'Student Name')
            ]].drop_duplicates()
            students_info = students_info.dropna(subset=[('Unnamed: 1_level_0', 'Student ID'),
                                                         ('Unnamed: 2_level_0', 'Student Name')])   
            clo_columns = list(clo_assessments.keys())
            category_columns = [f"Category_{clo}" for clo in clo_columns]
            student_clo_df = pd.DataFrame(columns=['Student_ID', 'Student_Name'] + clo_columns + category_columns)
            rows = []
            def categorize_clo(value):
                if value >= 80:
                    return 'Strong'
                elif value >= 50:
                    return 'Moderate'
                else:
                    return 'Weak'
            for _, row in students_info.iterrows():
                student_id = row[('Unnamed: 1_level_0', 'Student ID')]
                student_name = row[('Unnamed: 2_level_0', 'Student Name')]
                clo_values = student_clo_values.get((student_id, student_name), {})
                if not clo_values or any(pd.isna(v) for v in clo_values.values()):
                    continue
                row_data = {'Student_ID': student_id, 'Student_Name': student_name}
                formatted_clo_values = {clo: f"{value * 100:.2f}%" for clo, value in clo_values.items()}
                categorized_clo_values = {f"Category_{clo}": categorize_clo(value * 100) for clo, value in clo_values.items()}
                row_data.update(formatted_clo_values)
                row_data.update(categorized_clo_values)
                rows.append(row_data)
            student_clo_df = pd.concat([student_clo_df, pd.DataFrame(rows)], ignore_index=True)
            student_clo_df.reset_index(drop=True, inplace=True)
            results[file_name]['student_clo_df'] = student_clo_df
            
            # Overall PLOs DataFrame
            def categorize_value(value):
                percentage = value * 100
                if 80 <= percentage <= 100:
                    return 'Strong'
                elif 50 <= percentage < 80:
                    return 'Moderate'
                else:
                    return 'Weak'
            plo_clo_mappings = defaultdict(set)
            for clo, plo in clo_plo_mappings:
                plo_clo_mappings[plo].add(clo)
            student_plo_values = defaultdict(dict)
            for (student_id, student_name), clo_values in student_clo_values.items():
                plo_values = {}
                for plo in plo_clo_mappings.keys():
                    mapped_clos = plo_clo_mappings[plo]
                    student_clo_vals = [clo_values[clo] for clo in mapped_clos if clo in clo_values]
                    if student_clo_vals:
                        avg_value = sum(student_clo_vals) / len(student_clo_vals)
                        plo_values[plo] = avg_value
                    else:
                        plo_values[plo] = 0.0
                student_plo_values[(student_id, student_name)] = plo_values
            all_plo_columns = [f'PLO{i}' for i in range(1, 13)]
            category_columns = [f"Category_{plo}" for plo in all_plo_columns]
            student_plo_df = pd.DataFrame(columns=['Student_ID', 'Student_Name'] + all_plo_columns + category_columns)
            rows = []
            for (student_id, student_name), plo_values in student_plo_values.items():
                row_data = {'Student_ID': student_id, 'Student_Name': student_name}
                for plo in all_plo_columns:
                    row_data[plo] = '0.00%'
                    row_data[f"Category_{plo}"] = 'N/A'
                for plo, value in plo_values.items():
                    formatted_value = f"{value * 100:.2f}%"
                    category = categorize_value(value)
                    row_data[plo] = formatted_value
                    row_data[f"Category_{plo}"] = category
                rows.append(row_data)
            student_plo_df = pd.concat([student_plo_df, pd.DataFrame(rows)], ignore_index=True)
            results[file_name]['student_plo_df'] = student_plo_df

        except KeyError as e:
            if file_name in results:
                del results[file_name]
            unsupported_files.append(file_name)
            continue

    # Remove the progress bar together with the status text
    progress_bar.empty()
    status_text.empty()

    # print("Unsupported files:", unsupported_files)
    st.write("OBE forms processed successfully.")
    if unsupported_files:
        unsupported_files_warning = "**Unsupported files:**\n" + "\n".join([f"- {file}" for file in unsupported_files])
        st.warning(unsupported_files_warning)
    return results

# Main application
if selected == "Upload OBE Files":
    st.header("Upload OBE File (Zip)")
    uploaded_file = st.file_uploader("Upload OBE Forms ZIP file", type=["zip"], accept_multiple_files=False)
    course_codes_input = st.text_input("Enter course codes separated by commas (Enter 'all' to select all courses)")
    st.session_state['course_code_input'] = course_codes_input
    if st.button("Process Uploaded Files"):
        st.warning("Wait until a notice is printed out before moving to the other sidebar menu")
        if uploaded_file is not None:
            extract_folder = process_uploaded_file(uploaded_file)
            target_directory = tempfile.mkdtemp(prefix="OBE_")
            course_codes = [code.strip() for code in course_codes_input.split(",")]
            join_files(extract_folder, target_directory, course_codes)
            directory_path = target_directory
            print("Temporary directory created:", directory_path)
            results = process_obe_forms(target_directory)
            delete_temp_directory(directory_path)
            st.session_state['results'] = results  # Store results in session state
            st.write(results)
            st.success("File upload is finished. You can now access the OBE information.")
        else:
            st.error("Please upload the OBE Forms ZIP file to continue.")

results = st.session_state['results']
# Debug
# print(results)
def extract_student_id_and_name(df):
    if 'Student_ID' in df.columns and 'Student_Name' in df.columns:
        return df[['Student_ID', 'Student_Name']]
    else:
        raise ValueError("DataFrame does not have the required columns: 'Student_ID' and 'Student_Name'")
def display_all_unique_students(results):
    student_tracking = defaultdict(list)
    all_students = {}
    for file_name, data in results.items():
        try:
            student_plo_df = data['student_plo_df']
            if isinstance(student_plo_df, pd.DataFrame):
                extracted_data = extract_student_id_and_name(student_plo_df)
                extracted_data = extracted_data.dropna(subset=['Student_ID', 'Student_Name'], how='all')
                student_list = list(extracted_data.itertuples(index=False, name=None))
                student_tracking[file_name].extend(student_list)
                for student_id, student_name in student_list:
                    all_students[student_id] = student_name
            else:
                raise ValueError("student_plo_df is not a DataFrame")
        except ValueError as e:
            print(f"Error processing file {file_name}: {e}")
    return all_students, student_tracking
all_students, student_tracking = display_all_unique_students(results)
def standardize_allstudents_dict(all_students):
    new_dict = {}
    for stud_id, name in all_students.items():
        new_name = name.strip()
        new_id = stud_id.strip()
        if len(new_name) == 10 and new_name[:3].isalpha() and new_name[3:].isdigit():
            new_dict[new_name] = new_id
        else:
            new_dict[new_id] = new_name
        if new_name == new_id:
            new_dict.pop(new_id)
    all_students_new = new_dict
    return all_students_new
all_students_new = standardize_allstudents_dict(all_students)
def group_intakes(all_students_new):
    intakes = {}
    for id, name in all_students_new.items():
        intake = id[3:7]
        if intake in intakes:
            intakes[intake].append(id)
        else:
            intakes[intake] = [id]
    intakes = dict(sorted(intakes.items()))
    return intakes
intakes = group_intakes(all_students_new)
def make_intakes_array(intakes):
    for intake in intakes:
        globals()[f'student_ids_{intake}'] = intakes[intake]
    intakes_array = [globals()[f'student_ids_{intake}'] for intake in sorted(intakes.keys())]
    return intakes_array
intakes_array = make_intakes_array(intakes)
def student_files(student_ids_intake, results_arg): 
    student_files_dict = {}
    for student_id in tqdm(student_ids_intake):
        student_files = []
        for file_name, data in results_arg.items():
            df = data['student_plo_df']
            if student_id in df['Student_ID'].values:
                student_files.append(file_name)
        student_files_dict[student_id] = student_files
    return student_files_dict
def individual_display_all(student_files_dict, results_dict):
    individual_display_results = {}
    for student_id, file_list in tqdm(student_files_dict.items()):
        studentId, studentName, allStudentRows = individual_display_row(student_id, file_list, results_dict)
        individual_display_results[studentId] = (studentId, studentName, allStudentRows)
    return individual_display_results
def process_individual_display(intakes_arg, results_arg):
    for intake in intakes_arg:
        student_ids = globals()[f'student_ids_{intake}']
        student_files_intake = student_files(student_ids, results_arg)
        individual_display_intake = individual_display_all(student_files_intake, results_arg)
        individual_display_intake_combined = pd.concat([df[2] for df in individual_display_intake.values()], ignore_index=True)
        dfs_ID_name_intake = []
        for key, value in individual_display_intake.items():
            df = value[2].copy()
            df['Student_ID'] = value[0]
            df['Name'] = value[1]
            dfs_ID_name_intake.append(df)
        individual_display_intake_combined = pd.concat(dfs_ID_name_intake, ignore_index=True)
        cols = individual_display_intake_combined.columns.tolist()
        cols = cols[-2:] + cols[:-2]
        individual_display_intake_combined = individual_display_intake_combined[cols]
        globals()[f'student_files_{intake}'] = student_files_intake
        globals()[f'individual_display_{intake}'] = individual_display_intake
        globals()[f'dfs_ID_name_{intake}'] = dfs_ID_name_intake
        globals()[f'individual_display_{intake}_combined'] = individual_display_intake_combined
process_individual_display(intakes, results)

def calculate_po_attainment(individual_display_combined):
    plo_columns = ['PLO' + str(i) for i in range(1, 13)]
    for col in plo_columns:
        try:
            if individual_display_combined[col].dtype == 'object':
                individual_display_combined[col] = individual_display_combined[col].str.rstrip('%').astype(float)
            else:
                individual_display_combined[col] = individual_display_combined[col].astype(float)
        except Exception as e:
            print(f"Error processing column {col}: {e}")
            print(f"Exception: {e}")
            print(f"Data causing the error:\n{individual_display_combined[col]}")
            raise
    individual_display_combined[plo_columns] = individual_display_combined[plo_columns].fillna(0)
    student_names = individual_display_combined[['Student_ID', 'Name']].drop_duplicates()
    def mean_ignore_zeros(series):
        non_zero_values = series[series != 0]
        if len(non_zero_values) == 0:
            return 0
        return non_zero_values.mean()
    po_attainment = individual_display_combined.groupby('Student_ID')[plo_columns].apply(lambda x: x.apply(mean_ignore_zeros)).reset_index()
    po_attainment = po_attainment.merge(student_names, on='Student_ID', how='left')
    for col in plo_columns:
        po_attainment[col] = po_attainment[col].apply(lambda x: f"{x:.2f}%")
    def categorize_plo(value):
        if value >= 80:
            return 'Strong'
        elif value >= 50:
            return 'Moderate'
        elif value > 0:
            return 'Weak'
        else:
            return 'N/A'
    for col in plo_columns:
        numeric_col = col + '_numeric'
        try:
            if po_attainment[col].dtype == 'object':
                po_attainment[numeric_col] = po_attainment[col].str.rstrip('%').astype(float)
            else:
                po_attainment[numeric_col] = po_attainment[col].astype(float)
            po_attainment[col + '_Category'] = po_attainment[numeric_col].apply(categorize_plo)
        except Exception as e:
            # Debug
            print(f"Error processing column {col} in calculate_po_attainment (categorization)")
            print(f"Exception: {e}")
            print(f"Data causing the error:\n{po_attainment[col]}")
            raise
    category_columns = [col + '_Category' for col in plo_columns]
    po_attainment = po_attainment[['Student_ID', 'Name'] + plo_columns + category_columns]
    return po_attainment
def calculate_po_attainment_for_all(intakes):
    for intake in intakes:
        individual_display_combined = globals()[f'individual_display_{intake}_combined']
        po_attainment = calculate_po_attainment(individual_display_combined)
        globals()[f'po_attainment_{intake}'] = po_attainment
calculate_po_attainment_for_all(intakes)
def remove_duplicates_from_dict(data_dict):
    for key, value in data_dict.items():
        student_id, student_name, df = value
        df_cleaned = df.drop_duplicates().reset_index(drop=True)
        data_dict[key] = (student_id, student_name, df_cleaned)
for intake in intakes:
    remove_duplicates_from_dict(globals()[f'individual_display_{intake}'])
def clean_combined_displays(intakes):
    for intake in intakes:
        combined_df = globals()[f'individual_display_{intake}_combined']
        combined_df = combined_df.drop_duplicates().reset_index(drop=True)
        globals()[f'individual_display_{intake}_combined'] = combined_df
clean_combined_displays(intakes)
def combine_po_attainment(intakes):
    po_attainment_list = [globals()[f'po_attainment_{intake}'] for intake in intakes]
    po_attainment_combined = pd.concat(po_attainment_list, ignore_index=True)
    return po_attainment_combined
po_attainment_combined = combine_po_attainment(intakes)
def combine_individual_display_allintakes(intakes):
    individual_display_list = [globals()[f'individual_display_{intake}_combined'] for intake in intakes]
    individual_display_combined = pd.concat(individual_display_list, ignore_index=True)
    return individual_display_combined
individual_display_allintakes_combined = combine_individual_display_allintakes(intakes)
individual_display_allintakes_combined.set_index('Student_ID', inplace=True)
def get_student_entry(individual_display_arg, student_id):
    try:
        return individual_display_arg.loc[student_id]
    except KeyError:
        return f"Student ID {student_id} not found."
def normalize_academic_session(session):
    digits = re.sub(r'\D', '', session)
    if len(digits) == 6:
        if digits[:2] == "20":
            return digits[:4] + digits[4:]
        else:
            return digits[2:] + digits[:2]
    elif len(digits) == 4:
        if int(digits[:2]) > 12:
            return "20" + digits[:2] + digits[2:]
        else:
            return "20" + digits[2:] + digits[:2]
    return session
def normalize_course_code(course_code):
    return course_code.replace(' ', '')
def find_file_course_code_academic_session(results_df, target_course_code, target_academic_session):
    normalized_target_session = normalize_academic_session(target_academic_session)
    normalized_target_course_code = normalize_course_code(target_course_code)
    for file_name, data in results_df.items():
        sheet_title = data['sheet_title']
        course_info = data['course_info']
        assessment_info = data['assessment_info']
        df_clo_plo_mappings = data['df_clo_plo_mappings']
        clo_plo_mappings = data['clo_plo_mappings']
        df_clo_ass_ca = data['df_clo_ass_ca']
        clo_ass_ca = data['clo_ass_ca']
        total_marks_ca = data['total_marks_ca']
        df_clo_ass_fa = data['df_clo_ass_fa']
        clo_ass_fa = data['clo_ass_fa']
        total_marks_fa = data['total_marks_fa']
        unique_clos_list = data['unique_clos_list']
        clo_assessments = data['clo_assessments']
        clo_percentages = data['clo_percentages']
        clo_weightages = data['clo_weightages']
        df_student_marks_combined = data['df_student_marks_combined']
        student_clo_data = data['student_clo_data']
        student_clo_values = data['student_clo_values']
        student_clo_df = data['student_clo_df']
        student_plo_df = data['student_plo_df']
        if isinstance(course_info, pd.DataFrame):
            course_info = course_info.set_index('Attribute')['Value'].to_dict()
        course_code = course_info.get('Course Code:')
        academic_session = course_info.get('Academic Session')
        normalized_academic_session = normalize_academic_session(academic_session)
        normalized_course_code = normalize_course_code(course_code)
        if normalized_course_code == normalized_target_course_code and normalized_academic_session == normalized_target_session:
            return file_name, course_info
    else:
        print("No matching course found.")
        return None, None  
def get_student_clo(results_df, course_code, academic_session):
    file_name, _ = find_file_course_code_academic_session(results_df, course_code, academic_session)
    if file_name is None:
        st.error("No matching course found for the provided course code and academic session.")
        return pd.DataFrame() 
    return results_df[file_name]['student_clo_df']
def get_student_plo(results_df, course_code, academic_session):
    file_name, _ = find_file_course_code_academic_session(results_df, course_code, academic_session)
    return results_df[file_name]['student_plo_df']
def get_clo_plo_map(results_df, course_code, academic_session):
    file_name, _ = find_file_course_code_academic_session(results_df, course_code, academic_session)
    if file_name is None:
        st.error("No matching course found for the provided course code and academic session.")
        return {}  
    return results_df[file_name]['clo_plo_mappings']
def get_clo_assessments(results_df, course_code, academic_session):
    file_name, _ = find_file_course_code_academic_session(results_df, course_code, academic_session)
    return results_df[file_name]['clo_assessments']
def get_student_marks(results_df, course_code, academic_session):
    file_name, _ = find_file_course_code_academic_session(results_df, course_code, academic_session)
    return results_df[file_name]['df_student_marks_combined']
def get_clo_ass_ca(results_df, course_code, academic_session):
    file_name, _ = find_file_course_code_academic_session(results_df, course_code, academic_session)
    return results_df[file_name]['clo_ass_ca']
def get_clo_ass_fa(results_df, course_code, academic_session):
    file_name, _ = find_file_course_code_academic_session(results_df, course_code, academic_session)
    return results_df[file_name]['clo_ass_fa']

def drop_zero_percent_rows(intakes):
    plo_columns = [f'PLO{i}' for i in range(1, 13)]
    for intake in intakes:
        po_attainment = globals()[f'po_attainment_{intake}']
        for col in plo_columns:
            po_attainment[col] = po_attainment[col].str.rstrip('%').astype(float)
        po_attainment = po_attainment[(po_attainment[plo_columns] != 0.00).all(axis=1)]
        po_attainment.reset_index(drop=True, inplace=True)
        globals()[f'po_attainment_{intake}_dropped'] = po_attainment
drop_zero_percent_rows(intakes)  
def calculate_and_plot_plo_averages(intakes, specific_intake=None):
    plo_columns = [f'PLO{i}' for i in range(1, 13)]
    intakes_to_process = [specific_intake] if specific_intake else intakes
    
    for intake in intakes_to_process:
        po_attainment = globals()[f'po_attainment_{intake}']
        
        # Exclude 0 values before calculating the mean
        po_attainment_no_zeros = po_attainment[plo_columns].replace(0, pd.NA)
        plo_averages = po_attainment_no_zeros.mean()
        globals()[f'plo_averages_{intake}'] = plo_averages

        # Streamlit for the bar chart
        st.bar_chart(plo_averages, x_label="PLOs", y_label="Average Score (%)")
        
        # Sort and print PLO averages
        plo_averages_sorted = plo_averages.sort_values()
        plo_averages_sorted_df = plo_averages_sorted.reset_index()
        plo_averages_sorted_df.columns = ['PLO', 'Average Score']   
        st.write(f"PLO sorted by average score for intake {intake}:")
        st.write(plo_averages_sorted_df.to_html(index=False), unsafe_allow_html=True)
def calculate_and_plot_plo_categories(intakes, specific_intake=None):
    plo_category_columns = [f'PLO{i}_Category' for i in range(1, 13)]
    intakes_to_process = [specific_intake] if specific_intake else intakes
    
    for intake in intakes_to_process:
        po_attainment = globals()[f'po_attainment_{intake}']
        
        # Calculate category counts
        counts = {col: po_attainment[col].value_counts() for col in plo_category_columns}
        counts_df = pd.DataFrame(counts).fillna(0).astype(int)
        counts_df = counts_df.T
        
        # Exclude "N/A" category
        counts_df = counts_df.drop(columns=['N/A'], errors='ignore')
        
        globals()[f'plo_category_counts_{intake}'] = counts_df
        
        # Plot using Streamlit
        fig, ax = plt.subplots(figsize=(12, 8))
        counts_df.plot(kind='bar', stacked=True, ax=ax)
        ax.set_title(f'Programme Learning Outcomes Attainment\nStudent Intake {intake}')
        ax.set_xlabel('PLO Categories')
        ax.set_ylabel('Number of Students')
        ax.set_xticklabels([f'PLO{i}' for i in range(1, 13)], rotation=0)
        ax.legend(title='Category')
        plt.tight_layout()
        
        # Save the plot to a variable
        plot_variable_name = f'plo_category_plot_{intake}'
        globals()[plot_variable_name] = plt.gcf()
        
        # Show the plot only if a specific intake is provided
        if specific_intake:
            plt.show()

        # Streamlit plot
        st.pyplot(fig)


# Start of "Model Training Outputs"
def group_intakes(all_students_new):
    intakes = {}
    for id, name in all_students_new.items():
        intake = id[3:7]
        if intake in intakes:
            intakes[intake].append(id)
        else:
            intakes[intake] = [id]
    intakes = dict(sorted(intakes.items()))
    return intakes
def filter_by_year(df, year):
    year_str = str(year)
    return df[df['Course_Code'].str[3] == year_str].reset_index(drop=True)
years = [1, 2, 3, 4]
def get_yearly_individual_display(intakes, years):
    yearly_individualdisplay = {}
    for intake in intakes:
        combined_df = globals()[f'individual_display_{intake}_combined']
        # print(f"\nProcessing intake {intake}: combined_df shape {combined_df.shape}")
        for year in years:
            if year == 1:
                filtered_df = filter_by_year(combined_df, year)
            else:
                filtered_df = pd.concat([filter_by_year(combined_df, y) for y in range(1, year + 1)]).reset_index(drop=True)
            filtered_df = filtered_df.sort_values(by=['Student_ID', 'Course_Code']).reset_index(drop=True)
            yearly_individualdisplay[f'individual_display_{intake}_year_{year}'] = filtered_df
    return yearly_individualdisplay
yearly_individualdisplay = get_yearly_individual_display(intakes, years)
def remove_identical_yearly_data(yearly_individualdisplay, intakes):
    for intake in intakes.keys():
        for year in range(4, 1, -1):
            current_year_key = f'individual_display_{intake}_year_{year}'
            previous_year_key = f'individual_display_{intake}_year_{year - 1}'
            
            # Check if both keys exist in the dictionary
            if current_year_key in yearly_individualdisplay and previous_year_key in yearly_individualdisplay:
                current_year_data = yearly_individualdisplay[current_year_key]
                previous_year_data = yearly_individualdisplay[previous_year_key]
                
                # Compare the number of rows for the current year with the previous year
                if len(current_year_data) == len(previous_year_data):
                    print(f"Removing {current_year_key} as it is identical to {previous_year_key}")
                    del yearly_individualdisplay[current_year_key]
remove_identical_yearly_data(yearly_individualdisplay, intakes)
for intake in intakes:
    # Extract years dynamically from the keys of the dictionary
    years = sorted(set(int(key.split('_')[-1]) for key in yearly_individualdisplay.keys() if key.startswith(f'individual_display_{intake}_year_')))
    
    for year in years:
        key = f'individual_display_{intake}_year_{year}'
        if key in yearly_individualdisplay:
            globals()[f'po_attainment_{intake}_year_{year}'] = calculate_po_attainment(yearly_individualdisplay[key])

def convert_percentage_to_float(df, columns):
    for column in columns:
        if df[column].dtype == 'object':
            df[column] = df[column].str.rstrip('%').astype(float) / 100.0
        df[column] = df[column].fillna(0.0)
    return df
def build_model(hp):
    model = Sequential()
    # Tune the number of units in the first LSTM layer
    hp_units_1 = hp.Int('units_1', min_value=32, max_value=128, step=32)
    model.add(Bidirectional(LSTM(units=hp_units_1, return_sequences=True, kernel_regularizer=l2(0.001)), input_shape=(X_train_scaled.shape[1], X_train_scaled.shape[2])))
    model.add(Dropout(hp.Float('dropout_rate_1', min_value=0.1, max_value=0.5, step=0.1)))
    
    # Tune the number of units in the second LSTM layer
    hp_units_2 = hp.Int('units_2', min_value=32, max_value=128, step=32)
    model.add(Bidirectional(LSTM(units=hp_units_2, kernel_regularizer=l2(0.001))))
    model.add(Dropout(hp.Float('dropout_rate_2', min_value=0.1, max_value=0.5, step=0.1)))
    
    # Tune the number of units in the Dense layer
    hp_dense_units = hp.Int('dense_units', min_value=32, max_value=128, step=32)
    model.add(Dense(units=hp_dense_units, activation='relu', kernel_regularizer=l2(0.001)))
    model.add(Dense(12, activation='linear'))
    
    model.compile(optimizer='adam', loss='mse')
    return model


if selected == "Access OBE Information":
    st.header("Access OBE Information")
    
    # Text boxes for user input
    course_code_input = st.text_input("Enter Course Code:", value=st.session_state['course_code'])
    academic_session_input = st.text_input("Enter Academic Session:", value=st.session_state['academic_session'])
    
    # Update session state with the new input values
    st.session_state['course_code'] = course_code_input
    st.session_state['academic_session'] = academic_session_input
    
    # Button to process the input and show tabs
    if st.button("Process"):
        st.session_state['show_tabs'] = True
    
    # Display the status of the results variable
    if st.session_state['results'] is not None:
        st.success("Results are available.")
    else:
        st.warning("No results available. Please upload the OBE Forms ZIP file in the 'Upload OBE Files' section.")
    
    # Check if results are available in session state and show tabs if the button has been clicked
    if st.session_state['results'] is not None and st.session_state['show_tabs']:
        results = st.session_state['results']
        
        # Create tabs for each component
        tabs = st.tabs(["Student CLO", "Student PLO", "CLO-PLO Map", "CLO Assessments", "Student Marks", "CLO Assess CA", "CLO Assess FA"])
        
        with tabs[0]:
            st.header(f"Student CLO - {st.session_state['course_code']} ({st.session_state['academic_session']})")
            student_clo_df = get_student_clo(results, st.session_state['course_code'], st.session_state['academic_session'])
            student_clo_df = student_clo_df.dropna(subset=['Student_ID', 'Student_Name'])
            # st.dataframe(student_clo_df)
            show_colored_dataframe(student_clo_df)
        
        with tabs[1]:
            st.header(f"Student PLO - {st.session_state['course_code']} ({st.session_state['academic_session']})")
            student_plo_df = get_student_plo(results, st.session_state['course_code'], st.session_state['academic_session'])
            student_plo_df = student_plo_df.dropna(subset=['Student_ID', 'Student_Name'])
            show_colored_dataframe(student_plo_df)
        
        with tabs[2]:
            st.header(f"CLO-PLO Map - {st.session_state['course_code']} ({st.session_state['academic_session']})")
            clo_plo_map = get_clo_plo_map(results, st.session_state['course_code'], st.session_state['academic_session'])
            
            # Ensure clo_plo_map is a list of tuples
            if isinstance(clo_plo_map, list):
                # Convert the list of tuples to a dictionary
                clo_plo_dict = {}
                for clo, plo in clo_plo_map:
                    if clo not in clo_plo_dict:
                        clo_plo_dict[clo] = []
                    clo_plo_dict[clo].append(plo)
                
                # Create a DataFrame for the CLO-PLO map
                clo_list = sorted(clo_plo_dict.keys())
                plo_list = [f'PLO{i}' for i in range(1, 13)]
                clo_plo_df = pd.DataFrame(0, index=clo_list, columns=plo_list)
                
                for clo, plos in clo_plo_dict.items():
                    for plo in plos:
                        clo_plo_df.at[clo, plo] = 1
                
                # Create a heatmap to visualize the CLO-PLO map
                fig = px.imshow(clo_plo_df, text_auto=False, aspect="auto", color_continuous_scale="Blues")
                fig.update_coloraxes(showscale=False)
                fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='black')
                fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='black')
                fig.update_layout(
                    title="CLO-PLO Map",
                    xaxis_title="PLOs",
                    yaxis_title="CLOs",
                    yaxis=dict(scaleanchor="x", scaleratio=1)  # Make cells square
                )
                st.plotly_chart(fig)
            else:
                st.error("CLO-PLO Map data is not in the expected format.")
        
        with tabs[3]:
            st.header(f"CLO Assessments - {st.session_state['course_code']} ({st.session_state['academic_session']})")
            clo_assessments = get_clo_assessments(results, st.session_state['course_code'], st.session_state['academic_session'])
            
            # Convert the clo_assessments dictionary to a DataFrame for display
            clo_assessments_list = []
            for clo, assessments in clo_assessments.items():
                for assessment in assessments:
                    clo_assessments_list.append([clo] + list(assessment))
            
            clo_assessments_df = pd.DataFrame(clo_assessments_list, columns=['CLO', 'Question', 'Assessment', 'Weight'])
            st.dataframe(clo_assessments_df)
        
        with tabs[4]:
            st.header(f"Student Marks - {st.session_state['course_code']} ({st.session_state['academic_session']})")
            student_marks_df = get_student_marks(results, st.session_state['course_code'], st.session_state['academic_session'])
            st.dataframe(student_marks_df)
        
        with tabs[5]:
            st.header(f"CLO Continuous Assessment - {st.session_state['course_code']} ({st.session_state['academic_session']})")
            clo_ass_ca = get_clo_ass_ca(results, st.session_state['course_code'], st.session_state['academic_session'])
            st.dataframe(clo_ass_ca)
        
        with tabs[6]:
            st.header(f"CLO Final Assessment - {st.session_state['course_code']} ({st.session_state['academic_session']})")
            clo_ass_fa = get_clo_ass_fa(results, st.session_state['course_code'], st.session_state['academic_session'])
            st.dataframe(clo_ass_fa)
    else:
        st.write("Please enter the Course Code and Academic Session, then click 'Process' to access the OBE information.")

elif selected == "Display PO Attainment":
    st.header("Display PO Attainment")
    results = st.session_state['results']
    all_students, student_tracking = display_all_unique_students(results)
    all_students_new = standardize_allstudents_dict(all_students)
    intakes = group_intakes(all_students_new)
    calculate_po_attainment_for_all(intakes)
    po_attainment_vars = {k: v for k, v in globals().items() if k.startswith('po_attainment_')}
    po_attainment_choices = [k.split('_')[-1] for k in po_attainment_vars.keys()]
    
    # Create a dropdown menu
    selected_intake = st.selectbox("Select Intake", po_attainment_choices)
    
    # Display the PO attainment for the selected choice
    if selected_intake:
        po_attainment_var_name = f'po_attainment_{selected_intake}'
        po_attainment_df = po_attainment_vars[po_attainment_var_name]
        show_colored_dataframe(po_attainment_df)

elif selected == "Display Individual Student Data":
    st.header("Display Individual Student Data")
    student_id = st.text_input("Enter Student ID:")
    if student_id:
        student_entry = get_student_entry(individual_display_allintakes_combined, student_id)
        if student_entry is not None:
            st.dataframe(student_entry)
        else:
            st.error(f"Student ID {student_id} not found.")
    
elif selected == "Plot PLO Averages and Categories":
    st.header("Plot PLO Averages and Categories")
    graph_input_intake = st.selectbox("Select Intake", intakes)
    tabs = st.tabs(["Average POs Attainment Marks", "PLO Attainment"])
    with tabs[0]:
        st.header("Average POs Attainment Marks")
        calculate_and_plot_plo_averages(intakes, graph_input_intake)
    with tabs[1]:
        st.header("PLO Attainment")
        calculate_and_plot_plo_categories(intakes, graph_input_intake)

elif selected == "Model Training Outputs":
    st.header("Model Training Outputs")
    tabs = st.tabs(["Training", "Scatter Plot", "Line Chart", "Distribution of Prediction Errors", "Metrics", "Training and Validation Loss Plot"])
    with tabs[0]:  # Training
        if st.button("Start Training"):
            data_list = []
            for intake in intakes:
                yearly_data = []
                for key in yearly_individualdisplay.keys():
                    if key.startswith(f'individual_display_{intake}_year_'):
                        df_year = yearly_individualdisplay[key]
                        year = int(key.split('_')[-1])
                        df_year['Year'] = year
                        df_year['Intake'] = intake
                        yearly_data.append(df_year)
                if yearly_data:
                    df_intake = pd.concat(yearly_data, ignore_index=True)
                    data_list.append(df_intake)
            
            if not data_list:
                st.error("No data available for training. Please ensure your dataset contains multiple years of data.")
                st.stop()
            
            all_data = pd.concat(data_list, ignore_index=True)
            st.write("Data concatenated successfully.")
            
            plo_columns = [f'PLO{i}' for i in range(1, 13)]
            all_data = convert_percentage_to_float(all_data, plo_columns)
            
            encoder = OneHotEncoder(sparse_output=False)
            course_codes_encoded = encoder.fit_transform(all_data[['Course_Code']])
            course_code_columns = encoder.get_feature_names_out(['Course_Code'])
            course_codes_df = pd.DataFrame(course_codes_encoded, columns=course_code_columns)
            joblib.dump(encoder, 'onehot_encoder.save')
            
            all_data = pd.concat([all_data, course_codes_df], axis=1)
            st.write("One-hot encoding completed and data concatenated.")
            
            pivot_data = all_data.pivot_table(
                index=['Student_ID', 'Name', 'Intake'], 
                columns='Year', 
                values=plo_columns + list(course_code_columns)
            )
            pivot_data.columns = [f'{col}_Y{year}' for col, year in pivot_data.columns]
            pivot_data.reset_index(inplace=True)
            st.write("Pivot table created successfully.")
            
            available_years = sorted(set(int(col.split('_Y')[-1]) for col in pivot_data.columns if '_Y' in col))
            st.write(f"Available Years: {available_years}")
            
            if len(available_years) < 2:
                st.error("Insufficient years of data for training. At least two years of data are required.")
                st.stop()
            
            feature_years = available_years[:-1]
            target_year = available_years[-1]
            st.write(f"Feature Years: {feature_years}")
            st.write(f"Target Year: {target_year}")
            
            features = [f'PLO{i}_Y{year}' for year in feature_years for i in range(1, 13)] + \
                       [f'{col}_Y{year}' for year in feature_years for col in course_code_columns]
            target = [f'PLO{i}_Y{target_year}' for i in range(1, 13)]
            
            pivot_data = pivot_data.dropna(subset=target)
            pivot_data[features] = pivot_data[features].fillna(0.0)
            
            X = pivot_data[features].values
            y = pivot_data[target].values
            
            # Ensure that features and feature_years are correctly defined
            st.write(f"Number of features: {len(features)}")
            st.write(f"Number of feature years: {len(feature_years)}")
            
            if len(feature_years) == 0:
                st.error("No feature years available after processing. Please check your data.")
                st.stop()
            
            try:
                X = X.reshape(X.shape[0], len(feature_years), len(features) // len(feature_years))
            except ZeroDivisionError:
                st.error("Division by zero encountered while reshaping X. Please check the number of feature years and features.")
                st.stop()
            except Exception as e:
                st.error(f"An unexpected error occurred while reshaping X: {e}")
                st.stop()
            
            X_train_full, X_test, y_train_full, y_test, idx_train_full, idx_test = train_test_split(
                X, y, pivot_data.index.values, test_size=0.2, random_state=42
            )
            X_train, X_val, y_train, y_val = train_test_split(
                X_train_full, y_train_full, test_size=0.2, random_state=42
            )
            st.write("Data split into training, validation, and test sets.")
            
            X_train_shape = X_train.shape
            X_val_shape = X_val.shape
            X_test_shape = X_test.shape
            
            X_train_flat = X_train.reshape(-1, X_train.shape[-1])
            X_val_flat = X_val.reshape(-1, X_val_shape[2])
            X_test_flat = X_test.reshape(-1, X_test_shape[2])
            
            scaler_X = StandardScaler()
            scaler_y = StandardScaler()
            
            X_train_flat_scaled = scaler_X.fit_transform(X_train_flat)
            y_train_scaled = scaler_y.fit_transform(y_train)
            
            joblib.dump(scaler_X, 'scaler_X.save')
            joblib.dump(scaler_y, 'scaler_y.save')
            
            X_val_flat_scaled = scaler_X.transform(X_val_flat)
            X_test_flat_scaled = scaler_X.transform(X_test_flat)
            y_val_scaled = scaler_y.transform(y_val)
            y_test_scaled = scaler_y.transform(y_test)
            
            X_train_scaled = X_train_flat_scaled.reshape(X_train_shape)
            X_val_scaled = X_val_flat_scaled.reshape(X_val_shape)
            X_test_scaled = X_test_flat_scaled.reshape(X_test_shape)
            
            st.write("Data scaling completed.")
            
            tuner = kt.GridSearch(
                build_model,
                objective='val_loss',
                max_trials=10,  # Consider increasing for better results
                executions_per_trial=1,
                directory='my_dir',
                project_name='plo_prediction'
            )
            early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
            
            tuner.search(
                X_train_scaled, y_train_scaled,
                epochs=50,
                validation_data=(X_val_scaled, y_val_scaled),
                callbacks=[early_stopping]
            )
            best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
            
            st.write(f"""
            The hyperparameter search is complete. The optimal number of units in the first LSTM layer is {best_hps.get('units_1')}, 
            the second LSTM layer is {best_hps.get('units_2')}, the number of units in the Dense layer is {best_hps.get('dense_units')}, 
            and the dropout rates are {best_hps.get('dropout_rate_1')} and {best_hps.get('dropout_rate_2')}.
            """)
            
            model = tuner.hypermodel.build(best_hps)
            history = model.fit(
                X_train_scaled, y_train_scaled,
                epochs=50,
                batch_size=16,
                validation_data=(X_val_scaled, y_val_scaled),
                callbacks=[early_stopping]
            )
            model.save('plo_prediction_model.keras')
            st.write("Model training and saving completed.")
            
            # Model evaluation
            test_loss = model.evaluate(X_test_scaled, y_test_scaled)
            st.write(f'Test Loss: {test_loss}')
            
            y_pred_scaled = model.predict(X_test_scaled)
            y_pred = scaler_y.inverse_transform(y_pred_scaled)
            
            model_results = pd.DataFrame({
                'Student_ID': pivot_data.loc[idx_test, 'Student_ID'].values,
                'Name': pivot_data.loc[idx_test, 'Name'].values
            })
            for i in range(1, 13):
                model_results[f'Actual_PLO{i}'] = y_test[:, i - 1]
                model_results[f'Predicted_PLO{i}'] = y_pred[:, i - 1]
            st.write("Results DataFrame created.")
            st.write(model_results)
            
            # Store y_test, y_pred, and results in Streamlit's session state
            st.session_state['y_test'] = y_test
            st.session_state['y_pred'] = y_pred
            st.session_state['model_results'] = model_results

    with tabs[1]:  # Scatter Plot
        st.header("Scatter Plot")
        
        # Check if y_test and y_pred are in session state
        if 'y_test' in st.session_state and 'y_pred' in st.session_state:
            y_test = st.session_state['y_test']
            y_pred = st.session_state['y_pred']
            
            fig, axes = plt.subplots(3, 4, figsize=(20, 15))
            axes = axes.flatten()
            for i in range(12):
                plo_number = i + 1
                axes[i].scatter(y_test[:, i], y_pred[:, i], alpha=0.5)
                axes[i].set_title(f'Actual vs Predicted for PLO{plo_number}')
                axes[i].set_xlabel('Actual')
                axes[i].set_ylabel('Predicted')
                axes[i].plot([y_test[:, i].min(), y_test[:, i].max()], [y_test[:, i].min(), y_test[:, i].max()], 'r--')  # Diagonal line
            plt.tight_layout()
            
            # Display the plot using Streamlit
            st.pyplot(fig)
        else:
            st.error("Please run training in the 'Training' tab first to generate predictions.")

    with tabs[2]:  # Line Chart
        st.header("Line Chart")
        
        # Check if 'model_results' exists and is a DataFrame
        if 'model_results' in st.session_state:
            model_results = st.session_state['model_results']
            
            # Check if 'results' is a DataFrame
            if not isinstance(model_results, pd.DataFrame):
                st.error("The 'model_results' data is not a DataFrame. Please rerun the training in the 'Training' tab.")
                st.stop()
        else:
            st.error("Please run training in the 'Training' tab first to generate predictions.")
            st.stop()

        model_results.reset_index(drop=True, inplace=True)
        
        if 'Student_ID' not in model_results.columns:
            st.error("The 'Student_ID' column does not exist in the model_results DataFrame.")
            st.stop()
        
        num_plos = 12
        plo_numbers = list(range(1, num_plos + 1))
        num_samples = 3
        
        if len(model_results) < num_samples:
            st.error(f"Not enough samples to plot. The DataFrame contains only {len(model_results)} rows.")
            st.stop()
        
        sample_indices = random.sample(range(len(model_results)), num_samples)
        
        for idx in sample_indices:
            try:
                student_id = model_results.iloc[idx]['Student_ID']
                actual_plos = model_results.iloc[idx][[f'Actual_PLO{i}' for i in plo_numbers]].tolist()
                predicted_plos = model_results.iloc[idx][[f'Predicted_PLO{i}' for i in plo_numbers]].tolist()
                
                # Plot using Streamlit
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(plo_numbers, actual_plos, marker='o', label='Actual PLOs', color='blue')
                ax.plot(plo_numbers, predicted_plos, marker='x', label='Predicted PLOs', color='red')
                ax.set_title(f'Actual vs Predicted PLOs for Student ID {student_id}')
                ax.set_xlabel('PLO Number')
                ax.set_ylabel('PLO Attainment')
                ax.set_xticks(plo_numbers)
                ax.legend()
                ax.grid(True)
                
                # Display the plot using Streamlit
                st.pyplot(fig)
            except KeyError as e:
                st.write(f"KeyError: {e} for index {idx}. Please check if all PLO columns exist.")
            except IndexError:
                st.write(f"IndexError: Index {idx} is out of bounds for the model_results DataFrame.")
            except Exception as e:
                st.write(f"An unexpected error occurred for index {idx}: {e}")


    with tabs[3]:  # Distribution of Prediction Errors
        st.header("Distribution of Prediction Errors")
        errors = y_pred - y_test
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(errors.flatten(), bins=50)
        ax.set_title('Distribution of Prediction Errors')
        ax.set_xlabel('Prediction Error')
        ax.set_ylabel('Frequency')
        st.pyplot(fig)

    with tabs[4]:  # Metrics
        st.header("Metrics")
        st.subheader("Model metrics")
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        st.write(f"Mean Absolute Error: {mae:.4f}")
        st.write(f"Mean Squared Error: {mse:.4f}")
        st.write(f"Root Mean Squared Error: {rmse:.4f}")
        st.write(f"R^2 Score: {r2:.4f}")

        st.subheader("Metrics for each PLO")
        for i in range(12):
            plo_number = i + 1
            actual = y_test[:, i]
            predicted = y_pred[:, i]
            mae = mean_absolute_error(actual, predicted)
            mse = mean_squared_error(actual, predicted)
            r2 = r2_score(actual, predicted)
            st.write(f'PLO{plo_number}: MAE={mae:.4f}, MSE={mse:.4f}, R²={r2:.4f}')
    
    with tabs[5]:  # Training and Validation Loss Plot
        st.header("Training and Validation Metrics and Loss Plot")
        y_train_pred_scaled = model.predict(X_train_scaled)
        y_train_pred = scaler_y.inverse_transform(y_train_pred_scaled)

        mae_train = mean_absolute_error(y_train, y_train_pred)
        mse_train = mean_squared_error(y_train, y_train_pred)
        rmse_train = np.sqrt(mse_train)
        r2_train = r2_score(y_train, y_train_pred)

        st.write("Training Data Metrics:")
        st.write(f"Mean Absolute Error: {mae_train:.4f}")
        st.write(f"Mean Squared Error: {mse_train:.4f}")
        st.write(f"Root Mean Squared Error: {rmse_train:.4f}")
        st.write(f"R^2 Score: {r2_train:.4f}")

        # Evaluate on test data
        y_test_pred_scaled = model.predict(X_test_scaled)
        y_test_pred = scaler_y.inverse_transform(y_test_pred_scaled)

        mae_test = mean_absolute_error(y_test, y_test_pred)
        mse_test = mean_squared_error(y_test, y_test_pred)
        rmse_test = np.sqrt(mse_test)
        r2_test = r2_score(y_test, y_test_pred)

        st.write("\nTest Data Metrics:")
        st.write(f"Mean Absolute Error: {mae_test:.4f}")
        st.write(f"Mean Squared Error: {mse_test:.4f}")
        st.write(f"Root Mean Squared Error: {rmse_test:.4f}")
        st.write(f"R^2 Score: {r2_test:.4f}")

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(history.history['loss'], label='Training Loss')
        ax.plot(history.history['val_loss'], label='Validation Loss')
        ax.set_title('Model Loss Over Epochs')
        ax.set_ylabel('Loss')
        ax.set_xlabel('Epoch')
        ax.legend()
        st.pyplot(fig)

        model.save('plo_prediction_model.keras')

if selected == "Upload New OBE Files":
    course_codes_input = st.session_state['course_code_input']
    st.header("Upload New OBE File (Zip)")
    uploaded_file = st.file_uploader("Upload OBE Forms ZIP file", type=["zip"], accept_multiple_files=False)
    if st.button("Process Uploaded Files"):
        st.warning("Wait until a notice is printed out before moving to the other sidebar menu")
        if uploaded_file is not None:
            extract_folder = process_uploaded_file(uploaded_file)
            target_directory__newdata = tempfile.mkdtemp(prefix="OBE_newdata_")
            course_codes = [code.strip() for code in course_codes_input.split(",")]
            join_files(extract_folder, target_directory__newdata, course_codes)
            directory_path__newdata = target_directory__newdata
            print("Temporary directory created:", directory_path__newdata)
            results__newdata = process_obe_forms(target_directory__newdata)
            results__newdata_copy = results__newdata.copy()
            all_students__newdata, student_tracking__newdata = display_all_unique_students(results__newdata)
            all_students_new__newdata = standardize_allstudents_dict(all_students__newdata)
            intakes__newdata = group_intakes(all_students_new__newdata)
            intakes_array__newdata = make_intakes_array(intakes__newdata)
            # ---------
            process_individual_display(intakes__newdata, results__newdata)
            calculate_po_attainment_for_all(intakes__newdata)
            for intake__newdata in intakes__newdata:
                remove_duplicates_from_dict(globals()[f'individual_display_{intake__newdata}'])
            clean_combined_displays(intakes__newdata)
            po_attainment_combined__newdata = combine_po_attainment(intakes__newdata)
            individual_display_allintakes_combined__newdata = combine_individual_display_allintakes(intakes__newdata)
            individual_display_allintakes_combined__newdata.set_index('Student_ID', inplace=True)
            individual_display_allintakes_combined__newdata = pd.concat([individual_display_allintakes_combined, individual_display_allintakes_combined__newdata])
            intakes_plus_intakes__newdata = intakes.copy()
            for intake, students in intakes__newdata.items():
                if intake in intakes_plus_intakes__newdata:
                    intakes_plus_intakes__newdata[intake] = list(set(
                        intakes_plus_intakes__newdata[intake] + students
                    ))
                else:
                    intakes_plus_intakes__newdata[intake] = students
            extracted_student_ids = individual_display_allintakes_combined__newdata.index.get_level_values('Student_ID')
            individual_display_allintakes_combined__newdata['Intake'] = extracted_student_ids.str[3:7]
            for intake in intakes__newdata:
                combined_df_name = f'individual_display_{intake}_combined__newdata'
                globals()[combined_df_name] = individual_display_allintakes_combined__newdata[
                    individual_display_allintakes_combined__newdata['Intake'] == str(intake)
                ].reset_index()
            for intake in intakes__newdata:
                old_df_name = f'individual_display_{intake}_combined'
                new_df_name = f'individual_display_{intake}_combined__newdata'

                if old_df_name in globals():
                    globals()[old_df_name] = pd.concat(
                        [globals()[old_df_name], globals()[new_df_name]],
                        ignore_index=True
                    )
                else:
                    globals()[old_df_name] = globals()[new_df_name]
            #----------------
            years = [1, 2, 3, 4] 
            yearly_individualdisplay_combined = get_yearly_individual_display(
                intakes_plus_intakes__newdata,
                years
            )
            remove_identical_yearly_data(yearly_individualdisplay_combined, intakes__newdata)
            delete_temp_directory(directory_path__newdata)
            for key, df in yearly_individualdisplay_combined.items():
                yearly_individualdisplay_combined[key] = df.drop_duplicates().reset_index(drop=True)
            for intake in intakes__newdata:
                years = sorted(set(int(key.split('_')[-1]) for key in yearly_individualdisplay_combined.keys() if key.startswith(f'individual_display_{intake}_year_')))
                for year in years:
                    key = f'individual_display_{intake}_year_{year}'
                    print(f"Processing {key} in code block")
                    if key in yearly_individualdisplay_combined:
                        try:
                            print(yearly_individualdisplay_combined[key].dtypes)
                            globals()[f'po_attainment_{intake}_year_{year}'] = calculate_po_attainment(yearly_individualdisplay_combined[key])
                        except Exception as e:
                            print(f"Error processing {key} with calculate_po_attainment")
                            print(f"Exception: {e}")
                            print(f"DataFrame causing the error:\n{yearly_individualdisplay_combined[key]}")
                            raise
            for intake in intakes__newdata:
                years = sorted(set(int(key.split('_')[-1]) for key in yearly_individualdisplay_combined.keys() if key.startswith(f'individual_display_{intake}_year_')))
                for year in years:
                    var_name = f'po_attainment_{intake}_year_{year}'
                    if var_name in globals():
                        print(f'{var_name} is set')
                    else:
                        print(f'{var_name} is not set')
            st.session_state['intakes__newdata'] = intakes__newdata
            st.session_state['yearly_individualdisplay_combined'] = yearly_individualdisplay_combined
            st.success("OBE Forms uploaded successfully. You can proceed to other sections.")

if selected == "Display Predictions":
    st.header("Display Predictions")
    #----
    intakes__newdata = st.session_state['intakes__newdata']
    yearly_individualdisplay_combined = st.session_state['yearly_individualdisplay_combined']
    #----
    model = load_model('plo_prediction_model.keras')
    scaler_X = joblib.load('scaler_X.save')
    scaler_y = joblib.load('scaler_y.save')
    encoder = joblib.load('onehot_encoder.save')
    def convert_percentage_to_float(df, columns):
        for column in columns:
            if df[column].dtype == 'object':
                df[column] = df[column].str.rstrip('%').astype(float) / 100.0
            df[column] = df[column].fillna(0.0)
        return df
    plo_columns = [f'PLO{i}' for i in range(1, 13)]
    predicted_individualdisplay_combined = {}
    displayed_predictions = []
    for intake in intakes__newdata:
        intake_keys = [key for key in yearly_individualdisplay_combined.keys() if key.startswith(f'individual_display_{intake}_year_')]

        if not intake_keys:
            print(f'No data available for intake {intake}.')
            continue
        available_years = []
        for key in intake_keys:
            try:
                year = int(key.split('_year_')[-1])
                available_years.append(year)
            except ValueError:
                continue

        if not available_years:
            print(f'No valid year data for intake {intake}.')
            continue

        max_year = max(available_years)
        next_year = max_year + 1
        data_list = []
        for year in range(1, max_year + 1):
            key = f'individual_display_{intake}_year_{year}'
            if key in yearly_individualdisplay_combined:
                df_year = yearly_individualdisplay_combined[key].copy()
                df_year['Year'] = year
                df_year['Intake'] = intake
                data_list.append(df_year)

        if not data_list:
            print(f'No data to predict for intake {intake}.')
            continue
        df_intake = pd.concat(data_list, ignore_index=True)
        df_intake = convert_percentage_to_float(df_intake, plo_columns)
        course_codes_encoded = encoder.transform(df_intake[['Course_Code']])
        all_course_code_columns = encoder.get_feature_names_out(['Course_Code'])
        course_codes_df = pd.DataFrame(course_codes_encoded, columns=all_course_code_columns)
        df_intake = pd.concat([df_intake, course_codes_df], axis=1)
        missing_cols = set(all_course_code_columns) - set(course_codes_df.columns)
        for col in missing_cols:
            df_intake[col] = 0.0

        pivot_data = df_intake.pivot_table(index=['Student_ID', 'Name', 'Intake'], columns='Year', values=plo_columns + list(all_course_code_columns))

        pivot_data.columns = [f'{col}_Y{year}' for col, year in pivot_data.columns]

        pivot_data.reset_index(inplace=True)
        feature_years = sorted(available_years)
        features = [f'PLO{i}_Y{year}' for year in feature_years for i in range(1, 13)] + [f'{col}_Y{year}' for year in feature_years for col in all_course_code_columns]
        pivot_data[features] = pivot_data[features].fillna(0.0)
        X = pivot_data[features].values
        timesteps = len(feature_years)
        features_per_timestep = len(features) // timesteps
        X = X.reshape(X.shape[0], timesteps, features_per_timestep)

        X_flat = X.reshape(-1, features_per_timestep)

        X_flat_scaled = scaler_X.transform(X_flat)

        X_scaled = X_flat_scaled.reshape(X.shape[0], timesteps, features_per_timestep)

        y_pred_scaled = model.predict(X_scaled)

        y_pred_scaled = y_pred_scaled.reshape(-1, 12)
        print(f'Scaled predictions min: {y_pred_scaled.min()}, max: {y_pred_scaled.max()}')

        y_pred = scaler_y.inverse_transform(y_pred_scaled)

        predicted_data = pivot_data[['Student_ID', 'Name', 'Intake']].copy()
        for i in range(1, 13):
            predicted_data[f'PLO{i}_Y{next_year}'] = y_pred[:, i - 1]

        pred_key = f'individual_display_{intake}_year_{next_year}_predicted'
        predicted_individualdisplay_combined[pred_key] = predicted_data
        if next_year != 5:
            print(f'Predicted DataFrame for {intake} Year {next_year}:')
            display(predicted_individualdisplay_combined[pred_key])
            displayed_predictions.append(pred_key)
