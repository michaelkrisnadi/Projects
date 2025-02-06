import pandas as pd

def extract_student_row(data, student_id):
    course_info = data['course_info']
    df2 = data['student_plo_df']
    
    # Ensure the DataFrame has expected columns
    if 'Student_ID' not in df2.columns:
        raise ValueError("DataFrame does not have 'Student_ID' column")
    student_row = df2.loc[df2['Student_ID'] == student_id]
    
    if student_row.empty:
        return None, None
    
    student_name = student_row['Student_Name'].values[0]
    student_row = student_row.drop(columns=['Student_ID', 'Student_Name'])
    student_row = student_row.assign(
        Course_Code=course_info.get('Course Code:', ''),
        Course_Name=course_info.get('Course Name:', ''),
        Credit_Value=course_info.get('Credit Value:', ''),
        Academic_Session=course_info.get('Academic Session', '')
    )
    
    # Create a new list of columns
    cols = ['Course_Code', 'Course_Name', 'Academic_Session', 'Credit_Value'] + [col for col in student_row.columns if col not in ['Course_Code', 'Course_Name', 'Academic_Session', 'Credit_Value']]
    student_row = student_row[cols]
    
    return student_row, student_name

def individual_display_row(studentID, file_list, results_dict):

    student_id = studentID
    all_student_rows = pd.DataFrame()

    # Loop over the specified files
    for file_name in file_list:
        data = results_dict.get(file_name)
        if data is None:
            continue
        student_row, student_name = extract_student_row(data, student_id)
        if student_row is None:
            continue
        if isinstance(student_row, pd.Series):
            student_row = student_row.to_frame().T
        all_student_rows = pd.concat([all_student_rows, student_row], ignore_index=True)

    if all_student_rows.empty:
        print(f"No data found for student ID {student_id}")
        return student_id, None, all_student_rows

    # List of PLO columns
    plo_cols = [col for col in all_student_rows.columns if col.startswith('PLO')]
    categories_plo_cols = [col for col in all_student_rows.columns if col.startswith('Category_') or col.startswith('Categories ')]
    cols = ['Course_Code', 'Course_Name', 'Academic_Session', 'Credit_Value'] + plo_cols + categories_plo_cols
    all_student_rows = all_student_rows[cols]

    return student_id, student_name, all_student_rows
