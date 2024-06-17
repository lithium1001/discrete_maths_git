import pandas as pd
import streamlit as st
from time import time
from KM import kuhn_munkres
import sklearn
from cal_sim import compute_matching_weights

st.set_page_config(page_title="Job-Resume Matching System", page_icon=":bar_chart:", layout="wide")

# 初始化会话状态中的数据集
if 'resume_data' not in st.session_state:
    st.session_state.resume_data = pd.read_csv(r'data\mapped_resume_dataset.csv')
if 'job_data' not in st.session_state:
    st.session_state.job_data = pd.read_csv(r'data\job_data_with_id.csv')
if 'random_sampled_jobs' not in st.session_state:
    st.session_state.random_sampled_jobs = pd.read_csv(r'data\random_sampled_jobs.csv')

resume_data = st.session_state.resume_data
job_data = st.session_state.job_data
random_sampled_jobs = st.session_state.random_sampled_jobs

st.sidebar.header("Add New Entry")
entry_type = st.sidebar.radio("Select Entry Type", ( 'Resume','Job'), horizontal=True)

with st.sidebar.form(key='new_entry_form'):
    new_entry = {}
    if entry_type == 'Job':
        new_entry['gaTrackerData.industry'] = st.text_input("Industry", "")
        new_entry['gaTrackerData.sector'] = st.text_input("Sector", "")
        new_entry['header.jobTitle'] = st.text_input("Job Title", "")
        new_entry['header.location'] = st.text_input("Location", "")
        new_entry['salary.salaries'] = st.text_input("Salary", "")        
        new_entry['Translated_Desc'] = st.text_area("Job Description", "")
        new_entry['ID'] = job_data['ID'].max() + 1 if not job_data.empty else 1
    else:
        new_entry['industry'] = st.text_input("Industry", "")
        new_entry['sector'] = st.text_input("Sector", "")
        new_entry['jobTitle'] = st.text_input("Job Title", "")
        new_entry['location'] = st.text_input("Location", "")
        new_entry['salary'] = st.text_input("Salary", "")        
        new_entry['Resume'] = st.text_area("Resume", "")
        new_entry['ID'] = resume_data['ID'].max() + 1 if not resume_data.empty else 1

    submit_button = st.form_submit_button(label='Add Entry and Match')

@st.cache_data
def display_matching_results(job_data, resume_data, random_sampled_jobs):
    # 计算相似度矩阵
    start_time = time()
    similarity_matrix = compute_matching_weights(random_sampled_jobs, resume_data)
    end_time = time()

    st.write(f"Time taken to compute similarity matrix: {end_time - start_time:.2f} seconds")

    matches, runtime = kuhn_munkres(similarity_matrix)

    matches_df = pd.DataFrame(matches)

    job_len = len(random_sampled_jobs)
    resume_len = len(resume_data)

    job_rate = len(matches_df) * 100 / job_len
    resume_rate = len(matches_df) * 100 / resume_len

    # 构建匹配结果，并展示职位和简历的详细信息
    match_details = []
    for _, match in matches_df.iterrows():
        job_id = int(match['Job ID'])
        resume_id = int(match['Resume ID'])
        match_score = match['Match Score']

        job_info = job_data[job_data['ID'] == job_id].iloc[0]
        resume_info = resume_data[resume_data['ID'] == resume_id].iloc[0]
        match_details.append({
            'Job ID': job_id,
            'Job Title': job_info['header.jobTitle'],
            'Employer': job_info.get('header.employerName', 'N/A'),
            'Job Industry': job_info['gaTrackerData.industry'],
            'Job Sector': job_info['gaTrackerData.sector'],
            'Job Location': job_info['header.location'],
            'Resume ID': resume_id,
            'Resume Experience': resume_info.get('jobTitle', 'Unknown'),
            'Resume Location': resume_info['location'],
            'Resume Description': resume_info['Resume'][:200],
            'Match Score': match_score
        })

    # 转换为DataFrame
    matches_df = pd.DataFrame(match_details)

    return matches_df, runtime, job_rate, resume_rate, job_len, resume_len, similarity_matrix

def show_results(matches_df, runtime, job_rate, resume_rate, new_entry_id, entry_type, job_len, resume_len, similarity_matrix):
    # 标题和描述
    st.markdown("## Job-Resume Matching Results")
    st.markdown("#### This table shows the best matches between job postings and resumes using the **Kuhn-Munkres** algorithm.")
    st.subheader(f"Algorithm Execution Time: {runtime:.2f} seconds")

    st.write(f"Of the {job_len} jobs, {job_rate:.2f}% of job postings were matched with resumes.")
    st.write(f"Of the {resume_len} resumes, {resume_rate:.2f}% of resumes were matched with job postings.")

    # 显示匹配结果
    st.dataframe(matches_df.style.format({
        'Match Score': '{:.4f}'
    }).set_properties(**{
        'background-color': 'white',
        'color': 'black',
        'border-color': 'black',
        'text-align': 'left',
    }))

    # 过滤最新条目的匹配结果
    if new_entry_id is not None and entry_type is not None:
        if entry_type == 'Job':
            filtered_matches = matches_df[matches_df['Job ID'] == new_entry_id]
            if filtered_matches.empty:
                filtered_matches = pd.DataFrame([new_entry])
                st.warning("No matches found for the new entry. Showing the new entry instead.")
        else:
            filtered_matches = matches_df[matches_df['Resume ID'] == new_entry_id]
            if filtered_matches.empty:
                filtered_matches = pd.DataFrame([new_entry])
                st.warning("No matches found for the new entry. Showing the new entry instead.")

        st.markdown("## Latest Entry Matching Result")
        st.dataframe(filtered_matches.style.format({
            'Match Score': '{:.4f}'
        }).set_properties(**{
            'background-color': 'white',
            'color': 'black',
            'border-color': 'black',
            'text-align': 'left',
        }))



# 在提交按钮被按下时调用 show_results 函数时，传递正确的参数
if not submit_button:
    matches_df, runtime, job_rate, resume_rate, job_len, resume_len, similarity_matrix = display_matching_results(job_data, resume_data, random_sampled_jobs)
    show_results(matches_df, runtime, job_rate, resume_rate, None, None, job_len, resume_len, similarity_matrix)

if submit_button:
    all_fields_filled = all(value for value in new_entry.values())

    if all_fields_filled:
        new_entry_df = pd.DataFrame([new_entry])
        if entry_type == 'Job':
            st.session_state.job_data = pd.concat([st.session_state.job_data, new_entry_df], ignore_index=True)
            st.session_state.random_sampled_jobs = pd.concat([st.session_state.random_sampled_jobs, new_entry_df], ignore_index=True)
        else:
            st.session_state.resume_data = pd.concat([st.session_state.resume_data, new_entry_df], ignore_index=True)

        matches_df, runtime, job_rate, resume_rate, job_len, resume_len, similarity_matrix = display_matching_results(st.session_state.job_data, st.session_state.resume_data, st.session_state.random_sampled_jobs)
        show_results(matches_df, runtime, job_rate, resume_rate, new_entry['ID'], entry_type, job_len, resume_len, similarity_matrix)
    else:
        matches_df, runtime, job_rate, resume_rate, job_len, resume_len, similarity_matrix = display_matching_results(job_data, resume_data, random_sampled_jobs)
        show_results(matches_df, runtime, job_rate, resume_rate, None, None, job_len, resume_len, similarity_matrix)
        st.sidebar.warning("Please fill in all fields.")
