import re
import pandas as pd
import streamlit as st
import numpy as np
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
import random

job_data = pd.read_csv('job_data_with_id.csv')  
resume_data = pd.read_csv('resume_dataset_with_id.csv')
# random_sampled_jobs = job_data.sample(n=147, random_state=1).reset_index(drop=True)
# random_sampled_jobs.to_csv('random_sampled_jobs.csv', index=False)
random_sampled_jobs = pd.read_csv('random_sampled_jobs.csv')

###############################################################抽取特征##########################################################3
def preprocess_text(text):
    if not isinstance(text, str):
        text = str(text)
    text = re.sub(r'\W', ' ', text)  # 移除非字母字符
    text = re.sub(r'\s+', ' ', text)  # 移除多余的空格
    text = text.lower()  # 转换为小写
    return text

def extract_industry(resume_text, category):
    industry_keywords = {
        'Consumer Products Manufacturing': ['consumer products manufacturing'],
        'Internet': ['internet'],
        'Electrical & Electronic Manufacturing': ['electrical manufacturing', 'electronic manufacturing'],
        'Industrial Manufacturing': ['industrial manufacturing'],
        'Computer Hardware & Software': ['computer hardware', 'computer software', 'software', 'hardware'],
        'Oil & Gas Exploration & Production': ['oil exploration', 'gas exploration', 'oil production', 'gas production'],
        'Investment Banking & Asset Management': ['investment banking', 'asset management'],
        'Chemical Manufacturing': ['chemical manufacturing'],
        'Insurance Agencies & Brokerages': ['insurance agencies', 'brokerages'],
        'Advertising & Marketing': ['advertising', 'marketing'],
        'Telecommunications Manufacturing': ['telecommunications manufacturing'],
        'Biotech & Pharmaceuticals': ['biotech', 'pharmaceuticals'],
        'Energy': ['energy'],
        'Transportation Equipment Manufacturing': ['transportation equipment manufacturing'],
        'Health Care Products Manufacturing': ['health care products manufacturing'],
        'Insurance Carriers': ['insurance carriers'],
        'IT Services': ['it services', 'information technology services'],
        'Department, Clothing, & Shoe Stores': ['department stores', 'clothing stores', 'shoe stores'],
        'Food & Beverage Manufacturing': ['food manufacturing', 'beverage manufacturing'],
        'Research & Development': ['research and development', 'r&d'],
        'Hotels, Motels, & Resorts': ['hotels', 'motels', 'resorts'],
        'Video Games': ['video games'],
        'Travel Agencies': ['travel agencies'],
        'Consulting': ['consulting'],
        'Logistics & Supply Chain': ['logistics', 'supply chain'],
        'Museums, Zoos & Amusement Parks': ['museums', 'zoos', 'amusement parks'],
        'Financial Analytics & Research': ['financial analytics', 'financial research'],
        'Enterprise Software & Network Solutions': ['enterprise software', 'network solutions']
    }

    resume_text_lower = resume_text.lower()
    category_lower = category.lower()
    
    for industry, keywords in industry_keywords.items():
        for keyword in keywords:
            if keyword in resume_text_lower or keyword in category_lower:
                return industry
    return 'Unknown'

def extract_sector(resume_text):
    sector_keywords = {
        'Manufacturing': ['manufacturing'],
        'Information Technology': ['information technology', 'it'],
        'Oil, Gas, Energy & Utilities': ['oil', 'gas', 'energy', 'utilities'],
        'Finance': ['finance'],
        'Insurance': ['insurance'],
        'Business Services': ['business services'],
        'Telecommunications': ['telecommunications'],
        'Biotech & Pharmaceuticals': ['biotech', 'pharmaceuticals'],
        'Retail': ['retail'],
        'Travel & Tourism': ['travel', 'tourism'],
        'Media': ['media'],
        'Transportation & Logistics': ['transportation', 'logistics'],
        'Arts, Entertainment & Recreation': ['arts', 'entertainment', 'recreation']
    }
    
    for sector, keywords in sector_keywords.items():
        for keyword in keywords:
            if keyword.lower() in resume_text.lower():
                return sector
    return 'Unknown'

def extract_job_title(resume_text):
    # 枚举常见职位的关键字
    job_title_keywords = [
        'Product Manager', 'Business Development', 'Project Manager', 
        'Software Engineer', 'Data Scientist', 'Financial Analyst', 
        'Quality Engineer', 'Research Scientist', 'Account Manager', 
        'Marketing Manager', 'Operations Manager'
    ]
    
    # 使用正则表达式匹配关键字
    pattern = r'\b(' + '|'.join(job_title_keywords) + r')\b'
    match = re.search(pattern, resume_text, re.I)
    return match.group() if match else 'Unknown'

def extract_salary(resume_text):
    match = re.search(r'Salary:\s*([\d,]+)', resume_text)
    return match.group(1) if match else 'Unknown'

def map_resume_to_job(row):
    resume = row['Resume']
    category = row['Category']
    cleaned_resume = preprocess_text(resume)
    mapped_data = {
        'ID': row['ID'],
        'Resume': resume,
        'salary': extract_salary(cleaned_resume),
        'jobTitle': extract_job_title(cleaned_resume),
        'industry': extract_industry(cleaned_resume, category),
        'sector': extract_sector(cleaned_resume),
        'location': row['Location']
    }
    return mapped_data

# 应用 map_resume_to_job 函数
mapped_resumes = resume_data.apply(map_resume_to_job, axis=1)
mapped_df = pd.DataFrame(mapped_resumes.tolist())


# @st.cache_data
def compute_matching_weights(job_data, resume_data):
    job_fields = ['Translated_Desc', 'gaTrackerData.industry', 'gaTrackerData.sector', 'header.jobTitle', 'header.location', 'salary.salaries']
    resume_fields = ['Resume', 'industry', 'sector', 'jobTitle', 'location', 'salary']
    field_weights = {'Resume': 0.2, 'industry': 0.5, 'sector': 0.5, 'jobTitle': 0.5, 'location': 0.5, 'salary': 0.1}
    
    # 初始化结果矩阵
    final_scores = np.zeros((len(job_data), len(resume_data)))
    
    # 分别处理每个字段
    for job_field, resume_field in zip(job_fields, resume_fields):
        # 预处理字段文本
        job_texts = job_data[job_field].apply(preprocess_text)
        resume_texts = resume_data[resume_field].apply(preprocess_text)
        
        # 计算TF-IDF特征向量
        vectorizer = TfidfVectorizer(stop_words='english')
        all_texts = job_texts.tolist() + resume_texts.tolist()
        tfidf_matrix = vectorizer.fit_transform(all_texts)
        
        # 分离职位描述和简历的特征向量
        tfidf_job_desc = tfidf_matrix[:len(job_texts)]
        tfidf_resumes = tfidf_matrix[len(job_texts):]
        
        # 计算余弦相似度
        cosine_sim_matrix = cosine_similarity(tfidf_job_desc, tfidf_resumes)
        
        # 更新最终分数矩阵
        weight = field_weights[resume_field]
        for i in range(len(job_data)):
            for j in range(len(resume_data)):
                field_score = cosine_sim_matrix[i, j]
                if job_data.iloc[i][job_field] == 'unknown' or resume_data.iloc[j][resume_field] == 'unknown':
                    weight *= 0.5  # 如果字段是 'Unknown'，权重减半
                final_scores[i, j] += field_score * weight
    
    # 转换为DataFrame并添加职位和简历的ID
    final_scores_df = pd.DataFrame(final_scores, index=job_data['ID'], columns=resume_data['ID'])
    return final_scores_df

final_scores = compute_matching_weights(random_sampled_jobs, mapped_df)

final_scores.to_csv('resume_job_matching_scores.csv', index=False)

