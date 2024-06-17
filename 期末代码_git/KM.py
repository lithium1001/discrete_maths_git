import numpy as np
import pandas as pd
import time

INF = float('inf')

def bfs(favor, val1, val2, match, slack, pre, n, m, p):
    x = 0
    y = 0
    yy = 0
    vis2 = [False] * (m + 1)
    slack = [INF] * (m + 1)
    match[0] = p

    while True:
        vis2[y] = True
        x = match[y]
        d = INF

        for i in range(1, m + 1):
            if not vis2[i]:
                if slack[i] > val1[x] + val2[i] - favor[x][i]:
                    slack[i] = val1[x] + val2[i] - favor[x][i]
                    pre[i] = y
                if slack[i] < d:
                    d = slack[i]
                    yy = i

        for i in range(m + 1):
            if vis2[i]:
                val1[match[i]] -= d
                val2[i] += d
            else:
                slack[i] -= d
        
        y = yy
        if match[y] == 0:
            break
    
    while y:
        match[y] = match[pre[y]]
        y = pre[y]

def KM(favor, n, m):
    match = [0] * (m + 1)
    val1 = [0] * (n + 1)
    val2 = [0] * (m + 1)
    pre = [0] * (m + 1)
    slack = [INF] * (m + 1)

    for i in range(1, n + 1):
        bfs(favor, val1, val2, match, slack, pre, n, m, i)
    
    res = 0
    for i in range(1, m + 1):
        if match[i] > 0:
            res += favor[match[i]][i]
    
    return match, res

def kuhn_munkres(df):
    job_ids = df.index.values
    resume_ids = df.columns[1:].astype(int).values
    n = len(job_ids)
    m = len(resume_ids)

    favor = np.zeros((n + 1, m + 1))
    for i in range(n):
        for j in range(m):
            favor[i + 1][j + 1] = df.iat[i, j + 1]  # +1 to adjust for the ID column
    
    start_time = time.time()
    match, max_weight = KM(favor, n, m)
    end_time = time.time()
    
    print(max_weight)
    matches = []
    for resume in range(1, m + 1):
        job = match[resume]
        if job > 0 and job <= n:  # Ensure job index is within the original matrix dimensions
            matches.append({
                'Job ID': job_ids[job - 1],
                'Resume ID': resume_ids[resume - 1],
                'Match Score': df.iat[job - 1, resume - 1]  # Adjust index to match original DataFrame
            })
    
    return matches, end_time - start_time