# Overview
<p align="justify">
This is a personalized recommender system that I developed as my final project at Institut Teknologi Sepuluh Nopember (ITS). My journey with recommender systems began during a research internship at National Taiwan University of Science and Technology (NTUST), where I was first introduced to how recommendation engines work. From there, I discovered a strong interest in data science, especially in building systems that can adapt to user preferences. 

I have always admired how platforms like Amazon and Netflix recommend products or content that feel surprisingly relevant. They save users from the tedious process of filtering countless options, and most of the time the recommendations are spot on. Inspired by this, I asked myself: How can I design a recommender that goes beyond static methods and adapts more effectively to user preferences? To answer this, I explored Reinforcement Learning (RL), a branch of machine learning that allows systems to update recommendations based on user interactions.

</p>

<p align="justify">
For this project, I implemented the A3C (Asynchronous Advantage Actor-Critic) algorithm to address warm-start issues in recommender systems. By leveraging the advantage function and asynchronous updates, A3C aims to:
</p>

⚡ Reduce gradient variance  
📈 Accelerate learning in large action spaces  
🎯 Improve recommendation accuracy and diversity  

<p align="justify">
In short, this project represents my first step in applying reinforcement learning to build a personalized recommender system that is not only accurate but also adaptive.
</p>

  
---
 
# Problem RecSys
<img src="Images/Recys Problem.png" alt="My Diagram" width="700"/>


**Problem 1 – The Warm Start Challenge**

Even when recommender systems already have historical interaction data from users or items (unlike the cold start problem with no data at all), this information is often not sufficient to generate truly accurate and personalized recommendations. As a result, users still receive suggestions that do not fully reflect their preferences.

**Problem 2 – Limitations of Collaborative Filtering**

Most recommender systems rely on collaborative filtering methods such as Matrix Factorization. While effective in leveraging historical data, these approaches tend to be static and often produce repetitive recommendations. They struggle to capture evolving user preferences over time, which can lead to bias and less adaptive recommendations.

**Problem 3 – Instability of Reinforcement Learning Approaches**

Reinforcement Learning (RL) has been introduced to overcome the limitations of static models by continuously updating recommendations based on user interactions. However, conventional RL algorithms still face significant challenges: the high variance of gradients especially in large action spaces leads to unstable learning and suboptimal recommendations.

**Proposed Solution – A3C (Asynchronous Advantage Actor-Critic)**

To address these challenges, this research proposes the use of the A3C algorithm, which reduces gradient variance through the advantage function and leverages asynchronous parallel updates. This enables the recommender system to adapt more quickly to changing user preferences, stabilize learning, and deliver more accurate and diverse recommendations.

---
# How Recommender A3C works
<img src="Images/A3C CELL.png" alt="My Diagram" width="500"/>

This project applies **A3C (Asynchronous Advantage Actor-Critic)** to build an adaptive recommender system. The workflow can be summarized as follows:  

1. **Environment → State**  
   - User interaction history is modeled as the environment.  
   - The state vector represents the current user context.  

2. **Embedding Layer**  
   - Maps item IDs into a dense representation space.  
   - Captures semantic similarity between items.  

3. **EMGRU (Enhanced Memory GRU)**  
   - Maintains sequential memory of user interactions.  
   - Learns evolving preferences over time.  

4. **Asynchronous Workers**  
   - Multiple agents interact with the environment in parallel.  
   - Each worker updates the global model asynchronously for faster and more stable learning.  

5. **Actor–Critic Networks**  
   - **Policy Network (Actor):** outputs action probabilities (which item to recommend).  
   - **Value Network (Critic):** evaluates the quality of state–action pairs.  
   - **Advantage Function:** stabilizes training by reducing gradient variance.  

6. **Reward and Update**  
   - The system receives rewards based on user feedback (positive or negative).  
   - Actor and Critic are updated to improve future recommendations.  

7. **Output Recommendation**  
   - Produces recommendations that balance short-term relevance with long-term engagement.  


---
# Obtaining the Data
At the beginning of this project, I assumed that finding a well-structured dataset for building a recommender system would be straightforward. Fortunately, I came across the MovieLens 100K dataset
, one of the most widely used benchmark datasets in recommendation research. This dataset contains 100,000 ratings from 943 users on 1,682 movies, making it an excellent starting point for experimenting with different algorithms.

### MovieLens 100K

**📊 Raw Data Preview (MovieLens 100K)**

|No| user_id | movie_id | rating | timestamp  |
|-----|-----|-----|-----|-------|
1| 1       | 50       | 5      | 881250949  |
2| 1       | 172      | 5      | 881250949  |
| :       | :      | :      | :  | : |
99998| 276       | 1090      | 1      | 874795795  |
99999| 13       | 225      | 2      | 882399156  |
10000| 12       | 203       | 3      | 879959583  |



Format  The dataset is organized into several files. The main file `u.data` stores user–item interactions in the form of `(user_id, movie_id, rating, timestamp)`. Additional metadata files include `u.item`, which contains movie titles and genres, and `u.user`, which stores demographic information about users such as age, gender, occupation, and zip code.

Dataset files:  
- **u.data** → user–item interactions (user_id, movie_id, rating, timestamp)  
- **u.item** → movie titles and genres  
- **u.user** → user demographics (age, gender, occupation, zip code)  


### Processing  

To make the dataset usable, I wrote a preprocessing pipeline that:  

- Loads the `raw.dat` file into pandas DataFrames  
- Maps `movie_id` to human-readable titles and genres  
- Normalizes and filters users based on minimum interaction thresholds (e.g., removing users with fewer than 5 ratings)  
- Splits the dataset into training and test sets, preserving the temporal order of interactions to simulate realistic recommendation scenarios  


**🔹Dataset Statistics (Split Comparison)**

<table>
<tr>
<td>

**80% Training – 20% Testing**
|               | Users | Items | Ratings  |
|---------------|-------|-------|----------|
| Training Set  | 754   | 1346  | 80,000   |
| Testing Set   | 189   | 336   | 20,000   |
| **Total**     | 943   | 1682  | 100,000  |

</td>
<td>

**90% Training – 10% Testing**
|               | Users | Items | Ratings  |
|---------------|-------|-------|----------|
| Training Set  | 832   | 1500  | 90,000   |
| Testing Set   | 111   | 170   | 10,000   |
| **Total**     | 943   | 1682  | 100,000  |

</td>
</tr>
</table>

**Challenges**


One limitation of MovieLens 100K is its sparsity with only 100k ratings across nearly 1,700 movies, the user–item matrix is about 93% empty. This makes it difficult for collaborative filtering approaches to perform well, especially for users with very few interactions (the warm start problem). Another challenge is the lack of real-time interaction data. The ratings are static, so simulating an environment for reinforcement learning requires additional design choices.

**Why MovieLens 100K?**

Despite its limitations, the MovieLens 100K dataset provides a clean and reliable benchmark for testing recommender system models. It has been widely used in academic research, which allows me to compare my results with established baselines such as matrix factorization or neural collaborative filtering. More importantly, it serves as a foundation for experimenting with reinforcement learning techniques like A3C, where the challenge is to design a simulator that mimics user feedback dynamically.

# Prerequisite

This codebase was developed and tested with the following environment: 

- Python 3.7.12  
- PyTorch 1.12.1    
- TensorFlow 2.11.0  

 ```bash
conda env create --file environment.yml 
```
# Train
```bash
train_test_A3C.py
```

# Results

<h4>📊 Comparasions Warm-start Dataset MovieLens 100k</h4>

<table>
  <thead>
    <tr>
      <th rowspan="2">Metode</th>
      <th colspan="2">p = 10%</th>
      <th colspan="2">p = 30%</th>
      <th colspan="2">p = 50%</th>
      <th colspan="2">p = 70%</th>
      <th colspan="2">p = 90%</th>
    </tr>
    <tr>
      <th>hit@10</th>
      <th>N@10</th>
      <th>hit@10</th>
      <th>N@10</th>
      <th>hit@10</th>
      <th>N@10</th>
      <th>hit@10</th>
      <th>N@10</th>
      <th>hit@10</th>
      <th>N@10</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="center">Pop-Ranking</td>
      <td align="center">10.96%</td><td align="center">2.90%</td>
      <td align="center">9.49%</td><td align="center">2.16%</td>
      <td align="center">8.15%</td><td align="center">1.65%</td>
      <td align="center">7.41%</td><td align="center">1.31%</td>
      <td align="center">7.52%</td><td align="center">1.48%</td>
    </tr>
    <tr>
      <td align="center">Deep Q-Network</td>
      <td align="center">35.07%</td><td align="center">7.72%</td>
      <td align="center">-</td><td align="center">-</td>
      <td align="center">26.59%</td><td align="center">5.90%</td>
      <td align="center">-</td><td align="center">-</td>
      <td align="center">14.47%</td><td align="center">4.61%</td>
    </tr>
    <tr>
      <td align="center">Session-based RNN</td>
      <td align="center">39.06%</td><td align="center">8.54%</td>
      <td align="center">36.67%</td><td align="center">6.92%</td>
      <td align="center">32.93%</td><td align="center">5.45%</td>
      <td align="center">27.08%</td><td align="center">4.05%</td>
      <td align="center">19.11%</td><td align="center">3.66%</td>
    </tr>
    <tr>
      <td align="center">SLRL (Warm-case)</td>
      <td align="center">53.62%</td><td align="center">14.34%</td>
      <td align="center">50.11%</td><td align="center">12.07%</td>
      <td align="center">46.99%</td><td align="center">9.64%</td>
      <td align="center">43.12%</td><td align="center">7.90%</td>
      <td align="center">30.70%</td><td align="center">6.18%</td>
    </tr>
    <tr>
      <td align="center"><b>A3C 80% Train 20% Test</b></td>
      <td align="center"><b>55.18%</b></td><td align="center"><b>14.52%</b></td>
      <td align="center"><b>53.21%</b></td><td align="center"><b>12.92%</b></td>
      <td align="center"><b>47.74%</b></td><td align="center"><b>10.04%</b></td>
      <td align="center"><b>43.35%</b></td><td align="center"><b>7.99%</b></td>
      <td align="center"><b>28.80%</b></td><td align="center"><b>5.67%</b></td>
    </tr>
    <tr>
      <td align="center"><b>A3C 90% Train 10% Test</b></td>
      <td align="center"><b>56.43%</b></td><td align="center"><b>14.36%</b></td>
      <td align="center"><b>52.40%</b></td><td align="center"><b>12.53%</b></td>
      <td align="center"><b>48.30%</b></td><td align="center"><b>9.96%</b></td>
      <td align="center"><b>44.84%</b></td><td align="center"><b>8.28%</b></td>
      <td align="center"><b>32.12%</b></td><td align="center"><b>6.51%</b></td>
    </tr>
  </tbody>
</table>



### Performance Results
- **Hit@10 (P=10%)**
  - A3C (80/20 split): **55.18%** | NDCG@10: **14.52%**
  - A3C (90/10 split): **56.43%** | NDCG@10: **14.36%**
- **Compared to baselines**
  - Outperforms **SLRL** by **+3–4% Hit@10**
  - Significantly better than **DQN, sRNN, and Pop-Ranking**


### Stability and Generalization
- A3C consistently performs better across all **P values (10%–90%)**  
- High recall at **P=10–30%**, capturing user preferences even with limited history  
- Overfitting observed at **P=70–90%**, as performance drops when the model relies too much on long historical data  



### Performance Drop at Higher P%
- **80/20 split**  
  - Hit@10: **43.35% → 28.80%** (↓ 14.55%)  
  - NDCG@10: **7.99% → 5.67%** (↓ 2.32%)  
- **90/10 split**  
  - Hit@10: **44.84% → 32.12%** (↓ 12.72%)  
  - NDCG@10: **8.28% → 6.51%** (↓ 1.77%)  

**Reason:** The MovieLens 100K dataset is dominated by **popular items**. As P increases, recommendations become biased toward general consumption patterns, reducing personalization.  



### Key Insights
- **A3C outperforms** all baseline methods (Pop-Ranking, DQN, sRNN, SLRL)  
- **Strong in warm-start scenarios**: delivers relevant recommendations with sparse user history  
- **Needs overfitting mitigation** at high P values (e.g., via **regularization, data augmentation, or diversity-aware sampling**)  

