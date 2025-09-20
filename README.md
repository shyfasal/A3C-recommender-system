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
| :       | :      | :      | :  |
99998| 276       | 1090      | 1      | 874795795  |
99999| 13       | 225      | 2      | 882399156  |
10000| 12       | 203       | 3      | 879959583  |



Format  The dataset is organized into several files. The main file `u.data` stores user–item interactions in the form of `(user_id, movie_id, rating, timestamp)`. Additional metadata files include `u.item`, which contains movie titles and genres, and `u.user`, which stores demographic information about users such as age, gender, occupation, and zip code.



**Processing**  

To make the dataset usable, I wrote a preprocessing pipeline that:
1.Loads the `raw.dat` file into pandas DataFrames.
2.Maps `movie_id` to human-readable titles and genres.
3.Normalizes and filters users based on minimum interaction thresholds (e.g., removing users with fewer than 5 ratings).
4.Splits the dataset into training and test sets, preserving the temporal order of interactions to simulate realistic recommendation scenarios.

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


One limitation of MovieLens 100K is its sparsity with only 100k ratings across nearly 1,700 movies, the user–item matrix is about 93% empty. This makes it difficult for collaborative filtering approaches to perform well, especially for users with very few interactions (the warm start problem). Another challenge is the lack of real-time interaction data — the ratings are static, so simulating an environment for reinforcement learning requires additional design choices.

**Why MovieLens 100K?**

Despite its limitations, the MovieLens 100K dataset provides a clean and reliable benchmark for testing recommender system models. It has been widely used in academic research, which allows me to compare my results with established baselines such as matrix factorization or neural collaborative filtering. More importantly, it serves as a foundation for experimenting with reinforcement learning techniques like A3C, where the challenge is to design a simulator that mimics user feedback dynamically.


