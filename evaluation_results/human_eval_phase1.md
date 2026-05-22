# Phase 1 Human-Eval Check-in

**Purpose:** before locking in the Phase 1 re-ranker weights, scan a sample of individual and group recommendations and flag any that look off. The offline metrics tell us the re-ranker improved (see the table at the bottom), but only you can say if the *vibes* are right.

Models used: `models/svdpp.pkl` (ml-latest-small) + `models/als_small.pkl` (ml-latest-small).
User: `alex_data/ratings_with_tmdb.csv` (116 ratings mapped to MovieLens).
Synthetic friends: two random MovieLens users with 517 and 105 ratings respectively.

---

## Individual recommendations — by mode

### Mode: `balanced`

| # | Score | Pop | Title | Genre overlap | Source |
|--:|------:|:---:|:------|:--------------|:------:|
| 1 | 1.706 | blockbuster | Pulp Fiction (1994) | Drama, Thriller, Crime | svd+als |
| 2 | 1.421 | popular | Lives of Others, The (Das leben der Anderen) (2006) | Drama, Thriller, Romance | svd+als |
| 3 | 1.344 | popular | Brazil (1985) | Fantasy, Sci-Fi | svd+als |
| 4 | 1.274 | blockbuster | Seven (a.k.a. Se7en) (1995) | Thriller, Mystery | svd+als |
| 5 | 1.248 | obscure | Master of the Flying Guillotine (Du bi quan wang da po xue di zi) (1975) | Action | svd |
| 6 | 1.234 | obscure | Open Season (2006) | Adventure, Comedy, Animation | svd |
| 7 | 1.178 | popular | Pan's Labyrinth (Laberinto del fauno, El) (2006) | Drama, Thriller, Fantasy | svd+als |
| 8 | 1.175 | popular | Birds, The (1963) | Thriller, Horror | svd+als |
| 9 | 1.139 | obscure | Justice League: Doom (2012)  | Action, Fantasy, Animation | svd |
| 10 | 1.131 | obscure | Flight of the Phoenix, The (1965) | Drama, Action, Adventure | svd |

### Mode: `niche`

| # | Score | Pop | Title | Genre overlap | Source |
|--:|------:|:---:|:------|:--------------|:------:|
| 1 | 1.074 | obscure | Prime Suspect 2 (1992) | Drama, Thriller, Crime | svd |
| 2 | 0.982 | obscure | Justice League: Doom (2012)  | Action, Fantasy, Animation | svd |
| 3 | 0.916 | obscure | Law of Desire (Ley del deseo, La) (1987) | Drama, Comedy, Romance | svd |
| 4 | 0.900 | obscure | 12 Angry Men (1997) | Drama, Crime | svd |
| 5 | 0.889 | obscure | Open Season (2006) | Adventure, Comedy, Animation | svd |
| 6 | 0.876 | obscure | Flight of the Phoenix, The (1965) | Drama, Action, Adventure | svd |
| 7 | 0.851 | obscure | Survive Style 5+ (2004) | Thriller, Fantasy, Romance | svd |
| 8 | 0.843 | obscure | What Ever Happened to Baby Jane? (1962) | Drama, Thriller, Horror | svd |
| 9 | 0.834 | obscure | Master of the Flying Guillotine (Du bi quan wang da po xue di zi) (1975) | Action | svd |
| 10 | 0.819 | obscure | Blind Spot: Hitler's Secretary (Im toten Winkel - Hitlers Sekretärin) (2002) | Documentary | svd |

### Mode: `popular`

| # | Score | Pop | Title | Genre overlap | Source |
|--:|------:|:---:|:------|:--------------|:------:|
| 1 | 2.415 | blockbuster | Pulp Fiction (1994) | Drama, Thriller, Crime | svd+als |
| 2 | 1.994 | blockbuster | Seven (a.k.a. Se7en) (1995) | Thriller, Mystery | svd+als |
| 3 | 1.959 | popular | Lives of Others, The (Das leben der Anderen) (2006) | Drama, Thriller, Romance | svd+als |
| 4 | 1.875 | popular | Brazil (1985) | Fantasy, Sci-Fi | svd+als |
| 5 | 1.788 | popular | Pan's Labyrinth (Laberinto del fauno, El) (2006) | Drama, Thriller, Fantasy | svd+als |
| 6 | 1.772 | popular | Terminator, The (1984) | Thriller, Action, Sci-Fi | svd+als |
| 7 | 1.766 | blockbuster | Usual Suspects, The (1995) | Thriller, Crime, Mystery | svd+als |
| 8 | 1.762 | popular | Birds, The (1963) | Thriller, Horror | svd+als |
| 9 | 1.720 | blockbuster | Godfather, The (1972) | Drama, Crime | svd+als |
| 10 | 1.697 | blockbuster | Star Wars: Episode IV - A New Hope (1977) | Action, Adventure, Sci-Fi | svd+als |

### Mode: `serendipitous`

| # | Score | Pop | Title | Genre overlap | Source |
|--:|------:|:---:|:------|:--------------|:------:|
| 1 | 1.125 | popular | Lives of Others, The (Das leben der Anderen) (2006) | Drama, Thriller, Romance | svd+als |
| 2 | 0.952 | obscure | Justice League: Doom (2012)  | Action, Fantasy, Animation | svd |
| 3 | 0.869 | obscure | Live Nude Girls (1995) | Comedy | svd |
| 4 | 0.784 | obscure | Flight of the Phoenix, The (1965) | Drama, Action, Adventure | svd |
| 5 | 0.780 | obscure | Blind Spot: Hitler's Secretary (Im toten Winkel - Hitlers Sekretärin) (2002) | Documentary | svd |
| 6 | 0.779 | obscure | Android (1982) | Sci-Fi | svd |
| 7 | 0.748 | obscure | Long Goodbye, The (1973) | Crime, Film-Noir | svd |
| 8 | 0.744 | obscure | Open Season (2006) | Adventure, Comedy, Animation | svd |
| 9 | 0.727 | obscure | Unsinkable Molly Brown, The (1964) | Musical | svd |
| 10 | 0.704 | blockbuster | Pulp Fiction (1994) | Drama, Thriller, Crime | svd+als |

---

## Group recommendations — 3-friend group, mode=`balanced`

Each row shows the predicted score per member (normalized 0-1) and the fairness coefficient of variation (lower = more uniform agreement).

### Strategy: `average`

| # | Score | Fair | Title | Pop | Per-member scores |
|--:|------:|-----:|:------|:---:|:------------------|
| 1 | 0.797 | 0.18 | Pulp Fiction (1994) | blockbuster | alex=1.00, friend_A=0.66, friend_B=0.74 |
| 2 | 0.743 | 0.25 | Bellflower (2011) | obscure | alex=0.49, friend_A=0.93, friend_B=0.81 |
| 3 | 0.739 | 0.17 | Clockwork Orange, A (1971) | popular | alex=0.57, friend_A=0.86, friend_B=0.79 |
| 4 | 0.732 | 0.31 | Graduate, The (1967) | popular | alex=0.41, friend_A=0.92, friend_B=0.86 |
| 5 | 0.722 | 0.16 | Dark Knight, The (2008) | popular | alex=0.62, friend_A=0.88, friend_B=0.67 |
| 6 | 0.709 | 0.28 | Body Snatcher, The (1945) | obscure | alex=0.45, friend_A=0.93, friend_B=0.75 |
| 7 | 0.695 | 0.09 | Prime Suspect 2 (1992) | obscure | alex=0.67, friend_A=0.63, friend_B=0.79 |
| 8 | 0.692 | 0.06 | Casablanca (1942) | popular | alex=0.65, friend_A=0.75, friend_B=0.68 |
| 9 | 0.688 | 0.16 | Eternal Sunshine of the Spotless Mind (2004) | popular | alex=0.57, friend_A=0.66, friend_B=0.84 |
| 10 | 0.686 | 0.08 | Survive Style 5+ (2004) | obscure | alex=0.65, friend_A=0.64, friend_B=0.77 |

### Strategy: `least_misery`

| # | Score | Fair | Title | Pop | Per-member scores |
|--:|------:|-----:|:------|:---:|:------------------|
| 1 | 0.656 | 0.18 | Pulp Fiction (1994) | blockbuster | alex=1.00, friend_A=0.66, friend_B=0.74 |
| 2 | 0.646 | 0.06 | Casablanca (1942) | popular | alex=0.65, friend_A=0.75, friend_B=0.68 |
| 3 | 0.640 | 0.08 | Survive Style 5+ (2004) | obscure | alex=0.65, friend_A=0.64, friend_B=0.77 |
| 4 | 0.634 | 0.09 | Prime Suspect 2 (1992) | obscure | alex=0.67, friend_A=0.63, friend_B=0.79 |
| 5 | 0.630 | 0.01 | Breaking Away (1979) | niche | alex=0.63, friend_A=0.65, friend_B=0.63 |
| 6 | 0.617 | 0.16 | Dark Knight, The (2008) | popular | alex=0.62, friend_A=0.88, friend_B=0.67 |
| 7 | 0.603 | 0.14 | Girltrash: All Night Long (2014) | obscure | alex=0.63, friend_A=0.60, friend_B=0.82 |
| 8 | 0.602 | 0.02 | What Ever Happened to Baby Jane? (1962) | obscure | alex=0.63, friend_A=0.61, friend_B=0.60 |
| 9 | 0.575 | 0.04 | Master of the Flying Guillotine (Du bi quan wang da po xue di zi) (1975) | obscure | alex=0.63, friend_A=0.57, friend_B=0.59 |
| 10 | 0.575 | 0.11 | Kung Fury (2015) | niche | alex=0.57, friend_A=0.64, friend_B=0.75 |

### Strategy: `most_pleasure`

| # | Score | Fair | Title | Pop | Per-member scores |
|--:|------:|-----:|:------|:---:|:------------------|
| 1 | 1.000 | 0.76 | All-Star Superman (2011) | obscure | alex=0.37, friend_A=0.11, friend_B=1.00 |
| 2 | 1.000 | 0.18 | Pulp Fiction (1994) | blockbuster | alex=1.00, friend_A=0.66, friend_B=0.74 |
| 3 | 1.000 | 0.39 | Tiger and the Snow, The (La tigre e la neve) (2005) | obscure | alex=0.41, friend_A=1.00, friend_B=0.55 |
| 4 | 0.986 | 0.32 | Son of the Bride (Hijo de la novia, El) (2001) | obscure | alex=0.49, friend_A=0.99, friend_B=0.57 |
| 5 | 0.961 | 0.64 | Slumdog Millionaire (2008) | popular | alex=0.10, friend_A=0.58, friend_B=0.96 |
| 6 | 0.960 | 0.50 | Day of the Doctor, The (2013) | niche | alex=0.24, friend_A=0.96, friend_B=0.55 |
| 7 | 0.960 | 0.57 | Wild Tales (2014) | niche | alex=0.26, friend_A=0.96, friend_B=0.39 |
| 8 | 0.957 | 0.32 | Usual Suspects, The (1995) | blockbuster | alex=0.64, friend_A=0.96, friend_B=0.43 |
| 9 | 0.931 | 0.30 | Rabbits (2002) | obscure | alex=0.55, friend_A=0.93, friend_B=0.48 |
| 10 | 0.929 | 0.25 | Bellflower (2011) | obscure | alex=0.49, friend_A=0.93, friend_B=0.81 |

### Strategy: `consensus`

| # | Score | Fair | Title | Pop | Per-member scores |
|--:|------:|-----:|:------|:---:|:------------------|
| 1 | 0.775 | 0.18 | Pulp Fiction (1994) | blockbuster | alex=1.00, friend_A=0.66, friend_B=0.74 |
| 2 | 0.724 | 0.17 | Clockwork Orange, A (1971) | popular | alex=0.57, friend_A=0.86, friend_B=0.79 |
| 3 | 0.709 | 0.16 | Dark Knight, The (2008) | popular | alex=0.62, friend_A=0.88, friend_B=0.67 |
| 4 | 0.708 | 0.25 | Bellflower (2011) | obscure | alex=0.49, friend_A=0.93, friend_B=0.81 |
| 5 | 0.691 | 0.09 | Prime Suspect 2 (1992) | obscure | alex=0.67, friend_A=0.63, friend_B=0.79 |
| 6 | 0.690 | 0.06 | Casablanca (1942) | popular | alex=0.65, friend_A=0.75, friend_B=0.68 |
| 7 | 0.682 | 0.08 | Survive Style 5+ (2004) | obscure | alex=0.65, friend_A=0.64, friend_B=0.77 |
| 8 | 0.680 | 0.31 | Graduate, The (1967) | popular | alex=0.41, friend_A=0.92, friend_B=0.86 |
| 9 | 0.676 | 0.16 | Eternal Sunshine of the Spotless Mind (2004) | popular | alex=0.57, friend_A=0.66, friend_B=0.84 |
| 10 | 0.675 | 0.14 | Girltrash: All Night Long (2014) | obscure | alex=0.63, friend_A=0.60, friend_B=0.82 |

### Strategy: `hybrid`

| # | Score | Fair | Title | Pop | Per-member scores |
|--:|------:|-----:|:------|:---:|:------------------|
| 1 | 1.125 | 0.18 | Pulp Fiction (1994) | blockbuster | alex=1.00, friend_A=0.66, friend_B=0.74 |
| 2 | 1.030 | 0.16 | Dark Knight, The (2008) | popular | alex=0.62, friend_A=0.88, friend_B=0.67 |
| 3 | 1.024 | 0.17 | Clockwork Orange, A (1971) | popular | alex=0.57, friend_A=0.86, friend_B=0.79 |
| 4 | 1.015 | 0.06 | Casablanca (1942) | popular | alex=0.65, friend_A=0.75, friend_B=0.68 |
| 5 | 1.012 | 0.09 | Prime Suspect 2 (1992) | obscure | alex=0.67, friend_A=0.63, friend_B=0.79 |
| 6 | 1.006 | 0.08 | Survive Style 5+ (2004) | obscure | alex=0.65, friend_A=0.64, friend_B=0.77 |
| 7 | 0.987 | 0.25 | Bellflower (2011) | obscure | alex=0.49, friend_A=0.93, friend_B=0.81 |
| 8 | 0.985 | 0.14 | Girltrash: All Night Long (2014) | obscure | alex=0.63, friend_A=0.60, friend_B=0.82 |
| 9 | 0.971 | 0.16 | Eternal Sunshine of the Spotless Mind (2004) | popular | alex=0.57, friend_A=0.66, friend_B=0.84 |
| 10 | 0.960 | 0.19 | Marvel One-Shot: Agent Carter (2013) | obscure | alex=0.56, friend_A=0.86, friend_B=0.63 |

### Strategy: `group_taste_vector`

| # | Score | Fair | Title | Pop | Per-member scores |
|--:|------:|-----:|:------|:---:|:------------------|
| 1 | 1.609 | 0.16 | Casablanca (1942) | popular | alex=1.67, friend_A=1.12, friend_B=1.45 |
| 2 | 1.598 | 0.27 | Spider-Man 2 (2004) | popular | alex=0.79, friend_A=1.29, friend_B=0.73 |
| 3 | 1.326 | 0.39 | Barton Fink (1991) | niche | alex=1.83, friend_A=0.88, friend_B=0.83 |
| 4 | 1.326 | 0.26 | Simpsons Movie, The (2007) | popular | alex=0.95, friend_A=1.08, friend_B=0.56 |
| 5 | 1.296 | 0.29 | Spider-Man (2002) | popular | alex=0.62, friend_A=1.33, friend_B=1.07 |
| 6 | 1.271 | 0.27 | Laura (1944) | obscure | alex=0.62, friend_A=1.19, friend_B=0.82 |
| 7 | 1.269 | 0.28 | Eva (2011) | obscure | alex=1.23, friend_A=1.12, friend_B=0.60 |
| 8 | 1.252 | 0.37 | Stevie (2002) | obscure | alex=0.89, friend_A=1.24, friend_B=0.46 |
| 9 | 1.221 | 0.21 | Son of the Bride (Hijo de la novia, El) (2001) | obscure | alex=1.14, friend_A=1.27, friend_B=0.76 |
| 10 | 1.220 | 0.06 | Paterson | obscure | alex=0.67, friend_A=0.67, friend_B=0.76 |

---

## Offline metrics — baseline vs Phase 1 re-ranker

### Individual (averaged across alex + 20 sampled MovieLens users)

| Variant | NDCG@10 | Recall@50 | Coverage | Gini popularity (lower = less popular bias) | Diversity@10 |
|:--------|--------:|----------:|---------:|--------------------------------------------:|-------------:|
| baseline SVD++ | 0.0626 | 0.0745 | 0.0820 | 0.6310 | 0.8165 |
| reranker / balanced | 0.2164 | 0.1908 | 0.0907 | 0.8093 | 0.8722 |
| reranker / niche | 0.0691 | 0.0619 | 0.0888 | 0.4245 | 0.8206 |
| reranker / popular | 0.3250 | 0.3363 | 0.0685 | 0.5137 | 0.8182 |
| reranker / serendipitous | 0.1182 | 0.1237 | 0.0898 | 0.7829 | 0.9279 |

### Group (3 members: alex + 2 random MovieLens users)

| Variant | Strategy | Per-member NDCG@10 | Per-member Recall@50 | Fairness CV | Diversity@10 |
|:--------|:---------|------------------:|---------------------:|------------:|-------------:|
| baseline | average | 0.0511 | 0.0159 | 1.4142 | 0.9042 |
| baseline | least_misery | 0.0511 | 0.0159 | 1.4142 | 0.9042 |
| baseline | most_pleasure | 0.0511 | 0.0159 | 1.4142 | 0.8853 |
| baseline | consensus | 0.0000 | 0.0159 | 0.0000 | 0.7848 |
| baseline | hybrid | 0.0000 | 0.0159 | 0.0000 | 0.7907 |
| reranker | average / balanced | 0.0000 | 0.0000 | 0.0000 | 0.7418 |
| reranker | average / niche | 0.0000 | 0.0000 | 0.0000 | 0.7578 |
| reranker | least_misery / balanced | 0.0000 | 0.0000 | 0.0000 | 0.7468 |
| reranker | least_misery / niche | 0.0000 | 0.0000 | 0.0000 | 0.7678 |
| reranker | most_pleasure / balanced | 0.0839 | 0.1070 | 1.0230 | 0.7576 |
| reranker | most_pleasure / niche | 0.0000 | 0.0000 | 0.0000 | 0.6850 |
| reranker | consensus / balanced | 0.0000 | 0.0000 | 0.0000 | 0.7427 |
| reranker | consensus / niche | 0.0000 | 0.0000 | 0.0000 | 0.7534 |
| reranker | hybrid / balanced | 0.0000 | 0.0000 | 0.0000 | 0.7427 |
| reranker | hybrid / niche | 0.0000 | 0.0000 | 0.0000 | 0.7534 |
| reranker | group_taste_vector / balanced | 0.0000 | 0.0159 | 0.0000 | 0.8774 |
| reranker | group_taste_vector / niche | 0.0000 | 0.0000 | 0.0000 | 0.8324 |

---

## What I want your eye on

1. **Modes:** does `niche` feel niche (would you describe these as 'hidden gems' you don't already know)? Does `popular` lean too mainstream, or about right? Are `serendipitous` picks delightful-surprising or random-noise?
2. **Group strategies:** `group_taste_vector` is the new 6th — finding movies the fused taste vector predicts highly, not just averaging the individual predictions. Does it surface movies that feel like *the group's* picks?
3. **Explanations:** the 'Genre overlap' column shows the top genres in your rated set that match the recommendation. Does this 'why' make sense, or feel post-hoc?
4. **The friends in this demo are random MovieLens users with mismatched taste.** A real friend group would have far more overlap. If something looks weird, ask whether it's the algorithm or the mismatched test friends.
5. **Anything missing?** The point of the Phase 1 check-in is to surface things the offline metrics can't catch. Examples of valid feedback: 'all the niche picks are 70s/80s, I want more recent', 'the explanations should mention specific movies not just genres', 'cold-start fallback fires too aggressively'.
