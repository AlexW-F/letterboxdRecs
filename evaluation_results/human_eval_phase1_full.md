# Phase 1 Human-Eval Check-in — ml-32m (full catalog)

**Purpose:** scan a sample of individual and group recommendations and flag anything that looks off. The offline metrics say the re-ranker improved (table at the bottom), but only you can say if the *vibes* are right.

**Models:** `models/svd_full.pkl` + `models/als_full.pkl` — both trained on **ml-32m** (87k items, 200k users). This replaces the Phase 1 initial run on ml-latest-small.
**User:** `alex_data/ratings_with_tmdb.csv` (220 ratings mapped to MovieLens 32m, plus 258 watched-but-unrated films also excluded from recs).
**Synthetic friends:** two random MovieLens users with 155 and 177 ratings respectively.

---

## Individual recommendations — by mode

### Mode: `balanced`

| # | Score | Pop | Title | Genre overlap | Source |
|--:|------:|:---:|:------|:--------------|:------:|
| 1 | 1.480 | obscure | Twin Peaks (1989) | Drama, Mystery | svd+als |
| 2 | 1.249 | obscure | Son of the White Mare (1981) | Adventure, Fantasy, Animation | svd |
| 3 | 1.235 | popular | Parasite (2019) | Drama, Comedy | svd+als |
| 4 | 1.225 | obscure | Disappearance of Haruhi Suzumiya, The (Suzumiya Haruhi no shôshitsu) (2010) | Drama, Adventure, Sci-Fi | svd |
| 5 | 1.202 | obscure | Shoplifters (2018) | Drama | svd+als |
| 6 | 1.180 | obscure | Sherman's March (1985) | Documentary | svd |
| 7 | 1.133 | obscure | Heartstone (2016) | Drama, Romance | svd |
| 8 | 1.129 | obscure | Neon Genesis Evangelion: The End of Evangelion (Shin seiki Evangelion Gekijô-ban: Air/Magokoro wo, kimi ni) (1997) | Drama, Action, Sci-Fi | svd+als |
| 9 | 1.121 | obscure | 7 Boxes (2012) | Thriller | svd |
| 10 | 1.063 | obscure | Room 8 (2013) |  | svd |

### Mode: `niche`

| # | Score | Pop | Title | Genre overlap | Source |
|--:|------:|:---:|:------|:--------------|:------:|
| 1 | 0.896 | obscure | Interrogation (Przesluchanie) (1989) | Drama, Thriller, Crime | svd |
| 2 | 0.791 | obscure | Absolute Giganten (1999) | Drama, Comedy, Action | svd |
| 3 | 0.743 | obscure | Mushishi: The Shadow That Devours the Sun (2014) | Adventure, Fantasy, Animation | svd |
| 4 | 0.706 | obscure | Sweeney Todd: The Demon Barber of Fleet Street (1982) | Drama, Thriller, Horror | svd |
| 5 | 0.685 | obscure | Twin Peaks (1989) | Drama, Mystery | svd+als |
| 6 | 0.666 | obscure | Everything's Gonna Be Great (1998) | Drama, Comedy, Adventure | svd |
| 7 | 0.658 | obscure | Heartstone (2016) | Drama, Romance | svd |
| 8 | 0.656 | obscure | Cuba and the Cameraman (2017) | Documentary | svd |
| 9 | 0.645 | obscure | Le Plaisir (1952) | Drama, Comedy | svd |
| 10 | 0.633 | obscure | Li'l Quinquin (2014) | Comedy, Crime, Mystery | svd |

### Mode: `popular`

| # | Score | Pop | Title | Genre overlap | Source |
|--:|------:|:---:|:------|:--------------|:------:|
| 1 | 2.979 | blockbuster | Godfather, The (1972) | Drama, Crime | svd+als |
| 2 | 2.800 | popular | Casablanca (1942) | Drama, Romance | svd+als |
| 3 | 2.786 | popular | Parasite (2019) | Drama, Comedy | svd+als |
| 4 | 2.649 | popular | Godfather: Part II, The (1974) | Drama, Crime | svd+als |
| 5 | 2.638 | obscure | Twin Peaks (1989) | Drama, Mystery | svd+als |
| 6 | 2.553 | popular | One Flew Over the Cuckoo's Nest (1975) | Drama | svd+als |
| 7 | 2.534 | niche | Kiki's Delivery Service (Majo no takkyûbin) (1989) | Drama, Adventure, Fantasy | svd+als |
| 8 | 2.530 | popular | Vertigo (1958) | Drama, Thriller, Romance | svd+als |
| 9 | 2.495 | popular | Kill Bill: Vol. 1 (2003) | Thriller, Action, Crime | svd+als |
| 10 | 2.482 | obscure | Neon Genesis Evangelion: The End of Evangelion (Shin seiki Evangelion Gekijô-ban: Air/Magokoro wo, kimi ni) (1997) | Drama, Action, Sci-Fi | svd+als |

### Mode: `serendipitous`

| # | Score | Pop | Title | Genre overlap | Source |
|--:|------:|:---:|:------|:--------------|:------:|
| 1 | 1.096 | obscure | Twin Peaks (1989) | Drama, Mystery | svd+als |
| 2 | 0.874 | obscure | Son of the White Mare (1981) | Adventure, Fantasy, Animation | svd |
| 3 | 0.836 | obscure | Love in the Time of Hysteria (Sólo con tu pareja) (1991) | Comedy, Romance | svd |
| 4 | 0.793 | obscure | Interrogation (Przesluchanie) (1989) | Drama, Thriller, Crime | svd |
| 5 | 0.787 | obscure | Triumph Over Violence (1965) | Documentary, War | svd |
| 6 | 0.763 | obscure | Everything's Gonna Be Great (1998) | Drama, Comedy, Adventure | svd |
| 7 | 0.723 | obscure | Space Odyssey: Voyage to the Planets (2004) | Drama, Sci-Fi, Documentary | svd |
| 8 | 0.710 | obscure | Wolf Hall (2015) |  | svd |
| 9 | 0.687 | obscure | Room 8 (2013) |  | svd |
| 10 | 0.664 | obscure | Bunraku (2010) | Drama, Action, Fantasy | svd |

---

## Group recommendations — 3-friend group, mode=`balanced`

Each row shows the predicted score per member (normalized 0-1) and the fairness coefficient of variation (lower = more uniform agreement).

### Strategy: `average`

| # | Score | Fair | Title | Pop | Per-member scores |
|--:|------:|-----:|:------|:---:|:------------------|
| 1 | 0.789 | 0.10 | Invisible War, The (2012) | obscure | alex=0.69, friend_A=0.88, friend_B=0.79 |
| 2 | 0.774 | 0.10 | The Dawns Here are Quiet (1972) | obscure | alex=0.74, friend_A=0.70, friend_B=0.88 |
| 3 | 0.773 | 0.20 | Godfather: Part II, The (1974) | popular | alex=0.77, friend_A=0.58, friend_B=0.96 |
| 4 | 0.760 | 0.16 | Adventure Time: Islands (2017) | obscure | alex=0.75, friend_A=0.62, friend_B=0.91 |
| 5 | 0.759 | 0.06 | Everything's Gonna Be Great (1998) | obscure | alex=0.82, friend_A=0.71, friend_B=0.74 |
| 6 | 0.758 | 0.11 | Heartstone (2016) | obscure | alex=0.86, friend_A=0.75, friend_B=0.66 |
| 7 | 0.757 | 0.04 | Saint Frances (2020) | obscure | alex=0.76, friend_A=0.80, friend_B=0.72 |
| 8 | 0.753 | 0.15 | The Amazing Screw-On Head (2006) | obscure | alex=0.62, friend_A=0.76, friend_B=0.89 |
| 9 | 0.725 | 0.18 | 5 Fingers (1952) | obscure | alex=0.63, friend_A=0.64, friend_B=0.91 |
| 10 | 0.724 | 0.11 | Strike (Stachka) (1925) | obscure | alex=0.73, friend_A=0.82, friend_B=0.63 |

### Strategy: `least_misery`

| # | Score | Fair | Title | Pop | Per-member scores |
|--:|------:|-----:|:------|:---:|:------------------|
| 1 | 0.715 | 0.04 | Saint Frances (2020) | obscure | alex=0.76, friend_A=0.80, friend_B=0.72 |
| 2 | 0.712 | 0.06 | Everything's Gonna Be Great (1998) | obscure | alex=0.82, friend_A=0.71, friend_B=0.74 |
| 3 | 0.699 | 0.10 | The Dawns Here are Quiet (1972) | obscure | alex=0.74, friend_A=0.70, friend_B=0.88 |
| 4 | 0.690 | 0.10 | Invisible War, The (2012) | obscure | alex=0.69, friend_A=0.88, friend_B=0.79 |
| 5 | 0.678 | 0.05 | War of the Buttons (1994) | obscure | alex=0.76, friend_A=0.68, friend_B=0.72 |
| 6 | 0.664 | 0.11 | Heartstone (2016) | obscure | alex=0.86, friend_A=0.75, friend_B=0.66 |
| 7 | 0.635 | 0.09 | Innocent Voices (Voces inocentes) (2004) | obscure | alex=0.64, friend_A=0.80, friend_B=0.73 |
| 8 | 0.632 | 0.09 | Chris Rock: Bigger & Blacker (1999) | obscure | alex=0.63, friend_A=0.79, friend_B=0.72 |
| 9 | 0.631 | 0.08 | Beastie Boys: Sabotage (1994) | obscure | alex=0.63, friend_A=0.76, friend_B=0.72 |
| 10 | 0.631 | 0.18 | 5 Fingers (1952) | obscure | alex=0.63, friend_A=0.64, friend_B=0.91 |

### Strategy: `most_pleasure`

| # | Score | Fair | Title | Pop | Per-member scores |
|--:|------:|-----:|:------|:---:|:------------------|
| 1 | 1.000 | 0.54 | Room, The (2003) | obscure | alex=0.19, friend_A=1.00, friend_B=0.64 |
| 2 | 1.000 | 0.73 | All the President's Men (1976) | niche | alex=0.15, friend_A=0.34, friend_B=1.00 |
| 3 | 1.000 | 0.66 | Twin Peaks (1989) | obscure | alex=1.00, friend_A=0.07, friend_B=0.70 |
| 4 | 0.962 | 0.20 | Godfather: Part II, The (1974) | popular | alex=0.77, friend_A=0.58, friend_B=0.96 |
| 5 | 0.949 | 0.45 | Shoplifters (2018) | obscure | alex=0.95, friend_A=0.28, friend_B=0.61 |
| 6 | 0.938 | 0.83 | Bridge on the River Kwai, The (1957) | popular | alex=0.45, friend_A=0.00, friend_B=0.94 |
| 7 | 0.937 | 0.25 | City of God (Cidade de Deus) (2002) | popular | alex=0.71, friend_A=0.50, friend_B=0.94 |
| 8 | 0.934 | 0.40 | Taxi Driver (1976) | popular | alex=0.48, friend_A=0.38, friend_B=0.93 |
| 9 | 0.922 | 0.41 | North by Northwest (1959) | popular | alex=0.68, friend_A=0.30, friend_B=0.92 |
| 10 | 0.920 | 0.45 | Neon Genesis Evangelion: The End of Evangelion (Shin seiki Evangelion Gekijô-ban: Air/Magokoro wo, kimi ni) (1997) | obscure | alex=0.92, friend_A=0.50, friend_B=0.30 |

### Strategy: `consensus`

| # | Score | Fair | Title | Pop | Per-member scores |
|--:|------:|-----:|:------|:---:|:------------------|
| 1 | 0.782 | 0.10 | Invisible War, The (2012) | obscure | alex=0.69, friend_A=0.88, friend_B=0.79 |
| 2 | 0.768 | 0.10 | The Dawns Here are Quiet (1972) | obscure | alex=0.74, friend_A=0.70, friend_B=0.88 |
| 3 | 0.757 | 0.06 | Everything's Gonna Be Great (1998) | obscure | alex=0.82, friend_A=0.71, friend_B=0.74 |
| 4 | 0.756 | 0.04 | Saint Frances (2020) | obscure | alex=0.76, friend_A=0.80, friend_B=0.72 |
| 5 | 0.751 | 0.11 | Heartstone (2016) | obscure | alex=0.86, friend_A=0.75, friend_B=0.66 |
| 6 | 0.749 | 0.20 | Godfather: Part II, The (1974) | popular | alex=0.77, friend_A=0.58, friend_B=0.96 |
| 7 | 0.745 | 0.16 | Adventure Time: Islands (2017) | obscure | alex=0.75, friend_A=0.62, friend_B=0.91 |
| 8 | 0.741 | 0.15 | The Amazing Screw-On Head (2006) | obscure | alex=0.62, friend_A=0.76, friend_B=0.89 |
| 9 | 0.718 | 0.05 | War of the Buttons (1994) | obscure | alex=0.76, friend_A=0.68, friend_B=0.72 |
| 10 | 0.718 | 0.11 | Strike (Stachka) (1925) | obscure | alex=0.73, friend_A=0.82, friend_B=0.63 |

### Strategy: `hybrid`

| # | Score | Fair | Title | Pop | Per-member scores |
|--:|------:|-----:|:------|:---:|:------------------|
| 1 | 1.134 | 0.10 | Invisible War, The (2012) | obscure | alex=0.69, friend_A=0.88, friend_B=0.79 |
| 2 | 1.123 | 0.10 | The Dawns Here are Quiet (1972) | obscure | alex=0.74, friend_A=0.70, friend_B=0.88 |
| 3 | 1.115 | 0.06 | Everything's Gonna Be Great (1998) | obscure | alex=0.82, friend_A=0.71, friend_B=0.74 |
| 4 | 1.115 | 0.04 | Saint Frances (2020) | obscure | alex=0.76, friend_A=0.80, friend_B=0.72 |
| 5 | 1.090 | 0.11 | Heartstone (2016) | obscure | alex=0.86, friend_A=0.75, friend_B=0.66 |
| 6 | 1.067 | 0.16 | Adventure Time: Islands (2017) | obscure | alex=0.75, friend_A=0.62, friend_B=0.91 |
| 7 | 1.064 | 0.20 | Godfather: Part II, The (1974) | popular | alex=0.77, friend_A=0.58, friend_B=0.96 |
| 8 | 1.062 | 0.15 | The Amazing Screw-On Head (2006) | obscure | alex=0.62, friend_A=0.76, friend_B=0.89 |
| 9 | 1.058 | 0.05 | War of the Buttons (1994) | obscure | alex=0.76, friend_A=0.68, friend_B=0.72 |
| 10 | 1.041 | 0.18 | 5 Fingers (1952) | obscure | alex=0.63, friend_A=0.64, friend_B=0.91 |

### Strategy: `group_taste_vector`

| # | Score | Fair | Title | Pop | Per-member scores |
|--:|------:|-----:|:------|:---:|:------------------|
| 1 | 1.296 | 0.71 | Network (1976) | niche | alex=0.15, friend_A=0.22, friend_B=0.74 |
| 2 | 1.224 | 0.68 | Kiki's Delivery Service (Majo no takkyûbin) (1989) | niche | alex=1.04, friend_A=1.20, friend_B=0.03 |
| 3 | 1.157 | 0.46 | The Adventures of Sherlock Holmes and Doctor Watson: King of Blackmailers (1980) | obscure | alex=0.79, friend_A=1.26, friend_B=0.35 |
| 4 | 1.156 | 1.16 | All the President's Men (1976) | niche | alex=-0.13, friend_A=0.36, friend_B=1.25 |
| 5 | 1.149 | 0.45 | Connections (1978) | obscure | alex=0.60, friend_A=0.46, friend_B=1.26 |
| 6 | 1.116 | 0.27 | What's Opera, Doc? (1957) | obscure | alex=0.50, friend_A=0.89, friend_B=0.98 |
| 7 | 1.107 | 0.05 | The Dawns Here are Quiet (1972) | obscure | alex=0.89, friend_A=0.91, friend_B=1.00 |
| 8 | 1.093 | 0.31 | I Am So Proud of You (2008) | obscure | alex=0.47, friend_A=1.08, friend_B=0.86 |
| 9 | 1.071 | 0.31 | Samurai III: Duel on Ganryu Island (a.k.a. Bushido) (Miyamoto Musashi kanketsuhen: kettô Ganryûjima) (1956) | obscure | alex=0.95, friend_A=0.42, friend_B=0.76 |
| 10 | 1.064 | 0.27 | Beastie Boys: Sabotage (1994) | obscure | alex=0.71, friend_A=1.20, friend_B=0.71 |

---

## Offline metrics — baseline vs Phase 1 re-ranker

### Individual (averaged across alex + 20 sampled MovieLens users)

| Variant | NDCG@10 | Recall@50 | Coverage | Gini popularity (lower = less popular bias) | Diversity@10 |
|:--------|--------:|----------:|---------:|--------------------------------------------:|-------------:|
| baseline SVD++ | 0.0000 | 0.0243 | 0.0100 | 0.7378 | 0.7715 |
| reranker / balanced | 0.1256 | 0.0909 | 0.0109 | 0.8905 | 0.9001 |
| reranker / niche | 0.0000 | 0.0109 | 0.0115 | 0.9339 | 0.9092 |
| reranker / popular | 0.2177 | 0.1950 | 0.0089 | 0.6399 | 0.8043 |
| reranker / serendipitous | 0.0562 | 0.0575 | 0.0110 | 0.9170 | 0.9819 |

### Group (3 members: alex + 2 random MovieLens users)

| Variant | Strategy | Per-member NDCG@10 | Per-member Recall@50 | Fairness CV | Diversity@10 |
|:--------|:---------|------------------:|---------------------:|------------:|-------------:|
| baseline | average | 0.0000 | 0.0000 | 0.0000 | 0.7696 |
| baseline | least_misery | 0.0000 | 0.0000 | 0.0000 | 0.7956 |
| baseline | most_pleasure | 0.0000 | 0.0000 | 0.0000 | 0.5259 |
| baseline | consensus | 0.0000 | 0.0000 | 0.0000 | 0.7207 |
| baseline | hybrid | 0.0000 | 0.0000 | 0.0000 | 0.6607 |
| reranker | average / balanced | 0.0000 | 0.0728 | 0.0000 | 0.6863 |
| reranker | average / niche | 0.0000 | 0.0000 | 0.0000 | 0.6665 |
| reranker | least_misery / balanced | 0.0000 | 0.0083 | 0.0000 | 0.5848 |
| reranker | least_misery / niche | 0.0000 | 0.0000 | 0.0000 | 0.6404 |
| reranker | most_pleasure / balanced | 0.1380 | 0.1207 | 0.2166 | 0.6084 |
| reranker | most_pleasure / niche | 0.0000 | 0.0000 | 0.0000 | 0.6142 |
| reranker | consensus / balanced | 0.0000 | 0.0312 | 0.0000 | 0.6627 |
| reranker | consensus / niche | 0.0000 | 0.0000 | 0.0000 | 0.6665 |
| reranker | hybrid / balanced | 0.0000 | 0.0312 | 0.0000 | 0.6472 |
| reranker | hybrid / niche | 0.0000 | 0.0000 | 0.0000 | 0.6665 |
| reranker | group_taste_vector / balanced | 0.0000 | 0.0000 | 0.0000 | 0.9334 |
| reranker | group_taste_vector / niche | 0.0000 | 0.0000 | 0.0000 | 0.9667 |

---

## What I want your eye on

1. **Modes:** does `niche` feel niche (would you describe these as 'hidden gems' you don't already know)? Does `popular` lean too mainstream, or about right? Are `serendipitous` picks delightful-surprising or random-noise?
2. **Group strategies:** `group_taste_vector` is the new 6th — finding movies the fused taste vector predicts highly, not just averaging the individual predictions. Does it surface movies that feel like *the group's* picks?
3. **Explanations:** the 'Genre overlap' column shows the top genres in your rated set that match the recommendation. Does this 'why' make sense, or feel post-hoc?
4. **The friends in this demo are random MovieLens users with mismatched taste.** A real friend group would have far more overlap. If something looks weird, ask whether it's the algorithm or the mismatched test friends.
5. **Anything missing?** The point of the Phase 1 check-in is to surface things the offline metrics can't catch. Examples of valid feedback: 'all the niche picks are 70s/80s, I want more recent', 'the explanations should mention specific movies not just genres', 'cold-start fallback fires too aggressively'.
