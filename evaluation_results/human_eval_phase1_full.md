# Phase 1 Human-Eval Check-in — ml-32m (full catalog)

**Purpose:** scan a sample of individual and group recommendations and flag anything that looks off. The offline metrics say the re-ranker improved (table at the bottom), but only you can say if the *vibes* are right.

**Models:** `models/svd_full.pkl` + `models/als_full.pkl` — both trained on **ml-32m** (87k items, 200k users). This replaces the Phase 1 initial run on ml-latest-small.
**User:** `alex_data/ratings_with_tmdb.csv` (220 ratings mapped to MovieLens 32m).
**Synthetic friends:** two random MovieLens users with 155 and 177 ratings respectively.

---

## Individual recommendations — by mode

### Mode: `balanced`

| # | Score | Pop | Title | Genre overlap | Source |
|--:|------:|:---:|:------|:--------------|:------:|
| 1 | 1.645 | obscure | Twin Peaks (1989) | Drama, Mystery | svd+als |
| 2 | 1.478 | popular | Parasite (2019) | Drama, Comedy | svd+als |
| 3 | 1.442 | obscure | Neon Genesis Evangelion: The End of Evangelion (Shin seiki Evangelion Gekijô-ban: Air/Magokoro wo, kimi ni) (1997) | Drama, Action, Sci-Fi | svd+als |
| 4 | 1.400 | obscure | Interrogation (Przesluchanie) (1989) | Drama, Thriller, Crime | svd |
| 5 | 1.379 | obscure | Shoplifters (2018) | Drama | svd+als |
| 6 | 1.358 | obscure | Disappearance of Haruhi Suzumiya, The (Suzumiya Haruhi no shôshitsu) (2010) | Drama, Adventure, Sci-Fi | svd |
| 7 | 1.343 | obscure | A Cowgirl's Story (2017) | Children | svd |
| 8 | 1.335 | obscure | Be With You (2004) | Drama, Fantasy, Romance | svd |
| 9 | 1.331 | obscure | Near Death (1989) | Documentary | svd |
| 10 | 1.284 | obscure | Three Wise Cousins (2016) |  | svd |

### Mode: `niche`

| # | Score | Pop | Title | Genre overlap | Source |
|--:|------:|:---:|:------|:--------------|:------:|
| 1 | 1.164 | obscure | Parole Girl (1933) | Drama, Crime | svd |
| 2 | 1.008 | obscure | Abe & Phil's Last Poker Game (2018) | Drama, Comedy | svd |
| 3 | 1.007 | obscure | A Cowgirl's Story (2017) | Children | svd |
| 4 | 0.994 | obscure | Three Wise Cousins (2016) |  | svd |
| 5 | 0.990 | obscure | Loving the Bad Man (2012) | Drama | svd |
| 6 | 0.974 | obscure | Infinity Train (2016) | Adventure, Animation | svd |
| 7 | 0.956 | obscure | Seeing Red: Stories of American Communists (1983) |  | svd |
| 8 | 0.939 | obscure | Near Death (1989) | Documentary | svd |
| 9 | 0.915 | obscure | Be With You (2004) | Drama, Fantasy, Romance | svd |
| 10 | 0.898 | obscure | I-Be Area (2007) |  | svd |

### Mode: `popular`

| # | Score | Pop | Title | Genre overlap | Source |
|--:|------:|:---:|:------|:--------------|:------:|
| 1 | 2.231 | blockbuster | Pulp Fiction (1994) | Drama, Comedy, Thriller | svd+als |
| 2 | 2.093 | popular | Parasite (2019) | Drama, Comedy | svd+als |
| 3 | 2.076 | blockbuster | Godfather, The (1972) | Drama, Crime | svd+als |
| 4 | 2.065 | obscure | Twin Peaks (1989) | Drama, Mystery | svd+als |
| 5 | 2.040 | blockbuster | Star Wars: Episode IV - A New Hope (1977) | Action, Adventure, Sci-Fi | svd+als |
| 6 | 2.033 | popular | Casablanca (1942) | Drama, Romance | svd+als |
| 7 | 1.970 | obscure | Shoplifters (2018) | Drama | svd+als |
| 8 | 1.921 | obscure | Neon Genesis Evangelion: The End of Evangelion (Shin seiki Evangelion Gekijô-ban: Air/Magokoro wo, kimi ni) (1997) | Drama, Action, Sci-Fi | svd+als |
| 9 | 1.860 | popular | Godfather: Part II, The (1974) | Drama, Crime | svd+als |
| 10 | 1.858 | popular | Blade Runner (1982) | Thriller, Action, Sci-Fi | svd+als |

### Mode: `serendipitous`

| # | Score | Pop | Title | Genre overlap | Source |
|--:|------:|:---:|:------|:--------------|:------:|
| 1 | 1.140 | obscure | Twin Peaks (1989) | Drama, Mystery | svd+als |
| 2 | 0.972 | obscure | Infinity Train (2016) | Adventure, Animation | svd |
| 3 | 0.929 | obscure | A Cowgirl's Story (2017) | Children | svd |
| 4 | 0.908 | obscure | Near Death (1989) | Documentary | svd |
| 5 | 0.906 | obscure | Hungama (2003) | Comedy | svd |
| 6 | 0.899 | obscure | Three Wise Cousins (2016) |  | svd |
| 7 | 0.882 | obscure | Seeing Red: Stories of American Communists (1983) |  | svd |
| 8 | 0.876 | obscure | Neon Genesis Evangelion: The End of Evangelion (Shin seiki Evangelion Gekijô-ban: Air/Magokoro wo, kimi ni) (1997) | Drama, Action, Sci-Fi | svd+als |
| 9 | 0.858 | obscure | I-Be Area (2007) |  | svd |
| 10 | 0.825 | obscure | Interrogation (Przesluchanie) (1989) | Drama, Thriller, Crime | svd |

---

## Group recommendations — 3-friend group, mode=`balanced`

Each row shows the predicted score per member (normalized 0-1) and the fairness coefficient of variation (lower = more uniform agreement).

### Strategy: `average`

| # | Score | Fair | Title | Pop | Per-member scores |
|--:|------:|-----:|:------|:---:|:------------------|
| 1 | 0.824 | 0.18 | Godfather: Part II, The (1974) | popular | alex=0.83, friend_A=0.64, friend_B=1.00 |
| 2 | 0.796 | 0.01 | Princes and Princesses (2000) | obscure | alex=0.81, friend_A=0.78, friend_B=0.80 |
| 3 | 0.786 | 0.08 | Interrogation (Przesluchanie) (1989) | obscure | alex=0.87, friend_A=0.72, friend_B=0.76 |
| 4 | 0.749 | 0.22 | City of God (Cidade de Deus) (2002) | popular | alex=0.77, friend_A=0.54, friend_B=0.94 |
| 5 | 0.734 | 0.10 | Sweeney Todd: The Demon Barber of Fleet Street (1982) | obscure | alex=0.80, friend_A=0.77, friend_B=0.63 |
| 6 | 0.721 | 0.18 | The Collapsed (2011) | obscure | alex=0.69, friend_A=0.89, friend_B=0.58 |
| 7 | 0.720 | 0.16 | Zatoichi's Cane Sword (Zatôichi tekka tabi) (Zatôichi 15) (1967) | obscure | alex=0.66, friend_A=0.62, friend_B=0.88 |
| 8 | 0.715 | 0.22 | Pulp Fiction (1994) | blockbuster | alex=0.93, friend_A=0.57, friend_B=0.64 |
| 9 | 0.703 | 0.19 | Monkey in Winter, A (Un singe en hiver) (1962) | obscure | alex=0.67, friend_A=0.88, friend_B=0.56 |
| 10 | 0.703 | 0.09 | Goyokin (1969) | obscure | alex=0.67, friend_A=0.65, friend_B=0.79 |

### Strategy: `least_misery`

| # | Score | Fair | Title | Pop | Per-member scores |
|--:|------:|-----:|:------|:---:|:------------------|
| 1 | 0.784 | 0.01 | Princes and Princesses (2000) | obscure | alex=0.81, friend_A=0.78, friend_B=0.80 |
| 2 | 0.723 | 0.08 | Interrogation (Przesluchanie) (1989) | obscure | alex=0.87, friend_A=0.72, friend_B=0.76 |
| 3 | 0.650 | 0.09 | Goyokin (1969) | obscure | alex=0.67, friend_A=0.65, friend_B=0.79 |
| 4 | 0.646 | 0.08 | The Fatal Encounter (2014) | obscure | alex=0.77, friend_A=0.68, friend_B=0.65 |
| 5 | 0.643 | 0.18 | Godfather: Part II, The (1974) | popular | alex=0.83, friend_A=0.64, friend_B=1.00 |
| 6 | 0.638 | 0.07 | The Dawns Here are Quiet (1972) | obscure | alex=0.69, friend_A=0.64, friend_B=0.77 |
| 7 | 0.628 | 0.10 | Sweeney Todd: The Demon Barber of Fleet Street (1982) | obscure | alex=0.80, friend_A=0.77, friend_B=0.63 |
| 8 | 0.622 | 0.13 | I'm Staying (2007) | obscure | alex=0.82, friend_A=0.62, friend_B=0.64 |
| 9 | 0.619 | 0.16 | Zatoichi's Cane Sword (Zatôichi tekka tabi) (Zatôichi 15) (1967) | obscure | alex=0.66, friend_A=0.62, friend_B=0.88 |
| 10 | 0.618 | 0.14 | Invisible War, The (2012) | obscure | alex=0.62, friend_A=0.83, friend_B=0.65 |

### Strategy: `most_pleasure`

| # | Score | Fair | Title | Pop | Per-member scores |
|--:|------:|-----:|:------|:---:|:------------------|
| 1 | 1.000 | 0.18 | Godfather: Part II, The (1974) | popular | alex=0.83, friend_A=0.64, friend_B=1.00 |
| 2 | 1.000 | 0.73 | Twin Peaks (1989) | obscure | alex=1.00, friend_A=0.03, friend_B=0.61 |
| 3 | 1.000 | 0.76 | Room, The (2003) | obscure | alex=0.03, friend_A=1.00, friend_B=0.53 |
| 4 | 0.974 | 0.68 | All the President's Men (1976) | niche | alex=0.18, friend_A=0.35, friend_B=0.97 |
| 5 | 0.958 | 0.62 | Neon Genesis Evangelion: The End of Evangelion (Shin seiki Evangelion Gekijô-ban: Air/Magokoro wo, kimi ni) (1997) | obscure | alex=0.96, friend_A=0.49, friend_B=0.15 |
| 6 | 0.954 | 0.54 | Parasite (2019) | popular | alex=0.95, friend_A=0.21, friend_B=0.52 |
| 7 | 0.947 | 0.51 | Shoplifters (2018) | obscure | alex=0.95, friend_A=0.25, friend_B=0.50 |
| 8 | 0.940 | 0.22 | City of God (Cidade de Deus) (2002) | popular | alex=0.77, friend_A=0.54, friend_B=0.94 |
| 9 | 0.938 | 0.34 | Taxi Driver (1976) | popular | alex=0.53, friend_A=0.43, friend_B=0.94 |
| 10 | 0.937 | 0.81 | Bridge on the River Kwai, The (1957) | popular | alex=0.46, friend_A=0.01, friend_B=0.94 |

### Strategy: `consensus`

| # | Score | Fair | Title | Pop | Per-member scores |
|--:|------:|-----:|:------|:---:|:------------------|
| 1 | 0.803 | 0.18 | Godfather: Part II, The (1974) | popular | alex=0.83, friend_A=0.64, friend_B=1.00 |
| 2 | 0.796 | 0.01 | Princes and Princesses (2000) | obscure | alex=0.81, friend_A=0.78, friend_B=0.80 |
| 3 | 0.782 | 0.08 | Interrogation (Przesluchanie) (1989) | obscure | alex=0.87, friend_A=0.72, friend_B=0.76 |
| 4 | 0.728 | 0.10 | Sweeney Todd: The Demon Barber of Fleet Street (1982) | obscure | alex=0.80, friend_A=0.77, friend_B=0.63 |
| 5 | 0.723 | 0.22 | City of God (Cidade de Deus) (2002) | popular | alex=0.77, friend_A=0.54, friend_B=0.94 |
| 6 | 0.707 | 0.16 | Zatoichi's Cane Sword (Zatôichi tekka tabi) (Zatôichi 15) (1967) | obscure | alex=0.66, friend_A=0.62, friend_B=0.88 |
| 7 | 0.704 | 0.18 | The Collapsed (2011) | obscure | alex=0.69, friend_A=0.89, friend_B=0.58 |
| 8 | 0.699 | 0.09 | Goyokin (1969) | obscure | alex=0.67, friend_A=0.65, friend_B=0.79 |
| 9 | 0.697 | 0.08 | The Fatal Encounter (2014) | obscure | alex=0.77, friend_A=0.68, friend_B=0.65 |
| 10 | 0.695 | 0.07 | The Dawns Here are Quiet (1972) | obscure | alex=0.69, friend_A=0.64, friend_B=0.77 |

### Strategy: `hybrid`

| # | Score | Fair | Title | Pop | Per-member scores |
|--:|------:|-----:|:------|:---:|:------------------|
| 1 | 1.188 | 0.01 | Princes and Princesses (2000) | obscure | alex=0.81, friend_A=0.78, friend_B=0.80 |
| 2 | 1.148 | 0.08 | Interrogation (Przesluchanie) (1989) | obscure | alex=0.87, friend_A=0.72, friend_B=0.76 |
| 3 | 1.145 | 0.18 | Godfather: Part II, The (1974) | popular | alex=0.83, friend_A=0.64, friend_B=1.00 |
| 4 | 1.048 | 0.10 | Sweeney Todd: The Demon Barber of Fleet Street (1982) | obscure | alex=0.80, friend_A=0.77, friend_B=0.63 |
| 5 | 1.029 | 0.16 | Zatoichi's Cane Sword (Zatôichi tekka tabi) (Zatôichi 15) (1967) | obscure | alex=0.66, friend_A=0.62, friend_B=0.88 |
| 6 | 1.028 | 0.09 | Goyokin (1969) | obscure | alex=0.67, friend_A=0.65, friend_B=0.79 |
| 7 | 1.023 | 0.08 | The Fatal Encounter (2014) | obscure | alex=0.77, friend_A=0.68, friend_B=0.65 |
| 8 | 1.020 | 0.22 | City of God (Cidade de Deus) (2002) | popular | alex=0.77, friend_A=0.54, friend_B=0.94 |
| 9 | 1.017 | 0.07 | The Dawns Here are Quiet (1972) | obscure | alex=0.69, friend_A=0.64, friend_B=0.77 |
| 10 | 1.011 | 0.14 | Invisible War, The (2012) | obscure | alex=0.62, friend_A=0.83, friend_B=0.65 |

### Strategy: `group_taste_vector`

| # | Score | Fair | Title | Pop | Per-member scores |
|--:|------:|-----:|:------|:---:|:------------------|
| 1 | 1.578 | 0.87 | All the President's Men (1976) | niche | alex=0.05, friend_A=0.57, friend_B=1.60 |
| 2 | 1.474 | 0.38 | Kiki's Delivery Service (Majo no takkyûbin) (1989) | niche | alex=1.24, friend_A=1.33, friend_B=0.47 |
| 3 | 1.411 | 0.55 | Network (1976) | niche | alex=0.32, friend_A=0.47, friend_B=1.12 |
| 4 | 1.355 | 0.44 | North by Northwest (1959) | popular | alex=1.06, friend_A=0.44, friend_B=1.49 |
| 5 | 1.343 | 0.07 | Trials of Darryl Hunt, The (2006) | obscure | alex=0.92, friend_A=1.06, friend_B=1.07 |
| 6 | 1.323 | 0.21 | Godfather: Part II, The (1974) | popular | alex=1.25, friend_A=1.06, friend_B=1.73 |
| 7 | 1.293 | 0.28 | Three Wise Cousins (2016) | obscure | alex=1.03, friend_A=1.21, friend_B=0.58 |
| 8 | 1.268 | 0.21 | The Web (2013) | obscure | alex=0.77, friend_A=1.25, friend_B=0.87 |
| 9 | 1.267 | 0.57 | Rocky (1976) | popular | alex=0.58, friend_A=0.36, friend_B=1.39 |
| 10 | 1.254 | 0.29 | What's Opera, Doc? (1957) | obscure | alex=0.57, friend_A=1.06, friend_B=1.22 |

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
