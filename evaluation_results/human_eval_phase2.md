# Phase 2 Human-Eval Check-in — content features added

**Purpose:** scan a sample of individual and group recommendations after adding the Phase 2 content scorer. The new term blends TF-IDF cosine similarity (over ml-32m's 2M user-generated tags + genres) into the re-rank score. The offline metrics tell us if it improved (table at the bottom), but only you can say if it makes the picks feel more *yours*.

**Models:** `models/svd_full.pkl` + `models/als_full.pkl` + `data/content_features.npz` — all built on ml-32m (87k items, 200k users, 20k-vocab TF-IDF over tags+genres).
**User:** `alex_data/ratings_with_tmdb.csv` (220 ratings mapped to MovieLens 32m, plus 258 watched-but-unrated films also excluded from recs).
**Synthetic friends:** two random MovieLens users with 155 and 177 ratings respectively.

---

## Individual recommendations — by mode

### Mode: `balanced`

| # | Score | Pop | Title | Genre overlap | Source |
|--:|------:|:---:|:------|:--------------|:------:|
| 1 | 1.833 | popular | Casablanca (1942) | Drama, Romance | svd+als |
| 2 | 1.654 | blockbuster | Godfather, The (1972) | Drama, Crime | svd+als |
| 3 | 1.647 | niche | Kiki's Delivery Service (Majo no takkyûbin) (1989) | Drama, Adventure, Fantasy | svd+als |
| 4 | 1.602 | obscure | Twin Peaks (1989) | Drama, Mystery | svd+als |
| 5 | 1.550 | obscure | Sherman's March (1985) | Documentary | svd |
| 6 | 1.503 | obscure | Shoplifters (2018) | Drama | svd+als |
| 7 | 1.494 | obscure | Neon Genesis Evangelion: The End of Evangelion (Shin seiki Evangelion Gekijô-ban: Air/Magokoro wo, kimi ni) (1997) | Drama, Action, Sci-Fi | svd+als |
| 8 | 1.482 | niche | Grave of the Fireflies (Hotaru no haka) (1988) | Drama, Animation, War | svd+als |
| 9 | 1.460 | obscure | Disappearance of Haruhi Suzumiya, The (Suzumiya Haruhi no shôshitsu) (2010) | Drama, Adventure, Sci-Fi | svd |
| 10 | 1.452 | obscure | Mimino (1977) | Comedy | svd |

### Mode: `niche`

| # | Score | Pop | Title | Genre overlap | Source |
|--:|------:|:---:|:------|:--------------|:------:|
| 1 | 1.556 | obscure | Cuba and the Cameraman (2017) | Documentary | svd |
| 2 | 1.310 | obscure | Interrogation (Przesluchanie) (1989) | Drama, Thriller, Crime | svd |
| 3 | 1.275 | obscure | Mushishi: The Shadow That Devours the Sun (2014) | Adventure, Fantasy, Animation | svd |
| 4 | 1.110 | obscure | Edvard Munch (1974) | Drama | svd |
| 5 | 1.064 | obscure | Absolute Giganten (1999) | Drama, Comedy, Action | svd |
| 6 | 1.061 | obscure | Wings of Hope (Julianes Sturz in den Dschungel) (2000) | Adventure, Documentary | svd |
| 7 | 1.050 | obscure | Phenix City Story, The (1955) | Drama, Crime, Film-Noir | svd |
| 8 | 1.026 | obscure | Legend of the Galactic Heroes: My Conquest Is the Sea of Stars (1988) | Sci-Fi, Animation | svd |
| 9 | 1.017 | obscure | Heartstone (2016) | Drama, Romance | svd |
| 10 | 1.006 | obscure | Samurai III: Duel on Ganryu Island (a.k.a. Bushido) (Miyamoto Musashi kanketsuhen: kettô Ganryûjima) (1956) | Drama, Action, Adventure | svd |

### Mode: `popular`

| # | Score | Pop | Title | Genre overlap | Source |
|--:|------:|:---:|:------|:--------------|:------:|
| 1 | 3.249 | blockbuster | Godfather, The (1972) | Drama, Crime | svd+als |
| 2 | 3.100 | popular | Casablanca (1942) | Drama, Romance | svd+als |
| 3 | 2.911 | popular | Parasite (2019) | Drama, Comedy | svd+als |
| 4 | 2.877 | popular | Godfather: Part II, The (1974) | Drama, Crime | svd+als |
| 5 | 2.811 | popular | One Flew Over the Cuckoo's Nest (1975) | Drama | svd+als |
| 6 | 2.798 | niche | Kiki's Delivery Service (Majo no takkyûbin) (1989) | Drama, Adventure, Fantasy | svd+als |
| 7 | 2.791 | popular | Vertigo (1958) | Drama, Thriller, Romance | svd+als |
| 8 | 2.740 | obscure | Twin Peaks (1989) | Drama, Mystery | svd+als |
| 9 | 2.672 | popular | City of God (Cidade de Deus) (2002) | Drama, Thriller, Action | svd+als |
| 10 | 2.668 | popular | North by Northwest (1959) | Thriller, Action, Adventure | svd+als |

### Mode: `serendipitous`

| # | Score | Pop | Title | Genre overlap | Source |
|--:|------:|:---:|:------|:--------------|:------:|
| 1 | 1.514 | popular | Casablanca (1942) | Drama, Romance | svd+als |
| 2 | 1.467 | obscure | Cuba and the Cameraman (2017) | Documentary | svd |
| 3 | 1.285 | obscure | Mushishi: The Shadow That Devours the Sun (2014) | Adventure, Fantasy, Animation | svd |
| 4 | 1.115 | obscure | Interrogation (Przesluchanie) (1989) | Drama, Thriller, Crime | svd |
| 5 | 1.106 | niche | Grave of the Fireflies (Hotaru no haka) (1988) | Drama, Animation, War | svd+als |
| 6 | 1.071 | obscure | Mimino (1977) | Comedy | svd |
| 7 | 1.035 | obscure | Twin Peaks (1989) | Drama, Mystery | svd+als |
| 8 | 1.021 | obscure | Neon Genesis Evangelion: The End of Evangelion (Shin seiki Evangelion Gekijô-ban: Air/Magokoro wo, kimi ni) (1997) | Drama, Action, Sci-Fi | svd+als |
| 9 | 1.016 | obscure | The Adventures of Sherlock Holmes and Doctor Watson |  | svd |
| 10 | 1.000 | obscure | Wolf Hall (2015) |  | svd |

---

## Group recommendations — 3-friend group, mode=`balanced`

Each row shows the predicted score per member (normalized 0-1) and the fairness coefficient of variation (lower = more uniform agreement).

### Strategy: `average`

| # | Score | Fair | Title | Pop | Per-member scores |
|--:|------:|-----:|:------|:---:|:------------------|
| 1 | 0.956 | 0.00 | Godfather, The (1972) | blockbuster | alex=0.96 |
| 2 | 0.946 | 0.00 | Clockwork Orange, A (1971) | popular | friend_A=0.95 |
| 3 | 0.831 | 0.08 | Godfather: Part II, The (1974) | popular | alex=0.85, friend_A=0.74, friend_B=0.91 |
| 4 | 0.816 | 0.02 | City of God (Cidade de Deus) (2002) | popular | alex=0.80, friend_B=0.84 |
| 5 | 0.799 | 0.24 | Kill Bill: Vol. 2 (2004) | popular | alex=0.61, friend_A=0.99 |
| 6 | 0.787 | 0.00 | Goodfellas (1990) | popular | friend_B=0.79 |
| 7 | 0.786 | 0.27 | Kill Bill: Vol. 1 (2003) | popular | alex=0.57, friend_A=1.00 |
| 8 | 0.770 | 0.00 | Lord of the Rings: The Return of the King, The (2003) | blockbuster | friend_B=0.77 |
| 9 | 0.766 | 0.18 | Grave of the Fireflies (Hotaru no haka) (1988) | niche | alex=0.90, friend_A=0.63 |
| 10 | 0.763 | 0.28 | Donnie Darko (2001) | popular | alex=0.55, friend_A=0.98 |

### Strategy: `least_misery`

| # | Score | Fair | Title | Pop | Per-member scores |
|--:|------:|-----:|:------|:---:|:------------------|
| 1 | 0.741 | 0.08 | Godfather: Part II, The (1974) | popular | alex=0.85, friend_A=0.74, friend_B=0.91 |
| 2 | 0.589 | 0.06 | 5 Fingers (1952) | obscure | alex=0.61, friend_A=0.59, friend_B=0.68 |
| 3 | 0.576 | 0.14 | Invisible War, The (2012) | obscure | alex=0.68, friend_A=0.81, friend_B=0.58 |
| 4 | 0.552 | 0.05 | Adventure Time: Islands (2017) | obscure | alex=0.55, friend_A=0.63, friend_B=0.61 |
| 5 | 0.519 | 0.17 | Chinatown (1974) | popular | alex=0.75, friend_A=0.52, friend_B=0.79 |
| 6 | 0.511 | 0.06 | Attack On Titan (2013) | obscure | alex=0.51, friend_A=0.59, friend_B=0.57 |
| 7 | 0.509 | 0.14 | Innocent Voices (Voces inocentes) (2004) | obscure | alex=0.59, friend_A=0.72, friend_B=0.51 |
| 8 | 0.505 | 0.20 | Léon: The Professional (a.k.a. The Professional) (Léon) (1994) | popular | alex=0.50, friend_A=0.82, friend_B=0.62 |
| 9 | 0.502 | 0.10 | The War (2007) | obscure | alex=0.64, friend_A=0.50, friend_B=0.58 |
| 10 | 0.497 | 0.22 | Ballad of a Soldier (Ballada o soldate) (1959) | obscure | alex=0.78, friend_A=0.50, friend_B=0.51 |

### Strategy: `most_pleasure`

| # | Score | Fair | Title | Pop | Per-member scores |
|--:|------:|-----:|:------|:---:|:------------------|
| 1 | 1.000 | 0.27 | Kill Bill: Vol. 1 (2003) | popular | alex=0.57, friend_A=1.00 |
| 2 | 1.000 | 0.47 | Bridge on the River Kwai, The (1957) | popular | alex=0.66, friend_A=0.26, friend_B=1.00 |
| 3 | 1.000 | 0.31 | Casablanca (1942) | popular | alex=1.00, friend_A=0.52 |
| 4 | 0.991 | 0.24 | Kill Bill: Vol. 2 (2004) | popular | alex=0.61, friend_A=0.99 |
| 5 | 0.978 | 0.28 | Donnie Darko (2001) | popular | alex=0.55, friend_A=0.98 |
| 6 | 0.978 | 0.65 | Room, The (2003) | obscure | alex=0.16, friend_A=0.98, friend_B=0.43 |
| 7 | 0.974 | 0.51 | Children of Men (2006) | popular | alex=0.47, friend_A=0.97, friend_B=0.28 |
| 8 | 0.956 | 0.00 | Godfather, The (1972) | blockbuster | alex=0.96 |
| 9 | 0.950 | 0.49 | Kiki's Delivery Service (Majo no takkyûbin) (1989) | niche | alex=0.95, friend_A=0.74, friend_B=0.21 |
| 10 | 0.946 | 0.00 | Clockwork Orange, A (1971) | popular | friend_A=0.95 |

### Strategy: `consensus`

| # | Score | Fair | Title | Pop | Per-member scores |
|--:|------:|-----:|:------|:---:|:------------------|
| 1 | 0.956 | 0.00 | Godfather, The (1972) | blockbuster | alex=0.96 |
| 2 | 0.946 | 0.00 | Clockwork Orange, A (1971) | popular | friend_A=0.95 |
| 3 | 0.826 | 0.08 | Godfather: Part II, The (1974) | popular | alex=0.85, friend_A=0.74, friend_B=0.91 |
| 4 | 0.816 | 0.02 | City of God (Cidade de Deus) (2002) | popular | alex=0.80, friend_B=0.84 |
| 5 | 0.787 | 0.00 | Goodfellas (1990) | popular | friend_B=0.79 |
| 6 | 0.770 | 0.00 | Lord of the Rings: The Return of the King, The (2003) | blockbuster | friend_B=0.77 |
| 7 | 0.762 | 0.24 | Kill Bill: Vol. 2 (2004) | popular | alex=0.61, friend_A=0.99 |
| 8 | 0.760 | 0.00 | Silence of the Lambs, The (1991) | blockbuster | friend_A=0.76 |
| 9 | 0.758 | 0.00 | Dr. Strangelove or: How I Learned to Stop Worrying and Love the Bomb (1964) | popular | friend_B=0.76 |
| 10 | 0.747 | 0.18 | Grave of the Fireflies (Hotaru no haka) (1988) | niche | alex=0.90, friend_A=0.63 |

### Strategy: `hybrid`

| # | Score | Fair | Title | Pop | Per-member scores |
|--:|------:|-----:|:------|:---:|:------------------|
| 1 | 1.433 | 0.00 | Godfather, The (1972) | blockbuster | alex=0.96 |
| 2 | 1.419 | 0.00 | Clockwork Orange, A (1971) | popular | friend_A=0.95 |
| 3 | 1.214 | 0.02 | City of God (Cidade de Deus) (2002) | popular | alex=0.80, friend_B=0.84 |
| 4 | 1.201 | 0.08 | Godfather: Part II, The (1974) | popular | alex=0.85, friend_A=0.74, friend_B=0.91 |
| 5 | 1.181 | 0.00 | Goodfellas (1990) | popular | friend_B=0.79 |
| 6 | 1.155 | 0.00 | Lord of the Rings: The Return of the King, The (2003) | blockbuster | friend_B=0.77 |
| 7 | 1.140 | 0.00 | Silence of the Lambs, The (1991) | blockbuster | friend_A=0.76 |
| 8 | 1.136 | 0.00 | Dr. Strangelove or: How I Learned to Stop Worrying and Love the Bomb (1964) | popular | friend_B=0.76 |
| 9 | 1.102 | 0.24 | Kill Bill: Vol. 2 (2004) | popular | alex=0.61, friend_A=0.99 |
| 10 | 1.092 | 0.01 | 2001: A Space Odyssey (1968) | popular | friend_A=0.74, friend_B=0.72 |

### Strategy: `group_taste_vector`

| # | Score | Fair | Title | Pop | Per-member scores |
|--:|------:|-----:|:------|:---:|:------------------|
| 1 | 1.711 | 0.24 | Network (1976) | niche | alex=0.64, friend_A=0.78, friend_B=1.12 |
| 2 | 1.513 | 0.27 | North by Northwest (1959) | popular | alex=1.34, friend_A=0.72, friend_B=1.43 |
| 3 | 1.471 | 0.14 | Godfather: Part II, The (1974) | popular | alex=1.46, friend_A=1.38, friend_B=1.87 |
| 4 | 1.457 | 0.68 | All the President's Men (1976) | niche | alex=0.21, friend_A=0.68, friend_B=1.55 |
| 5 | 1.433 | 0.61 | Bridge on the River Kwai, The (1957) | popular | alex=0.83, friend_A=0.37, friend_B=1.88 |
| 6 | 1.374 | 0.38 | Rocky (1976) | popular | alex=0.77, friend_A=0.53, friend_B=1.32 |
| 7 | 1.361 | 0.23 | Chinatown (1974) | popular | alex=1.20, friend_A=0.81, friend_B=1.46 |
| 8 | 1.356 | 0.68 | Kiki's Delivery Service (Majo no takkyûbin) (1989) | niche | alex=1.59, friend_A=1.44, friend_B=0.05 |
| 9 | 1.336 | 0.33 | Vertigo (1958) | popular | alex=1.47, friend_A=0.61, friend_B=1.38 |
| 10 | 1.286 | 0.70 | Ben-Hur (1959) | niche | alex=0.13, friend_A=0.67, friend_B=1.36 |

---

## Offline metrics — baseline vs Phase 1 re-ranker

### Individual (averaged across alex + 20 sampled MovieLens users)

| Variant | NDCG@10 | Recall@50 | Coverage | Gini popularity (lower = less popular bias) | Diversity@10 |
|:--------|--------:|----------:|---------:|--------------------------------------------:|-------------:|
| baseline SVD++ | 0.0000 | 0.0243 | 0.0100 | 0.7378 | 0.7715 |
| reranker / balanced | 0.2023 | 0.1397 | 0.0102 | 0.8021 | 0.8155 |
| reranker / niche | 0.0617 | 0.0812 | 0.0105 | 0.8976 | 0.8128 |
| reranker / popular | 0.2763 | 0.2646 | 0.0075 | 0.5159 | 0.7555 |
| reranker / serendipitous | 0.1957 | 0.1452 | 0.0099 | 0.7983 | 0.9110 |

### Group (3 members: alex + 2 random MovieLens users)

| Variant | Strategy | Per-member NDCG@10 | Per-member Recall@50 | Fairness CV | Diversity@10 |
|:--------|:---------|------------------:|---------------------:|------------:|-------------:|
| baseline | average | 0.0000 | 0.0000 | 0.0000 | 0.7696 |
| baseline | least_misery | 0.0000 | 0.0000 | 0.0000 | 0.7956 |
| baseline | most_pleasure | 0.0000 | 0.0000 | 0.0000 | 0.5259 |
| baseline | consensus | 0.0000 | 0.0000 | 0.0000 | 0.7467 |
| baseline | hybrid | 0.0000 | 0.0000 | 0.0000 | 0.6607 |
| reranker | average / balanced | 0.0506 | 0.0728 | 0.7109 | 0.6100 |
| reranker | average / niche | 0.0000 | 0.0000 | 0.0000 | 0.6274 |
| reranker | least_misery / balanced | 0.0000 | 0.0083 | 0.0000 | 0.5404 |
| reranker | least_misery / niche | 0.0000 | 0.0000 | 0.0000 | 0.6078 |
| reranker | most_pleasure / balanced | 0.2108 | 0.1623 | 0.4912 | 0.6146 |
| reranker | most_pleasure / niche | 0.0239 | 0.0500 | 1.4142 | 0.8172 |
| reranker | consensus / balanced | 0.0000 | 0.0457 | 0.0000 | 0.6144 |
| reranker | consensus / niche | 0.0000 | 0.0000 | 0.0000 | 0.6274 |
| reranker | hybrid / balanced | 0.0000 | 0.0457 | 0.0000 | 0.5781 |
| reranker | hybrid / niche | 0.0000 | 0.0000 | 0.0000 | 0.6544 |
| reranker | group_taste_vector / balanced | 0.0000 | 0.0083 | 0.0000 | 0.8453 |
| reranker | group_taste_vector / niche | 0.0000 | 0.0000 | 0.0000 | 0.8828 |

---

## What I want your eye on (Phase 2)

1. **Content alignment:** the new content term uses 2M user-generated tags from ml-32m, so each rec is now scored partly on tag/genre similarity to your rated films. Do the modes feel more 'yours' than the Phase 1 version (e.g. world cinema, Ghibli, anime, classic noir clusters showing up if your taste pulls there)?
2. **Modes:** `niche` should now be hidden-gems-with-substance rather than long-tail noise. `serendipitous` should be delightful jumps that still rhyme with your taste. Are they?
3. **Group strategies:** `group_taste_vector` fuses everyone's taste tags into one signal. Does it find movies your group would *actually* watch together, not just compromise picks?
4. **Explanations / 'why' column:** does showing 'Genre overlap' help you trust the rec, or is it too coarse — would you want top-3 contributing rated films named?
5. **The friends in this demo are random MovieLens users with mismatched taste.** A real friend group would have far more overlap; if something looks weird, ask whether it's the algorithm or the mismatched test friends.
6. **Anything missing?** Examples of valid feedback: 'I want the content scorer weighted higher in balanced', 'niche picks too many 70s/80s — would be nice to filter by decade', 'tag noise still slipping through (e.g. tags like "01 10")'.
