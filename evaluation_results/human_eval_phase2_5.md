# Phase 2.5 Human-Eval Check-in — Tag Genome 2021 content features

**Purpose:** the content scorer is now Tag Genome 2021 — GroupLens's curated, deep-learning-fitted tag-relevance dataset (1,084 tags × 9,734 movies, 10.5M relevance scores in [0, 1]). Drop-in replacement for the noisy user-tag TF-IDF. Offline metrics show clear gains on group strategies (the headline use case) and mixed-to-flat on individual modes; qualitative output is the better signal here.

**Models:** `models/svd_full_slim.pkl` (SVD on ml-32m, trainset slimmed for inference) + `models/als_full.pkl` (implicit ALS on ml-32m) + `data/content_genome.npz` (Tag Genome 2021).
**User:** `alex_data/ratings_with_tmdb.csv` (220 ratings mapped to MovieLens 32m, plus 258 watched-but-unrated films also excluded from recs).
**Synthetic friends:** two random MovieLens users with 155 and 177 ratings respectively.

---

## Individual recommendations — by mode

### Mode: `balanced`

| # | Score | Pop | Title | Genre overlap | Source |
|--:|------:|:---:|:------|:--------------|:------:|
| 1 | 1.791 | blockbuster | Godfather, The (1972) | Drama, Crime | svd+als |
| 2 | 1.647 | popular | Casablanca (1942) | Drama, Romance | svd+als |
| 3 | 1.647 | obscure | Twin Peaks (1989) | Drama, Mystery | svd+als |
| 4 | 1.618 | obscure | Neon Genesis Evangelion: The End of Evangelion (Shin seiki Evangelion Gekijô-ban: Air/Magokoro wo, kimi ni) (1997) | Drama, Action, Sci-Fi | svd+als |
| 5 | 1.535 | popular | Parasite (2019) | Drama, Comedy | svd+als |
| 6 | 1.513 | obscure | Disappearance of Haruhi Suzumiya, The (Suzumiya Haruhi no shôshitsu) (2010) | Drama, Adventure, Sci-Fi | svd |
| 7 | 1.504 | obscure | Bad Sleep Well, The (Warui yatsu hodo yoku nemuru) (1960) | Drama, Thriller | svd |
| 8 | 1.502 | obscure | Shoplifters (2018) | Drama | svd+als |
| 9 | 1.497 | obscure | Don't Look Back (1967) | Documentary, Musical | svd |
| 10 | 1.478 | obscure | Time of the Gypsies (Dom za vesanje) (1989) | Drama, Comedy, Fantasy | svd |

### Mode: `niche`

| # | Score | Pop | Title | Genre overlap | Source |
|--:|------:|:---:|:------|:--------------|:------:|
| 1 | 1.346 | obscure | Interrogation (Przesluchanie) (1989) | Drama, Thriller, Crime | svd |
| 2 | 1.241 | obscure | Absolute Giganten (1999) | Drama, Comedy, Action | svd |
| 3 | 1.193 | obscure | Mushishi: The Shadow That Devours the Sun (2014) | Adventure, Fantasy, Animation | svd |
| 4 | 1.156 | obscure | Sweeney Todd: The Demon Barber of Fleet Street (1982) | Drama, Thriller, Horror | svd |
| 5 | 1.135 | obscure | Twin Peaks (1989) | Drama, Mystery | svd+als |
| 6 | 1.117 | obscure | Time of the Gypsies (Dom za vesanje) (1989) | Drama, Comedy, Fantasy | svd |
| 7 | 1.116 | obscure | Everything's Gonna Be Great (1998) | Drama, Comedy, Adventure | svd |
| 8 | 1.108 | obscure | Heartstone (2016) | Drama, Romance | svd |
| 9 | 1.106 | obscure | Cuba and the Cameraman (2017) | Documentary | svd |
| 10 | 1.095 | obscure | Le Plaisir (1952) | Drama, Comedy | svd |

### Mode: `popular`

| # | Score | Pop | Title | Genre overlap | Source |
|--:|------:|:---:|:------|:--------------|:------:|
| 1 | 3.251 | blockbuster | Godfather, The (1972) | Drama, Crime | svd+als |
| 2 | 3.074 | popular | Casablanca (1942) | Drama, Romance | svd+als |
| 3 | 2.936 | popular | Parasite (2019) | Drama, Comedy | svd+als |
| 4 | 2.911 | popular | Godfather: Part II, The (1974) | Drama, Crime | svd+als |
| 5 | 2.840 | popular | One Flew Over the Cuckoo's Nest (1975) | Drama | svd+als |
| 6 | 2.822 | popular | Vertigo (1958) | Drama, Thriller, Romance | svd+als |
| 7 | 2.763 | obscure | Twin Peaks (1989) | Drama, Mystery | svd+als |
| 8 | 2.743 | niche | Kiki's Delivery Service (Majo no takkyûbin) (1989) | Drama, Adventure, Fantasy | svd+als |
| 9 | 2.729 | popular | Kill Bill: Vol. 1 (2003) | Thriller, Action, Crime | svd+als |
| 10 | 2.699 | popular | Chinatown (1974) | Thriller, Crime, Mystery | svd+als |

### Mode: `serendipitous`

| # | Score | Pop | Title | Genre overlap | Source |
|--:|------:|:---:|:------|:--------------|:------:|
| 1 | 1.512 | popular | Casablanca (1942) | Drama, Romance | svd+als |
| 2 | 1.261 | obscure | Divorce - Italian Style (Divorzio all'italiana) (1961) | Comedy | svd |
| 3 | 1.259 | popular | Chinatown (1974) | Thriller, Crime, Mystery | svd+als |
| 4 | 1.246 | obscure | Neon Genesis Evangelion: The End of Evangelion (Shin seiki Evangelion Gekijô-ban: Air/Magokoro wo, kimi ni) (1997) | Drama, Action, Sci-Fi | svd+als |
| 5 | 1.188 | obscure | Wings of Hope (Julianes Sturz in den Dschungel) (2000) | Adventure, Documentary | svd |
| 6 | 1.167 | blockbuster | Godfather, The (1972) | Drama, Crime | svd+als |
| 7 | 1.134 | obscure | Sweeney Todd: The Demon Barber of Fleet Street (1982) | Drama, Thriller, Horror | svd |
| 8 | 1.113 | obscure | Twin Peaks (1989) | Drama, Mystery | svd+als |
| 9 | 1.063 | obscure | Everything's Gonna Be Great (1998) | Drama, Comedy, Adventure | svd |
| 10 | 1.061 | niche | Grave of the Fireflies (Hotaru no haka) (1988) | Drama, Animation, War | svd+als |

---

## Group recommendations — 3-friend group, mode=`balanced`

Each row shows the predicted score per member (normalized 0-1) and the fairness coefficient of variation (lower = more uniform agreement).

### Strategy: `average`

| # | Score | Fair | Title | Pop | Per-member scores |
|--:|------:|-----:|:------|:---:|:------------------|
| 1 | 0.988 | 0.00 | Godfather, The (1972) | blockbuster | alex=0.99 |
| 2 | 0.926 | 0.05 | City of God (Cidade de Deus) (2002) | popular | alex=0.88, friend_B=0.97 |
| 3 | 0.899 | 0.00 | Goodfellas (1990) | popular | friend_B=0.90 |
| 4 | 0.890 | 0.00 | Clockwork Orange, A (1971) | popular | friend_A=0.89 |
| 5 | 0.883 | 0.11 | Godfather: Part II, The (1974) | popular | alex=0.91, friend_A=0.75, friend_B=0.98 |
| 6 | 0.881 | 0.00 | Dr. Strangelove or: How I Learned to Stop Worrying and Love the Bomb (1964) | popular | friend_B=0.88 |
| 7 | 0.828 | 0.13 | Kill Bill: Vol. 2 (2004) | popular | alex=0.72, friend_A=0.93 |
| 8 | 0.797 | 0.14 | Kill Bill: Vol. 1 (2003) | popular | alex=0.68, friend_A=0.91 |
| 9 | 0.791 | 0.23 | Taxi Driver (1976) | popular | friend_A=0.61, friend_B=0.97 |
| 10 | 0.789 | 0.00 | Lord of the Rings: The Return of the King, The (2003) | blockbuster | friend_B=0.79 |

### Strategy: `least_misery`

| # | Score | Fair | Title | Pop | Per-member scores |
|--:|------:|-----:|:------|:---:|:------------------|
| 1 | 0.752 | 0.11 | Godfather: Part II, The (1974) | popular | alex=0.91, friend_A=0.75, friend_B=0.98 |
| 2 | 0.696 | 0.07 | Everything's Gonna Be Great (1998) | obscure | alex=0.82, friend_A=0.72, friend_B=0.70 |
| 3 | 0.689 | 0.11 | Invisible War, The (2012) | obscure | alex=0.69, friend_A=0.89, friend_B=0.73 |
| 4 | 0.668 | 0.07 | Divorce - Italian Style (Divorzio all'italiana) (1961) | obscure | alex=0.78, friend_A=0.67, friend_B=0.67 |
| 5 | 0.663 | 0.09 | Teddy Bear (Mis) (1981) | obscure | alex=0.70, friend_A=0.81, friend_B=0.66 |
| 6 | 0.644 | 0.12 | Heartstone (2016) | obscure | alex=0.86, friend_A=0.76, friend_B=0.64 |
| 7 | 0.631 | 0.10 | Adventure Time: Islands (2017) | obscure | alex=0.75, friend_A=0.63, friend_B=0.82 |
| 8 | 0.630 | 0.08 | Beastie Boys: Sabotage (1994) | obscure | alex=0.63, friend_A=0.77, friend_B=0.68 |
| 9 | 0.620 | 0.11 | The Adventures of Sherlock Holmes and Doctor Watson: King of Blackmailers (1980) | obscure | alex=0.66, friend_A=0.80, friend_B=0.62 |
| 10 | 0.617 | 0.11 | The Amazing Screw-On Head (2006) | obscure | alex=0.62, friend_A=0.77, friend_B=0.80 |

### Strategy: `most_pleasure`

| # | Score | Fair | Title | Pop | Per-member scores |
|--:|------:|-----:|:------|:---:|:------------------|
| 1 | 1.000 | 0.65 | Twin Peaks (1989) | obscure | alex=1.00, friend_A=0.09, friend_B=0.67 |
| 2 | 1.000 | 0.52 | All the President's Men (1976) | niche | alex=0.30, friend_A=0.45, friend_B=1.00 |
| 3 | 1.000 | 0.54 | Room, The (2003) | obscure | alex=0.19, friend_A=1.00, friend_B=0.63 |
| 4 | 0.990 | 0.44 | Casablanca (1942) | popular | alex=0.99, friend_A=0.38 |
| 5 | 0.988 | 0.00 | Godfather, The (1972) | blockbuster | alex=0.99 |
| 6 | 0.987 | 0.64 | Bridge on the River Kwai, The (1957) | popular | alex=0.61, friend_A=0.10, friend_B=0.99 |
| 7 | 0.983 | 0.11 | Godfather: Part II, The (1974) | popular | alex=0.91, friend_A=0.75, friend_B=0.98 |
| 8 | 0.972 | 0.23 | Taxi Driver (1976) | popular | friend_A=0.61, friend_B=0.97 |
| 9 | 0.971 | 0.05 | City of God (Cidade de Deus) (2002) | popular | alex=0.88, friend_B=0.97 |
| 10 | 0.965 | 0.38 | Vertigo (1958) | popular | alex=0.96, friend_A=0.35, friend_B=0.92 |

### Strategy: `consensus`

| # | Score | Fair | Title | Pop | Per-member scores |
|--:|------:|-----:|:------|:---:|:------------------|
| 1 | 0.988 | 0.00 | Godfather, The (1972) | blockbuster | alex=0.99 |
| 2 | 0.924 | 0.05 | City of God (Cidade de Deus) (2002) | popular | alex=0.88, friend_B=0.97 |
| 3 | 0.899 | 0.00 | Goodfellas (1990) | popular | friend_B=0.90 |
| 4 | 0.890 | 0.00 | Clockwork Orange, A (1971) | popular | friend_A=0.89 |
| 5 | 0.881 | 0.00 | Dr. Strangelove or: How I Learned to Stop Worrying and Love the Bomb (1964) | popular | friend_B=0.88 |
| 6 | 0.873 | 0.11 | Godfather: Part II, The (1974) | popular | alex=0.91, friend_A=0.75, friend_B=0.98 |
| 7 | 0.816 | 0.13 | Kill Bill: Vol. 2 (2004) | popular | alex=0.72, friend_A=0.93 |
| 8 | 0.789 | 0.00 | Lord of the Rings: The Return of the King, The (2003) | blockbuster | friend_B=0.79 |
| 9 | 0.784 | 0.14 | Kill Bill: Vol. 1 (2003) | popular | alex=0.68, friend_A=0.91 |
| 10 | 0.764 | 0.17 | Grave of the Fireflies (Hotaru no haka) (1988) | niche | alex=0.91, friend_A=0.65 |

### Strategy: `hybrid`

| # | Score | Fair | Title | Pop | Per-member scores |
|--:|------:|-----:|:------|:---:|:------------------|
| 1 | 1.482 | 0.00 | Godfather, The (1972) | blockbuster | alex=0.99 |
| 2 | 1.367 | 0.05 | City of God (Cidade de Deus) (2002) | popular | alex=0.88, friend_B=0.97 |
| 3 | 1.348 | 0.00 | Goodfellas (1990) | popular | friend_B=0.90 |
| 4 | 1.335 | 0.00 | Clockwork Orange, A (1971) | popular | friend_A=0.89 |
| 5 | 1.322 | 0.00 | Dr. Strangelove or: How I Learned to Stop Worrying and Love the Bomb (1964) | popular | friend_B=0.88 |
| 6 | 1.259 | 0.11 | Godfather: Part II, The (1974) | popular | alex=0.91, friend_A=0.75, friend_B=0.98 |
| 7 | 1.188 | 0.13 | Kill Bill: Vol. 2 (2004) | popular | alex=0.72, friend_A=0.93 |
| 8 | 1.184 | 0.00 | Lord of the Rings: The Return of the King, The (2003) | blockbuster | friend_B=0.79 |
| 9 | 1.138 | 0.14 | Kill Bill: Vol. 1 (2003) | popular | alex=0.68, friend_A=0.91 |
| 10 | 1.121 | 0.08 | 2001: A Space Odyssey (1968) | popular | friend_A=0.71, friend_B=0.83 |

### Strategy: `group_taste_vector`

| # | Score | Fair | Title | Pop | Per-member scores |
|--:|------:|-----:|:------|:---:|:------------------|
| 1 | 1.868 | 0.33 | Network (1976) | niche | alex=0.70, friend_A=0.79, friend_B=1.42 |
| 2 | 1.711 | 0.67 | All the President's Men (1976) | niche | alex=0.30, friend_A=0.81, friend_B=1.91 |
| 3 | 1.613 | 0.55 | Kiki's Delivery Service (Majo no takkyûbin) (1989) | niche | alex=1.16, friend_A=1.25, friend_B=0.19 |
| 4 | 1.603 | 0.12 | Heima (2007) | obscure | alex=1.07, friend_A=0.89, friend_B=1.18 |
| 5 | 1.599 | 0.11 | Decalogue, The (Dekalog) (1989) | obscure | alex=1.08, friend_A=1.32, friend_B=1.40 |
| 6 | 1.586 | 0.38 | North by Northwest (1959) | popular | alex=1.29, friend_A=0.57, friend_B=1.61 |
| 7 | 1.532 | 0.23 | Malcolm X (1992) | niche | alex=0.60, friend_A=0.80, friend_B=1.06 |
| 8 | 1.522 | 0.39 | Vertigo (1958) | popular | alex=1.56, friend_A=0.57, friend_B=1.69 |
| 9 | 1.503 | 0.17 | Samurai I: Musashi Miyamoto (Miyamoto Musashi) (1954) | obscure | alex=0.98, friend_A=0.64, friend_B=0.81 |
| 10 | 1.501 | 0.33 | Chinatown (1974) | popular | alex=1.31, friend_A=0.73, friend_B=1.75 |

---

## Offline metrics — baseline vs Phase 1 re-ranker

### Individual (averaged across alex + 20 sampled MovieLens users)

| Variant | NDCG@10 | Recall@50 | Coverage | Gini popularity (lower = less popular bias) | Diversity@10 |
|:--------|--------:|----------:|---------:|--------------------------------------------:|-------------:|
| reranker / balanced | 0.1785 | 0.1428 | 0.0103 | 0.8105 | 0.8237 |
| reranker / niche | 0.0621 | 0.0724 | 0.0106 | 0.8684 | 0.8122 |
| reranker / popular | 0.2661 | 0.2664 | 0.0075 | 0.5173 | 0.7675 |
| reranker / serendipitous | 0.1528 | 0.1427 | 0.0101 | 0.8069 | 0.9017 |

### Group (3 members: alex + 2 random MovieLens users)

| Variant | Strategy | Per-member NDCG@10 | Per-member Recall@50 | Fairness CV | Diversity@10 |
|:--------|:---------|------------------:|---------------------:|------------:|-------------:|
| reranker | average / balanced | 0.0766 | 0.1913 | 0.7965 | 0.5544 |
| reranker | average / niche | 0.0000 | 0.0000 | 0.0000 | 0.6404 |
| reranker | least_misery / balanced | 0.0260 | 0.0312 | 1.4142 | 0.5944 |
| reranker | least_misery / niche | 0.0000 | 0.0000 | 0.0000 | 0.6759 |
| reranker | most_pleasure / balanced | 0.2668 | 0.1913 | 0.1323 | 0.5775 |
| reranker | most_pleasure / niche | 0.0309 | 0.0500 | 1.4142 | 0.6566 |
| reranker | consensus / balanced | 0.0327 | 0.1580 | 1.0104 | 0.6193 |
| reranker | consensus / niche | 0.0000 | 0.0000 | 0.0000 | 0.6496 |
| reranker | hybrid / balanced | 0.0333 | 0.1101 | 0.9852 | 0.6396 |
| reranker | hybrid / niche | 0.0000 | 0.0000 | 0.0000 | 0.6463 |
| reranker | group_taste_vector / balanced | 0.0000 | 0.0083 | 0.0000 | 0.8133 |
| reranker | group_taste_vector / niche | 0.0000 | 0.0000 | 0.0000 | 0.8157 |

---

## What I want your eye on (Phase 2)

1. **Content alignment:** the new content term uses 2M user-generated tags from ml-32m, so each rec is now scored partly on tag/genre similarity to your rated films. Do the modes feel more 'yours' than the Phase 1 version (e.g. world cinema, Ghibli, anime, classic noir clusters showing up if your taste pulls there)?
2. **Modes:** `niche` should now be hidden-gems-with-substance rather than long-tail noise. `serendipitous` should be delightful jumps that still rhyme with your taste. Are they?
3. **Group strategies:** `group_taste_vector` fuses everyone's taste tags into one signal. Does it find movies your group would *actually* watch together, not just compromise picks?
4. **Explanations / 'why' column:** does showing 'Genre overlap' help you trust the rec, or is it too coarse — would you want top-3 contributing rated films named?
5. **The friends in this demo are random MovieLens users with mismatched taste.** A real friend group would have far more overlap; if something looks weird, ask whether it's the algorithm or the mismatched test friends.
6. **Anything missing?** Examples of valid feedback: 'I want the content scorer weighted higher in balanced', 'niche picks too many 70s/80s — would be nice to filter by decade', 'tag noise still slipping through (e.g. tags like "01 10")'.
