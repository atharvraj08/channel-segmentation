# Learning Guide: Omnichannel Customer Segmentation

This guide is for a first-year Stat/ML student who knows basic Python. You can read it from top to bottom in about 30 minutes.

---

## 1. The big picture in 5 sentences

1. We want to split retail customers into groups ("segments") by how they shop online versus in store, without anyone telling us what the groups are. This is **unsupervised learning**.
2. We have no real customer data, so `generate_data.py` invents 5,000 customers drawn from 5 hidden "true" types, with enough noise that the types overlap.
3. The pipeline turns each customer's raw purchases into 15 numbers (features), rescales them, and lets **KMeans** find groups, trying k = 2…8 and keeping the k with the best **silhouette score**.
4. Because we secretly know the true types, we can grade the clusters with the **Adjusted Rand Index (ARI)**. We got **k = 5, silhouette 0.303 and ARI 0.941**, so the model recovered the planted structure very well.
5. 68 automated tests check every step. A "mutation check" (breaking the code on purpose) showed that the tests really catch bugs.

## 2. How to run it

From the project folder:

```bash
# one-time setup (already done on this machine)
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install -e .

# run the whole pipeline (≈ 3 s): data → features → clusters → metrics → charts
.venv/bin/python run_all.py

# run all 68 tests (≈ 10–15 s)
.venv/bin/python -m pytest -q

# run one test file, with names shown
.venv/bin/python -m pytest tests/test_cluster.py -v
```

Outputs appear in `data/` (the CSVs) and `outputs/` (`metrics.json`, `segment_profiles.csv`, `customer_segments.csv`, `charts/*.png`).

## 3. File-by-file walkthrough (in pipeline order)

### `src/segmentation/config.py`: the settings
Holds all the constants in one place: `SEED = 42`, `N_CUSTOMERS = 5000`, `K_RANGE = range(2, 9)`, the folder paths, the 5 product categories, and a fixed `END_DATE`. A fixed date matters because using "today" would give different data every day.

### `src/segmentation/generate_data.py`: the fake data with planted answers
- `SEGMENTS` (top of file): one dictionary per hidden type. It sets that type's share of customers, typical number of orders, how much it buys online (`online_ab`, Beta-distribution parameters), basket size, discount use, category preferences, engagement and loyalty. Read this first. It *is* the answer key, written as behaviour.
- `_draw_customer_params`: gives each customer their *own* parameters drawn around their type's values. This per-customer noise is what makes the types overlap.
- `_build_transactions` / `_build_engagement`: turn those parameters into ~136,000 purchase rows and one engagement row per customer.
- `generate_data(n_customers, seed)`: ties it all together with a single `np.random.default_rng(seed)`, so the same seed always gives identical data. It returns `(transactions, engagement, ground_truth)`. The truth is kept in its own table.
- `save_data`: writes the 3 CSVs.

### `src/segmentation/features.py`: from purchases to one row per customer
- `FEATURE_COLUMNS`: the 15 features (total spend, order count, average basket, online share, online/store orders, discount rate, recency, number of categories, top-category share, grocery share, email open rate, app/web sessions, loyalty).
- `_safe_divide`: turns 0/0 into 0. For example, a customer with no store orders gets 0 rather than NaN.
- `_purchase_features`, `_category_features`, `_engagement_features`: each computes one family of features with pandas `groupby`.
- `build_features`: joins them into one table indexed by `customer_id`.
- `LOG_COLUMNS` and `scale_features`: apply `np.log1p` to the skewed money and count columns, then `StandardScaler`. The result `X` is the matrix KMeans actually sees.
- Note: this file never touches the answer key. A test checks the source text for that.

### `src/segmentation/cluster.py`: finding the groups
- `_kmeans(k, seed)`: `KMeans(n_clusters=k, n_init=10, random_state=seed)`.
- `choose_k`: fits KMeans for every k from 2 to 8, computes the silhouette score for each, and returns the best k plus all the scores.
- `fit_segments`: the final fit, which returns one label (0…k-1) per customer.
- `save_segments`: writes `customer_segments.csv`.

### `src/segmentation/evaluate.py`: grading and describing
- `compute_silhouette`: how well separated the final clusters are.
- `load_ground_truth` and `compute_ari`: the **only** place the answer key is read. It is matched to predictions by `customer_id`, then scored with `adjusted_rand_score`.
- `build_profiles`: the average of key features per segment, plus size and share.
- `_name_one` / `name_segments`: plain if/else rules on those averages. The thresholds are named constants at the top of the file (`GROCERY_HEAVY = 0.5`, `ONLINE_HEAVY = 0.65`, …). The first rule that matches wins.
- `save_metrics`, `save_profiles`: write `metrics.json` and `segment_profiles.csv`.

### `src/segmentation/visualize.py`: the 4 charts
`plot_silhouette_by_k`, `plot_pca_segments`, `plot_segment_profiles` (heatmap), `plot_channel_mix` (online vs store bars), and `make_all_charts` to draw all of them.

### `run_all.py`: the one command
`run_pipeline(...)` calls everything above in order, and `_build_metrics` collects the numbers. `main()` prints a summary. Read `run_pipeline` top to bottom: it is the whole project in about 30 lines.

## 4. The ML concepts, simply

| Concept | Plain explanation | Where in the code |
|---|---|---|
| **Feature engineering** | Models can't read 136k purchase rows. We summarise each customer into a few meaningful numbers, like "what fraction of spend was online?" Good features matter more than fancy models. | `features.py: build_features` |
| **Log transform** | Spend and counts are *skewed*: a few VIPs spend 50× more than others. Left as is, they would dominate distances. `log1p(x) = log(1 + x)` squashes big values and handles 0. | `features.py: LOG_COLUMNS`, `scale_features` |
| **Standardization** | KMeans measures distance, so a feature measured in dollars (thousands) would drown one measured as a share (0–1). `StandardScaler` rescales every column to mean 0 and standard deviation 1, so all features get an equal say. | `features.py: scale_features` |
| **KMeans** | Pick k centre points, assign each customer to the nearest centre, move each centre to the average of its customers, and repeat until nothing changes. `n_init=10` runs this from 10 random starts and keeps the best. | `cluster.py: _kmeans`, `fit_segments` |
| **Silhouette score** | For each customer: *a* = average distance to its own cluster, *b* = average distance to the nearest other cluster, and the score is (b − a) / max(a, b). It runs from −1 to 1, and higher means tighter, better-separated clusters. | `cluster.py: choose_k`, `evaluate.py: compute_silhouette` |
| **Choosing k** | We don't know how many groups exist, so we try k = 2…8 and keep the one with the highest silhouette. Here k = 5 won (0.303), ahead of k = 4 (0.284) and k = 6 (0.273). See `outputs/charts/silhouette_by_k.png`. | `cluster.py: choose_k` |
| **PCA** | We have 15 features but screens are 2-D. PCA finds the 2 directions along which the data varies most and projects onto them. This is only for the picture; clustering uses all 15 features. | `visualize.py: plot_pca_segments` |
| **ARI** | Compares two ways of grouping the same customers, ignoring what the groups are *called* (cluster "3" vs "Occasional Browsers"). 1 = identical groupings, ≈ 0 = no better than random. Ours is 0.941. | `evaluate.py: compute_ari` |
| **Why synthetic ground truth** | Real unsupervised problems have no answer key, so you can only judge clusters by "do they look sensible?". By planting the answer ourselves (and hiding it from the model), we can measure objectively whether the method works. The catch: it only proves the method works on data like *our* simulation. | `generate_data.py` (plants), `evaluate.py` (grades) |

## 5. How the tests work

- **Fixtures** (`tests/conftest.py`): expensive things are computed once per test session and shared. These include the generated data, the feature table, `X`, the `choose_k` result, the labels, and one full `run_pipeline` run into a temporary folder. That's why 68 tests take only seconds.
- `test_generate_data.py` (17 tests): columns, no missing values, valid ranges, every type ≥ 8%, same seed gives identical data, different seed gives different data, and no answer-key leakage into the data files.
- `test_features.py` (19 tests): one row per customer, no NaN/inf, hand-checked values on tiny made-up customers (only-online, only-store), scaled columns with mean 0 and std 1, and a check that the source of `features.py` and `cluster.py` never mentions the answer key.
- `test_cluster.py` (10 tests): k is 5 ± 1, silhouette ≥ 0.15, every cluster ≥ 3% of customers, a different KMeans seed gives nearly the same clusters (ARI ≥ 0.80), and results are deterministic.
- `test_evaluate.py` (18 tests): **ARI ≥ 0.70 on the full pipeline**, at least one mostly-online and one mostly-store segment, the profile table, the names, and the `metrics.json` keys.
- `test_pipeline.py` (3 tests): runs `python run_all.py` as a real subprocess, which must finish in under 120 s with exit code 0 and write every output file. It also checks that two runs give the same metrics.

**The mutation check** (see `REVIEW.md`): a reviewer copied the project and broke it 4 different ways, one at a time. The tests caught every one:

| Bug introduced | Test that caught it |
|---|---|
| Shuffle feature rows, so labels no longer line up with customers | `test_pipeline_ari_at_least_0_70` (ARI fell to ≈ 0) |
| Always choose k = 2 | `test_chosen_k_is_five_plus_minus_one` |
| Skip scaling | k test plus the mean/std tests |
| Generator ignores the seed | the same-seed reproducibility tests |

This shows the tests are not just "does it run?" checks. They notice when the ML is actually wrong.

## 6. Try this next (5 exercises)

1. **Raise the noise.** In `generate_data.py`, increase `ORDER_AMOUNT_SIGMA` and `CUSTOMER_BASKET_SIGMA`, or make the `online_ab` pairs closer together. Rerun and watch ARI fall. At what point does the ARI test fail? *Expect: types blur together, and silhouette and ARI both drop.*
2. **Drop the log transform.** Set `LOG_COLUMNS = []` in `features.py` and rerun. *Expect: spend-heavy VIPs dominate the distances, k may change, and ARI should fall.* Run the tests to see which ones notice.
3. **Swap in GaussianMixture.** In `cluster.py`, replace `KMeans` with `sklearn.mixture.GaussianMixture(n_components=k, random_state=seed)` (use `.fit(X).predict(X)`). *Expect: similar ARI. GMM allows oval-shaped clusters of different sizes.*
4. **Plant a 6th type.** Add, say, "Weekend Home Improvers" to `SEGMENTS` (home-category heavy, store, big baskets), fix the shares so they sum to 1, and set `N_TRUE_SEGMENTS = 6`. *Expect: does silhouette now pick k = 6?* Note that the tests assume 5 types, so some will fail. That's the point.
5. **Remove a feature.** Set `MODEL_FEATURES` to everything except `online_share`. *Expect: the channel split gets weaker. Check the A11 test and `channel_mix.png`.*

After each experiment, put things back with `git checkout -- src/`.

## 7. Glossary

- **Segment / cluster**: a group of customers that behave alike.
- **Unsupervised learning**: finding structure without labelled answers.
- **Feature**: one number describing a customer (e.g. online share).
- **Skewed**: most values are small, and a few are huge.
- **log1p**: log(1 + x), a log that works at 0.
- **Standardize (z-score)**: (value − mean) / standard deviation.
- **KMeans**: clustering by repeatedly assigning points to the nearest centre.
- **k**: the number of clusters.
- **Centroid**: the average point of a cluster.
- **n_init**: how many random restarts KMeans tries.
- **Silhouette score**: a −1…1 measure of cluster separation (higher is better).
- **PCA**: squeezing many features into 2–3 directions, for plotting.
- **ARI (Adjusted Rand Index)**: agreement between two groupings, corrected for chance (1 = perfect).
- **Ground truth**: the real answer (here, the planted customer types).
- **Seed**: a number that fixes the random generator, so results are repeatable.
- **Fixture (pytest)**: shared setup that tests can reuse.
- **Mutation testing**: breaking code on purpose to check that the tests notice.
- **Data leakage**: the model accidentally seeing the answers. This is prevented here, and a test checks for it.
