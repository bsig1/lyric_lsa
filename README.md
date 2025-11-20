# Lyric LSA Project

**Higher Level Overview**
The ultimate goal is to embed artists into a semantic space based purely on their lyrical content and compare that structure to their 
genre-based relationships. The central hypothesis is that
*lyrical similarity* between two artists "correlates to"/"can predict" the *genre similarity* between two artists.

## Step 1

Find a dataset of many different songs and their lyrics with artists.

In this case, it was a subset of the Genius lyric database found online for download in the form of a csv.

Convert this into the form of a database for access efficiency, for the sake of the scale of the data.
anything else would be slow.

## Step 2:
Take all of the lyrics in the database and process them.
- Start by removing punctuation and lowercasing all letters.
- Then begin "lemmatization" the process of taking words and bringing them into a common form.
- The python package SpaCy does dictionary lookups to merge words like (make making makes made) into a common (make)
- Use spacy on all lyrics to group together and count common words for each song.
- The final output is word "tokens" where similar words all become one token.

## Step 3:
Take the lyrics and associate them to artists.
- Group all lyrics in all tracks by artist. Resulting in a tally for each word for each artist with words merged together.

## Step 4:
Build lyric matrix
- Define a sparce matrix X whose rows are the terms and columns are artists
- Run SciKit TfidfTransformer, this is an algorithm that weights rarer, more meaningful words higher, while weighting down words like
(I a that the) that are less meaningful. Then the algorithm normalizes document length. TF-IDF is essential because raw word counts are
dominated by extremely common words; TF-IDF reweights terms so that rare, content-bearing words drive the embedding.
- Run SVD on this matrix
Much easier said than done. Consider this matrix, the dimensions are the number of different terms (n), by the number
of artists (m) ~(16,000,000 X 700,000). A naive implementation of SVD would be basically computationally impossible. The only
reason this matrix is useable right now is because of how sparse it is, running SVD would create twice as many dense values
this is just a non-starter. So instead, this program
uses the Python SciKit package's Truncated SVD, with Fit Transform. Instead of computing the entire transform, then reducing
to the number of singular values (k). This function immediately reduces the matrices to (k X n), orders of magnitude more workable
than the full matrix.
Here's how it works:

### Randomized SVD
*Nathan Halko, Per-Gunnar Martinsson, Joel Tropp (2011)*
| Step | What happens                                            |
| ---- | ------------------------------------------------------- |
| 1    | Generate Gaussian random matrix (Ω)                     |
| 2    | Compute sketch (Y = XΩ)                                 |
| 3    | Do power iterations (Y = X (X^T Y))                     |
| 4    | QR decomposition (Y = QR)                               |
| 5    | Project: (B = Q^T X)                                    |
| 6    | SVD of small matrix: (B=U' Σ V^T)                       |
| 7    | Reconstruct left singular vectors: (U_k = Q U'_k)       |
| 8    | Return embedding: (Z = U_k Σ_k)                         |
| 9    | Store artist vectors U_ki Σ_ki for i in artists         |

0. Performing randomzied SVD on matrix X that is (n X d).
There are a couple important hyper parameters q, k and p. k and p relate to the dimensions of the aproximation and q relates
to the number of power iterations, more on both later.

1. Start by defining a gaussian projection matrix Ω. That is a matrix where every number is an instantiated random
variable whose values are distributed on the unit normal N(0,1). This is chosen because a Gaussian matrix spans 
all directions uniformly, is rotationally invariant, concentrates strongly, and guarantees that the 
randomized projection captures the dominant singular subspace with extremely high probability. It is the optimal choice
to create a projection that isn't effected by rotations. Multiplying Ω by orthogonal matricies does not change it. This
is important because the singular vectors of X are directions, and Ω needs to treat all of these directions equally.

2. Define Y = XΩ, this projects X into the dimensions n x (k+p).
Because Ω is defined as a Gaussian Projection Matrix, and thus rotationally agnostic, Y will encode information about the 
important singular directions of X.

Why k+p? Because of Oversampling. This is a process that allows for additional accuracy 
by creating a matrix with size greater than the end result. The ideal output of this algorithm is a k rank approximation 
of X; starting with a matrix of size n X k would be dangerous, if even 2 columns were linearly dependent, the k rank approximation
would become invalid, but adding some amount of extra room(p) allows for a few of the weaker columns to be dropped, in
fact they will be dropped to give the final k sized approximation.

What makes a column weak?
- Some columns of Ω may land mostly in low-energy singular directions

- Some may be nearly linearly dependent

- Some singular vectors may be poorly represented

- Numerical stability can suffer

- The approximation quality becomes noisy and unpredictable

Adding a few columns gives enough wiggle room to be able to drop "misbehaving" columns.


3. Run power iteration, this is a process that tries to numerically stabilize the, at this point completely random Ω matrix.
This process is simply Y = X (X^T Y). Which iteratively improves the accuracy of Y. This process
essentially amplifies the effect of singular values of X on the output Y, while retaining its overall "structure".

Expanding the SVD of X shows that this process does not effect either the right singular vectors or left singular vectors,
since they are orthogonal and cancel out in X X^T, instead the only thing that is affected is every time this process is ran
the singular values of X are squared. This stabilizes the approximation Y by amplifying the larger, *more important* singular
values of X, and diminishes the *less important* singular values.
This is still computationally cheap in comparison to running the full SVD of X.

4. Now an absolutely essential step is to decompose y into QR where Q has orthonormal columns. QR is the bridge between a 
random sketch and a stable subspace representation. So why is it so important?

- **Orthonormalization** of the columns of y. By running QR decomposition, a lot of noise is removed, Q is the important information
about singular directions, while R is somewhat random stretching directions heavily influenced by Σ.

- **Balancing Singular Values** in step 3 the larger singular values of X are increased exponentially and the smaller values 
are reduced to near 0, wich with floating point precision is 0. By normalizing the columns a singular value is prevented from 
completely dominating all of the columns into a single directions, and the matrix is rescaled into something more usable.

- **Numeric Stability** an important quality of QR decomosition is it guarentees a well-conditioned matrix. Without this,
later calculations could become wildly different from what they are supposed to be. Computational error could meaningfully
change the information in the matrix, especially in very large matricies.

*Steps 3 and 4 are repeated q times. In practice, q is usually between 5 and 10. Repeating these steps increases the accuracy
of the process, but anything greater than 10 has quickly diminishing returns.*

5. Now we have a matrix Q that is a projection matrix that has strongly encoded into it the directional qualities of X,
using this compute B = Q^T X. This projects the important information of X into the desired subspace (of size Q).
Note the Q^T, this is an important conceptual detail of this algorithm. Since Q has orthonormal columns, its pseudo-inverse can
be calculated this way. (It would be the inverse if Q was square). Q is a good approximation of the 
top-k subspace of X, so Q† extracts the part of X living in that subspace. Q, then, approximates U in the context of a SVD
of X, thus Q^T X ≈ Σk​V^⊤.

6. Finally SVD can be ran. B is much smaller than the starting dimensions of X, so it is very possible to compute B's
SVD.​B=U' Σ V^T.

Then take the top k slices of the SVD of B. The SVD of B is (approximately) the SVD of the top-k part of X.
Nothing outside that subspace survived the projection. Define these as U'_k Σ_k V_k^T (The underscore is my best attempt at
a subscript in Markdown).

7. Now recover the left singular subspace by U_k = Q U'_k.

8. Return embedding: Z = U_k Σ_k. This is what is stored. Note that Z = U_k​Σ_k​= XV_k. Multiplying by the singular values Σ ensures that
   artists with larger contributions along important latent directions receive proportionally stronger weights.

In the context of this problem, V represents the term embeddings of each singular value, while U represents the artist embeddings.
The vector stored for each artist in the artist embeddings weighted by the corresponding singular values, then normalized for 
consistency between artists. Without this something like an artist having a larger amount of songs could give them a 
larger simmilarity score between other artists, somewhat arbitrarily.

## Step 5:
Define Genre vectors
- For each artist define a "genre vector" these vectors could be something like
(rock,pop,alternative)
(20%,10%,70%)
Normalize each genre vector under the 2 norm so all artists live on the same unit hypersphere.
- An artists genre is defined by the ratio of how many of there songs are reported as what genre under the dataset.
- In the data set there are 6 genres, so every vector is of length 6, these genres are:
(rock,pop,misc,rap,r&b,country)

## Step 6:
What is similarity?

For genre vectors, we have a normalized vector only with only positive domain. These vectors point in some direction in
hyperspace that is associated with some amount of whatever genre. For example in 2D if the x axis is rock and
the y axis is pop, artists would be an angle in the unit circle between 0 and 90 degrees, where 0 is more rock
and 90 is more pop. Anything in between is some amount of each, given by the angle.

To get "similarity" we use cosine similarity. Given two unit vectors, their cos similarity is calculated simply through
the dot product. This gives a number between 0 and 1; intuitively a score of how parallel they are, more riggorously,
it is the cosine of the angle between the two vectors. When the vectors are perpendicular, they have a shared genre score
of 0. This could happen if one artist were 100% rock and another were 100% pop. Whereas, if this number is 1, the artists have
100% the same genre.

LSA vectors are very similar, but they are less intuitive because instead of being defined through genres, each scalar in the
vector is associated with a latent variable inside of the data. These latent variables are nebulous variables that have some
kind of meaning, but it is unclear what they mean in a vacuum. But through the same process, cosine similarity, the vectors can
be compared through the dot product. The higher it is, the more similar the encoding of the lyrical content of two artists are. 
Note all of these vectors in these data are normalized aswell.

## Step 6:
The research question this entire process is built around is trying to compare Latent Semantic Analysis difference to genre difference..
So how do we test this?

We can start by taking an artist and plotting its similarity to each other artist on a scatterplot. On the X axis we can put
genre similarity and the Y axis we can put LSA similarity. Drawing a least squared regression line through these data gives
a correlation, standard error, standard deviation, etc.

Running this same process for many different random artists, gives some kind of average correlation between artists.
Ideally, this would be ran for every artist, but this is too computationally expensive (700,000 artists ran on 700,000 artists
is about 4.9*10^11 operations). So instead we sample a subset of these artists, (1000 in this case).

For each k value (10,50,100,300,500,800) and the artists 1 -> 9 we run both Pearson and Spearman regressions between this artist
and all other artists. Pearson and Spearman are used to test for correlation in both
direct comparison, and rank comparison. Spearman is especially useful here because absolute distances vary widely across artists,
but rankings of similar artists remain stable. Note that the parameters were setup to have a
95% confidence interval.

In practice, larger k values than 800 became incomputable, even with a very strong computer.

## Step 7:
Note in the core idea of this assignment is different k values for svd. This analysis was ran with k values (10,50,100,300,500,800)
These correlation values are measured against different k values to see how different levels of compression of the semantic data
results in different values of correlation.


# Data: 
Across all tested LSA dimensions, LSA-based artist distances are significantly correlated with genre-based distances (p ≪ 0.001 due to large n). 
At low dimensionality (k = 10), effect sizes are negligible (Pearson r ≈ 0.01–0.03). For k ≥ 50, correlations are small-to-moderate (r ≈ 0.3–0.4), 
and for k between 300 and 800 they reach moderate strength (up to r ≈ 0.53), indicating that LSA captures a nontrivial but incomplete component of genre structure.


## What does the data look like at different k values?

| LSA Dim (k) | Genre Distances (μ ± σ) | LSA Distances (μ ± σ) | Difference σ        | Interpretation                                               |
| ----------- | ----------------------- | --------------------- | ------------------- | ------------------------------------------------------------ |
| **10**      | **0.532 ± 0.43**        | **0.0384 ± 0.0762**   | **0.0767**          | LSA collapses semantic space; distances extremely compressed |
| **50**      | **0.532 ± 0.43**        | **0.1859 ± 0.1577**   | **0.1599**          | Distances more expressive; LSA begins to encode structure    |
| **100**     | **0.532 ± 0.43**        | **0.2299 ± 0.1648**   | **0.1643**          | Distances further separate; captures more semantic variance  |
| **300**     | **0.532 ± 0.43**        | **0.7898 ± 0.1144**   | **0.1146**          | Distances well spread; LSA strongly encodes artist semantics |
| **500**     | **0.532 ± 0.43**        | **0.7978 ± 0.0793**   | **0.0794**          | Very stable; variance decreases as k increases               |
| **800**     | **0.532 ± 0.43**        | **0.8352 ± 0.1002**   | **0.1001**          | Slightly more variance; distances remain high-quality        |

## What do different artist's plots look like?

| Artist (ID) | Genre Dist μ ± σ | LSA Dist μ ± σ  | Difference σ | Notes                                       |
| ----------- | ---------------- | --------------- | ------------ | ------------------------------------------- |
| **1**       | 0.5224 ± 0.4241  | 0.9214 ± 0.0510 | 0.0510       | Strong separation; stable                   |
| **2**       | 0.5652 ± 0.4647  | 0.5069 ± 0.1384 | 0.1384       | More variable LSA distribution              |
| **3**       | 0.5349 ± 0.4374  | 0.8856 ± 0.0520 | 0.0520       | Similar to Artist 1                         |
| **4**       | 0.5274 ± 0.4297  | 0.9124 ± 0.0510 | 0.0510       | Very stable high-quality embedding          |
| **5**       | 0.5179 ± 0.4192  | 0.2575 ± 0.1994 | 0.1994       | LSA distances poorly spread for this artist |
| **6**       | 0.5444 ± 0.4467  | 0.2907 ± 0.1972 | 0.1972       | Similar to Artist 5 (clusters collapsed)    |
| **7**       | 0.4915 ± 0.3869  | 0.3481 ± 0.1999 | 0.1999       | Noisy distribution                          |
| **8**       | 0.5686 ± 0.4685  | 0.9232 ± 0.0554 | 0.0554       | Very clean, stable LSA distances            |
| **9**       | 0.5478 ± 0.4498  | 0.3877 ± 0.1917 | 0.1917       | Moderate performance                        |


## How well do these data correlate?

| LSA Dim (k) | Pearson r (range across artists) | Spearman r (range across artists) | Interpretation                                          |
| ----------- | -------------------------------- | --------------------------------- | ------------------------------------------------------- |
| **10**      | 0.007 → 0.034                    | 0.078 → 0.170                     | Very weak; LSA underfits; genre signal barely detected  |
| **50**      | 0.292 → 0.399                    | 0.397 → 0.486                     | Small-to-moderate linear/monotonic correlation          |
| **100**     | 0.260 → 0.447                    | 0.327 → 0.474                     | Moderate correlation; performance stabilizes            |
| **300**     | 0.223 → 0.476                    | 0.242 → 0.489                     | Moderate correlation; LSA best reflects genre structure |
| **500**     | 0.227 → 0.484                    | 0.221 → 0.465                     | Similar to k=300; stable, moderate relationships        |
| **800**     | 0.246 → 0.533                    | 0.273 → 0.486                     | Strongest correlations; diminishing returns beyond this |

## Conclusions

While there is a non-trivial correlation between lyric LSA and genre, it does not seem to be sufficient to predict genre from lyrical content alone.
There is ongoing research about using machine learning to predict genre off of song content, but this uses much more sophisticated methods than simply
lyric analysis, though it does factor in.
ex. (Multimodal Deep Learning for Music Genre Classification (Oramas et al., 2018))

Notice that as the k value gets higher, artists tend to be more and more similar. This seems counter intuitive at first. But, my speculation
is that it is related to the IDF algorithm. This algorithm weights uncommon words a lot, while diminishing the effect of common words that show up everywhere.
Thus as k values get larger, the LSA vectors start to encode very common words, even though their weighted down, they are still present, and with a high enough
k value, they are going to be encoded. These words show up in pretty much all songs in high quanity, making all songs,
with respect to there word usage, atleast somewhat similar with them present in the data. See table below for aproximate weighting.

## Top 20 out of 430,703,936 different mentioned tokens
| word | count      | percent   |~weight |  
| ---- | ---------- | --------- | ------ |
| i    | 51,194,268 | 4.496792% | 0.287  |
| the  | 42,139,306 | 3.703446% | 0.315  |
| you  | 33,740,687 | 2.964660% | 0.345  |
| to   | 24,588,582 | 2.160015% | 0.383  |
| and  | 22,104,536 | 1.942804% | 0.396  |
| a    | 21,763,634 | 1.911848% | 0.398  |
| it   | 18,049,559 | 1.585547% | 0.421  |
| my   | 15,419,594 | 1.354934% | 0.437  |
| me   | 14,911,646 | 1.309794% | 0.440  |
| in   | 14,279,648 | 1.254260% | 0.444  |
| that | 12,623,881 | 1.108612% | 0.454  |
| of   | 12,601,138 | 1.106625% | 0.454  |
| n't  | 11,889,248 | 1.044361% | 0.458  |
| do   | 10,547,418 | 0.926576% | 0.466  |
| 's   | 9,953,656  | 0.874500% | 0.469  |
| on   | 9,711,829  | 0.853105% | 0.471  |
| we   | 8,674,277  | 0.761903% | 0.477  |
| 'm   | 8,589,188  | 0.754439% | 0.478  |
| your | 8,086,510  | 0.710202% | 0.481  |
| is   | 7,813,084  | 0.686664% | 0.483  |

## Top 20 words of English generally for comparison
| Rank | Word | Approx % of all words |
| ---- | ---- | --------------------- |
| 1    | the  | **5.0%**              |
| 2    | be   | **1.0%**              |
| 3    | to   | **1.0%**              |
| 4    | of   | **1.0%**              |
| 5    | and  | **0.9%**              |
| 6    | a    | **0.8%**              |
| 7    | in   | **0.7%**              |
| 8    | that | **0.6%**              |
| 9    | have | **0.5%**              |
| 10   | I    | **0.5%**              |
| 11   | it   | **0.5%**              |
| 12   | for  | **0.4%**              |
| 13   | not  | **0.4%**              |
| 14   | on   | **0.4%**              |
| 15   | with | **0.3%**              |
| 16   | he   | **0.3%**              |
| 17   | as   | **0.3%**              |
| 18   | you  | **0.3%**              |
| 19   | do   | **0.2%**              |
| 20   | at   | **0.2%**              |








