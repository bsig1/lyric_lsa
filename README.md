# Lyric LSA Project

**Higher Level Overview**
The point of this project is to see how the lyrics of an artist compare to there genre. The central hypothesis is that
*lyrical similarity* between two artists correlates to the *genre similarity* between two artists.

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

## Step 3:
Take the lyrics and associate them to artists.
- Group all lyrics in all tracks by artist. Resulting in a tally for each word for each artist with words merged together.

## Step 4:
Build lyric matrix
- Define a matrix whose rows are the terms and columns are artists
- Run SVD on this matrix
Much easier said than done. Consider this matrix, the dimmensions are the number of different terms (n), by the number
of artists (m) ~(16,000,000 X 700,000). A nieve implementation of SVD would be basically computationally impossible, so this program
uses the Python SciKit package's Truncated SVD, with Fit Transform. Instead of computing the entire transform, then reducing
to the number of singular values (k). This function immediately reduces the matricies to (k X n), orders of magnitude more workable
than the full matrix.
Heres how it works:

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

8. Return embedding: Z = U_k Σ_k. This is what is stored. Note that Z = U_k​Σ_k​= XV_k.

In the context of this problem, V represents the term embeddings of each singular value, while U represents the artist embeddings.
The vector stored for each artist in the artist embeddings weighted by the corresponding singular values.

## Step 5:
Define Genre vectors
- For each artist define a "genre vector" these vectors could be something like
(rock,pop,alternative)
(20%,10%,70%)
Then normalize it under the 2 norm for comparison between them.
- An artists genre is defined by the ratio of how many of there songs are reported as what genre under the dataset.
- In the data set there are 6 genres, so every vector is of length 6, these genres are:
(rock,pop,misc,rap,r&b,country)

## Step 5:
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

TODO: Elaborate on multiple correlation stats.

## Step 7:
Note in the core idea of this assignment is different k values for svd. This analysis was ran with k values (10,50,100,300,500,800)
These correlation values are measured against different k values to see how different levels of compression of the semantic data
results in different values of correlation.


# Data: 
TODO: Put in experiemental data.




