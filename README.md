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
1. Much easier said than done. Consider this matrix, the dimmensions are the number of different terms (n), by the number
of artists (m) ~(16,000,000 X 700,000). A nieve implementation of SVD would be basically computationally impossible, so this program
uses the Python SciKit package's Truncated SVD, with Fit Transform. Instead of computing the entire transform, then reducing
to the number of singular values (k). This function immediately reduces the matricies to (k X n), orders of magnitude more workable
than the full matrix.

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
and 90 is so pop.

To get "similarity" we use cosine similarity. Given two unit vectors, their cos similarity is calculated simply through
the dot product. This gives a number between 0 and 1; intuitively a score of how parallel they are, more riggorously,
it is the cosine of the angle between the two vectors. When the vectors are perpendicular, they have a shared genre score
of 0. This could happen if one artist were 100% rock and another were 100% pop. Whereas, if this number is 1, the artists have
100% the same genre.

LSA vectors are very similar, but they are less intuitive because instead of being defined through genres, each scalar in the
vector is associated with a latent variable inside of the data. These latent variables are nebulous variables that have some
kind of meaning, but it is unclear what they mean in a vacuum. But through the same process, cosine similarity, the vectors can
be compared through the dot product. The higher it is, the more similar the lyrical content of two artists are. Note all of these
vectors in the data are normalized aswell.

## Step 6:
The research question this entire process is built around is trying to compare Latent Semantic Analysis difference to genre difference..
So how do we test this?

We can start by taking an artist and plotting its similarity to each other artist on a scatterplot. On the X axis we can put
genre similarity and the Y axis we can put LSA similarity. Drawing a least squared regression line through these data gives
a correlation, standard error, standard deviation, etc.

Running this same process for many different random artists, gives some kind of average correlation between artists.
Ideally, this would be ran for every artist, but this is too computationally expensive (700,000 artists ran on 700,000 artists
is about 4.9*10^11 operations). So instead we sample a subset of these artists, (1000 in this case).

Elaborate on multiple correlation stats.

## Step 7:
Note in the core idea of this assignment is different k values for svd. This analysis was ran with k values (10,50,100,300,500,800)
These correlation values are measured against different k values to see how different levels of compression of the semantic data
results in different values of correlation.


# Data: 





